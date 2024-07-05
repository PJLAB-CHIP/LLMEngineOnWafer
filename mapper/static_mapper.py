from mapper import manual_mapper
from mapper_decode import decode_mapper
import transformer_block as tbk
from arch_execution import Tx8
import math
from util import load_config, find_powers_of_two, find_powers_of_two_nearest

MB = 8*1024*1024


def distribute_elements(elements, num_groups):
    groups = [[] for _ in range(num_groups)]
    for i, element in enumerate(elements):
        groups[i % num_groups].append(element)
    return groups


'''
    megatron shard llm with llm_config, die_num, die_NOC
    return sharded llm_config & comm_time
    DONE: verify successfully!
'''


def megatron_shard(llm_config, die_num, die_NOC):

    # 对于MLP层 A matrix采用按列切分，B matrix按行切分
    # 对于all-reduce 通信量是原矩阵大小的2倍，包括reduce-scatter & all-gather
    # 最后只需要对最终输出结果做all-reduce，输出大小为b*s*h_FD
    MLP_comm = 2*(llm_config["Q"]*llm_config["B"] *
                  llm_config['S']*llm_config['H_FD']/MB)
    MLP_time = MLP_comm / die_NOC / 1024
    # 修改输入的权重大小
    llm_config['H_FU'] = math.ceil(llm_config['H_FU']/die_num)
    llm_config['D_FD'] = math.ceil(llm_config['D_FD']/die_num)

    # 对于Self-Attn先按列切分qkv三个权重矩阵！然后切分head维度，然后按行切分o weight
    # 最后只需要对最终输出结果做all-reduce，输出大小为b*s*h_o
    SelfAttn_comm = 2 * (llm_config["Q"]*llm_config["B"] *
                         llm_config['S']*llm_config['H_O']/MB)
    SelfAttn_time = SelfAttn_comm / die_NOC / 1024
    tot_time = SelfAttn_time + MLP_time
    # shard QKV proj along column
    llm_config['H_QKV'] = math.ceil(llm_config['H_QKV']/die_num)
    # shard attention along head_dim
    llm_config["H_A"] = math.ceil(llm_config["H_A"] / die_num)
    llm_config["N_A"] = math.ceil(llm_config["N_A"] / die_num)

    # shard O proj along row
    llm_config['D_O'] = math.ceil(llm_config['D_O']/die_num)

    return llm_config, tot_time

# 针对batch prefill的场景，我们只需增加对应的prefill len即可

# 由于模型包括了L=32的配置，因此最终执行的结果是 单个Die执行32个block的结果了！
# NOTE: only single batch, no pipeline only tensor parallel
# 默认采用megatron策略，简单评估通信时间！ forward is 2X all-reduce, one is MLP, other is Self-Attn


def prefill_static_mapper(input_len, die_num, die_NOC, model_name="llama2_70b"):
    prefill_len = find_powers_of_two_nearest(input_len)
    prefill_name = f"prefill/prefill_{prefill_len}.json"

    prefill_input_path = f"./input/{model_name}/{prefill_name}"
    prefill_output_path = f"./output/{model_name}/{prefill_name}"

    llm_config = load_config(prefill_input_path)

    # 默认只采用TP并行，获取切分后的权重大小
    llm_config, comm_time = megatron_shard(llm_config, die_num, die_NOC)

    print(llm_config)
    llama7b = tbk.Llama_block(llm_config)

    tx8_config = load_config('tile_parameter.json')
    hardware = Tx8(tx8_config)
    # print(hardware.config)
    # preset 是否使用预设切分;details是否打印映射的详细信息
    mapping_result = manual_mapper(
        llama7b, hardware, output_path=prefill_output_path, preset=False, details=True)

    # 单个block，单个die的结果
    comp_time = mapping_result["TotalLayer"]["latency"]

    # DONE: 对比了prefill不切分的执行时间，近似为1/die_num
    tot_time = comp_time + comm_time
    # scale power 2 input into origin input
    final_time = tot_time * (1.0*input_len/prefill_len)
    return final_time


def decode_static_mapper(KV_len, die_num, die_NOC,  model_name="llama2_70b"):
    decode_len = find_powers_of_two_nearest(KV_len+1)
    decode_name = f"decode/decode_1_prefill_{decode_len-1}.json"

    decode_input_path = f"./input/{model_name}/{decode_name}"
    decode_output_path = f"./output/{model_name}/{decode_name}"

    llm_config = load_config(decode_input_path)

    # 默认只采用TP并行，获取切分后的权重大小
    llm_config, comm_time = megatron_shard(llm_config, die_num, die_NOC)
    mapping_result = decode_mapper(
        llm_config,  decode_output_path)

    comp_time = mapping_result["TotalLayer"]["latency"]

    # DONE: 对比了不切分的执行时间，近似为1/die_num
    tot_time = comp_time + comm_time
    # scale power 2 input into origin input
    final_time = tot_time * (1.0*(KV_len+1)/decode_len)
    return final_time


# 给定prefill长度和decode长度生成对应的prefill所需的die和decode所需的die


# def static_mapper_test(prefill_len, decode_len, KV_len, die_num, die_NOC):
#     prefill_time = prefill_static_mapper(prefill_len, die_num, die_NOC)

#     all_KV = find_powers_of_two(KV_len, KV_len+decode_len)

#     decode_time = 0
#     for i in range(len(all_KV)):
#         KV = all_KV[i] - 1
#         if i+1 < len(all_KV):
#             next_KV = all_KV[i+1] - 1
#         else:
#             next_KV = decode_len + KV_len
#         if i == 0:
#             decode_time += (next_KV - KV_len) * \
#                 decode_static_mapper(KV, die_num, die_NOC)
#         else:
#             decode_time += (next_KV - KV) * \
#                 decode_static_mapper(KV, die_num, die_NOC)

#     # # decode 相对划分的多一些资源
#     # factor = decode_time * 1.0 / (decode_time+prefill_time)

#     # # 四舍五入进行
#     # decode_die = round(die_num * factor)
#     # prefill_die = die_num - decode_die
#     return prefill_time, decode_time

# # 找到距离a最近的2^n


# def static_mapper(prefill_len, decode_len, KV_len, die_num, die_NOC):

#     prefill_time = prefill_static_mapper(prefill_len, die_num, die_NOC)

#     all_KV = find_powers_of_two(KV_len, KV_len+decode_len)

#     decode_time = 0
#     for i in range(len(all_KV)):
#         KV = all_KV[i] - 1
#         if i+1 < len(all_KV):
#             next_KV = all_KV[i+1] - 1
#         else:
#             next_KV = decode_len + KV_len
#         if i == 0:
#             decode_time += (next_KV - KV_len) * \
#                 decode_static_mapper(KV, die_num, die_NOC)
#         else:
#             decode_time += (next_KV - KV) * \
#                 decode_static_mapper(KV, die_num, die_NOC)

#     # # decode 相对划分的多一些资源
#     # factor = decode_time * 1.0 / (decode_time+prefill_time)

#     # # 四舍五入进行
#     # decode_die = round(die_num * factor)
#     # prefill_die = die_num - decode_die
#     return prefill_time, decode_time


if __name__ == "__main__":
    die_config = load_config('die_parameter.json')
    die_num = die_config["DIE_NUM"]
    die_NOC = die_config["NOC_BW(GB/s)"]

    tx8_config = load_config('tile_parameter.json')
    hardware = Tx8(tx8_config)
    # 最小输入长度为大于等于tile_num的2次幂
    min_len = find_powers_of_two_nearest(tx8_config['TILE_NUM'])
    llm_max_config = load_config(
        "input/llama2_70b/prefill/prefill_32.json")
    max_len = llm_max_config["max_pos"]

    prefill_len = find_powers_of_two(min_len, max_len)
    for len in prefill_len:
        for model_name in ["llama2_7b", "llama2_13b", "llama2_70b"]:
            prefill_static_mapper(len, die_num, die_NOC, model_name=model_name)

    for len in prefill_len:
        for model_name in ["llama2_7b", "llama2_13b", "llama2_70b"]:
            decode_static_mapper(len-1, die_num, die_NOC,
                                 model_name=model_name)

    # # 2048
    # prefill_len = 2048

    # decode_len = 20
    # KV_len = 2044
    # prefill len = 2044
    # # 21 = 1 + 20
    # static_mapper(prefill_len, decode_len, KV_len, die_num, die_NOC)