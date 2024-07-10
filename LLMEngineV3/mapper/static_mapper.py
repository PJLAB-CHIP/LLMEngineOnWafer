from .mapper import manual_mapper
from .mapper_decode import decode_mapper
from . import transformer_block as tbk
from .arch_execution import Tx8
import math
from .util import load_config, find_powers_of_two, find_powers_of_two_nearest
from task import TaskType, PromptTask, TokenTask
import ipdb
import os


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


def get_static_mapper_duration(batch, instance):
    model_name = instance.model.name
    tile = instance.processors[0]
    die_NOC = tile._die.d2d_bw
    tile_num = len(instance.processors)
    die_num = tile_num / 144 if tile_num / \
        144 >= 1 else 1  # HACK: 暂定用tile_num/144来估计，不足1就设为1
    prompt_tasks = []
    token_tasks = []
    batch_tokens = 0
    # 统计该batch的token总数
    for task in batch:
        if isinstance(task, PromptTask):
            prompt_tasks.append(task)
            batch_tokens += task.request.prompt_size
        elif isinstance(task, TokenTask):
            token_tasks.append(task)  # 已经生成的token数+1
            batch_tokens += 1
        else:
            raise NotImplementedError

    if len(prompt_tasks) == len(batch):  # pure prefill batch
        # print(f'prompt batch, len(batch): {len(batch)}, batch_tokens: {batch_tokens}')
        prompt_time = prefill_static_mapper(
            batch_tokens, die_num, die_NOC, tile_num, model_name)
        return prompt_time
    elif len(token_tasks) == len(batch):  # pure decode batch
        # print(f'token batch, len(batch): {len(batch)}, batch_tokens: {batch_tokens}')
        kv_list = [token_task.request.prompt_size + token_task.request.generated_tokens
                   for token_task in token_tasks]  # 获取每个task的kv长度
        # print(f'KV list: {kv_list}')
        # ipdb.set_trace()
        decode_time = batch_decode_static_mapper(
            kv_list, die_num, die_NOC, tile_num, model_name)
        return decode_time
    else:  # 没有混合池策略
        raise NotImplementedError


def prefill_static_mapper(input_len, die_num, die_NOC, tile_num, model_name="llama2_70b"):
    die_num = int(die_num)
    prefill_len = find_powers_of_two_nearest(input_len)
    if prefill_len < 256:  # 256长度才能达到访存瓶颈
        prefill_name = f"prefill_256.json"
    # elif prefill_len > 4096:
    else:
        prefill_name = f"prefill_{prefill_len}.json"

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = f"./output/{model_name}/die-{die_num}-tile-{tile_num}/prefill"

    prefill_output_path = os.path.join(
        cur_dir, os.path.join(output_folder, prefill_name))

    prefill_input_path = os.path.join(
        cur_dir, f"./input/{model_name}/prefill/{prefill_name}")

    llm_config = load_config(prefill_input_path)
    # 默认只采用TP并行，获取切分后的权重大小
    llm_config, comm_time = megatron_shard(llm_config, die_num, die_NOC)

    mapping_result = load_config(prefill_output_path)

    # 单个block，单个die的结果
    comp_time = mapping_result["TotalLayer"]["latency"]

    # DONE: 对比了prefill不切分的执行时间，近似为1/die_num
    tot_time = comp_time + comm_time
    # scale power 2 input into origin input
    if prefill_len < 256:
        final_time = tot_time
    else:
        final_time = tot_time * (1.0*input_len/prefill_len)
    return final_time


def batch_decode_static_mapper(kv_list, die_num, die_NOC, tile_num, model_name="llama2_70b"):
    # ipdb.set_trace()
    # 只取出第一个用户的KV len值
    # HACK: 后面会把json文件里的KV换成kv_list里面的值, 这里随便读一个
    die_num = int(die_num)
    tile_num = int(tile_num)

    KV_len = 31  # kv_list[0]
    decode_len = find_powers_of_two_nearest(KV_len+1)
    decode_name = f"decode/decode_1_prefill_{decode_len-1}.json"

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    decode_input_path = os.path.join(
        cur_dir, f"./input/{model_name}/{decode_name}")

    # NOTE: only use for profile, no need run manual_mapper()
    decode_baseline_path = os.path.join(
        cur_dir, f"output/{model_name}/die-{die_num}-tile-{tile_num}/prefill")

    llm_config = load_config(decode_input_path)
    tx8_config = load_config(os.path.join(cur_dir, './tile_parameter.json'))
    tx8_config['TILE_NUM'] = tile_num
    hardware = Tx8(tx8_config)
    # 默认只采用TP并行，获取切分后的权重大小
    llm_config, comm_time = megatron_shard(llm_config, die_num, die_NOC)

    len1 = find_powers_of_two_nearest(hardware.config["TILE_NUM"])
    len2 = find_powers_of_two_nearest(len(kv_list))
    prefill_len = max(len1, len2)
    # DONE：verify successfully 大概是prefill=32，不切分模型执行时间的 1/13倍！
    decode_baseline_path = os.path.join(
        decode_baseline_path, f"prefill_{prefill_len}.json")

    mapping_result = decode_mapper(  # NEW: 将hardware接口暴露在外面方便修改tx8_config里的tile_num
        kv_list, llm_config, hardware, decode_baseline_path, details=False)

    comp_time = mapping_result["TotalLayer"]["latency"]

    # DONE: 对比了不切分的执行时间，近似为1/die_num
    tot_time = comp_time + comm_time
    # scale power 2 input into origin input
    final_time = tot_time
    return final_time


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
