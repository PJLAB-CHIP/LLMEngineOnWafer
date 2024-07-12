import json
from copy import deepcopy
from .arch_execution import Tx8
from .mapper import manual_mapper
from . import transformer_block as tbk
from .util import find_powers_of_two, find_powers_of_two_nearest


def load_config(input_path):
    # 读取json配置文件
    with open(input_path, 'r') as file:
        # 从文件中加载 JSON 数据
        config = json.load(file)
    return config


def decode_mapper(kv_list, llm_config, hardware, baseline_path, details):

    # baseline result for non-FlashAttn
    # baseline_path = "./result/llama2_7b/prefill/prefill_32_result.json"
    # tx8_config = load_config('tile_parameter.json')
    # hardware = Tx8(tx8_config)

    baseline_result = load_config(baseline_path)
    # # load prefill=4096 result
    # baseline_result = load_config(baseline_path)

    result = deepcopy(baseline_result)

    llama_config = deepcopy(llm_config)

    flashAttn_time = 0
    flashAttn_cp_time = 0
    flashAttn_energy = 0
    flashAttn_NoC_energy = 0
    flashAttn_DRAM_energy = 0
    flashAttn_Compute_energy = 0

    for KV_len in kv_list:
        S = llama_config["S"] = 1
        H_QKV = llama_config["H_QKV"]

        D_O = llama_config["D_O"]
        H_O = llama_config["H_O"]
        KV = 1+KV_len

        # load memory scale factor for origin K&V
        # S*H_QKV+KV*H_QKV: Q*K^T
        # S*KV + KV*H_QKV: P*O
        Memory_scale = (S*H_QKV+KV*H_QKV + S*KV + KV*H_QKV) * \
            1.0 / (S*D_O + D_O*H_O)

        Compute_scale = (S*H_QKV*KV + S*KV*H_QKV)*1.0 / (S*D_O*H_O)
        flashAttn_time += Memory_scale * result["Linear"]["latency"]
        flashAttn_energy += Memory_scale * result["Linear"]["total_energy"]
        flashAttn_NoC_energy += Memory_scale * result["Linear"]["NoC_energy"]
        flashAttn_DRAM_energy += Memory_scale * result["Linear"]["DRAM_energy"]
        flashAttn_Compute_energy += Memory_scale * result["Linear"]["Compute_energy"]

        flashAttn_cp_time += Compute_scale * result["Linear"]["cp_latency"]

    key = "Flashatten"

    result[key]["latency"] = flashAttn_time
    result[key]["total_energy"] = flashAttn_energy
    result[key]["NoC_energy"] = flashAttn_NoC_energy
    result[key]["DRAM_energy"] = flashAttn_DRAM_energy
    result[key]["Compute_energy"] = flashAttn_Compute_energy

    result[key]["cp_latency"] = flashAttn_cp_time
    #
    # result[key]["utilization"] *= scale

    tot_latency = 0
    tot_cp_latency = 0
    tot_utilization = 0
    total_energy = 0
    NoC_energy = 0
    DRAM_energy = 0
    Compute_energy = 0

    utilization = 0
    Layers = llama_config["L"]
    for key, item in result.items():
        try:
            tot_latency += item['latency']
            tot_cp_latency += item['cp_latency']
            tot_utilization += item['utilization']
            total_energy += item['total_energy']
            NoC_energy += item["NoC_energy"]
            DRAM_energy += item["DRAM_energy"]
            Compute_energy += item["Compute_energy"]

            if details:
                print('{:<15}, latency(ms)={:>10.6f}, energy = {:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format(
                    key, item['latency'], item['total_energy'], item['utilization']*100, item['cp_latency']))
        except:
            print('{:<15}, No suitable mapping result! '.format(key))
    utilization = tot_cp_latency/(tot_latency+1e-35)
    result['TotalLayer'] = {
        "latency": tot_latency*Layers/1000, "total_energy": total_energy, "NoC_energy": NoC_energy, "DRAM_energy":DRAM_energy, "Compute_energy": Compute_energy, 'utilization': utilization*100, 'cp_latency': tot_cp_latency*Layers}
    if details:
        print(result)

    return result


if __name__ == "__main__":

    tx8_config = load_config('tile_parameter.json')
    hardware = Tx8(tx8_config)
    # 最小输入长度为大于等于tile_num的2次幂
    min_len = find_powers_of_two_nearest(tx8_config['TILE_NUM'])
    llm_max_config = load_config(
        "input/llama2_70b/prefill/prefill_32.json")
    max_len = llm_max_config["max_pos"]

    prefill_len = find_powers_of_two(min_len, max_len)
    decode_name = "decode/decode_1_prefill_2047.json"

    input_path = f"./input/llama2_7b/{decode_name}"
    output_path = f"./output/llama2_7b/{decode_name}"

    llama_config = load_config(input_path)

    decode_mapper(llama_config, output_path)
