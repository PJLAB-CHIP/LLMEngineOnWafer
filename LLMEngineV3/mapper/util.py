
import json
import pandas as pd
import math
from typing import Union
TB = 1024**4
GB = 1024**3
MB = 1024**2
KB = 1024
T = 1000**4
G = 1000**3
M = 1000**2
K = 1000


# 查找a到b范围内是不是有2次幂


def find_powers_of_two(a, b):
    # 找到大于等于a的最小2的次幂
    start = 1
    while start < a:
        start *= 2

    # 找到小于等于b的最大2的次幂
    powers_of_two = []
    while start <= b:
        powers_of_two.append(start)
        start *= 2

    return powers_of_two


def find_powers_of_two_nearest(a):
    # 找到最接近a的的2次幂
    upper_bound = 1
    while upper_bound < a:
        upper_bound *= 2
    return upper_bound if upper_bound <= 4096 else 4096  # HACK: 有些逆天数据


def load_config(input_path):
    # 读取json配置文件
    with open(input_path, 'r') as file:
        # 从文件中加载 JSON 数据
        config = json.load(file)
    return config


def save_file(data, file_path):
    # 转换为list+dict类型
    df = pd.DataFrame(data)
    # 将DataFrame保存到Excel中，index参数用于指定是否包含行索引
    df.to_excel(file_path, index=False)


def MBytes(list0, bytes=2):
    if list0 == None or list0 == 0:
        return 0
    # List 维度乘积
    res = 1
    for i in list0:
        res *= i
    return res/1024/1024*bytes


def min_multiple_of(num, factor=16):
    return math.ceil(num/factor)*factor


def dim_norm(dims, tile_num=16):
    # 将维度规则到tile_num的倍数
    newdims = []
    for dim in dims:
        newdims.append(min_multiple_of(dim, tile_num))
    return newdims


def dim_analysis(optype, dims, para_dims):
    # 将gemm矩阵维度按并行切分度重新计算数据shape
    if optype == 'GEMM':  # [b,m,k,n]
        reduce = True if para_dims[2] > 1 else False
        newdims = [math.ceil(dims[0]/para_dims[0]), math.ceil(dims[1]/para_dims[1]),
                   math.ceil(dims[2]/para_dims[2]), math.ceil(dims[3]/para_dims[3])]
        i_shape = [math.ceil(newdims[0]), math.ceil(
            newdims[1]), math.ceil(newdims[2])]
        w_shape = [math.ceil(newdims[2]), math.ceil(newdims[3])]
        o_shape = [math.ceil(newdims[0]), math.ceil(
            newdims[1]), math.ceil(newdims[3])]
        return newdims, i_shape, o_shape, w_shape, reduce


def block_range(dim, min_block=1, max_block=None):
    # 遍历dim可以因式分解的所有公因子，满足大于等于min_block，且为min_block的倍数，且小于等于max_block
    if max_block == None:
        max_block = dim
    factors = []
    sqrt_n = int(math.sqrt(dim))
    for i in range(1, sqrt_n + 1):
        if dim % i == 0:
            if i % min_block == 0 and i <= max_block:
                factors.append(i)
            if i != dim // i:
                if dim // i % min_block == 0 and dim // i <= max_block:
                    factors.append(dim // i)
    return factors


if __name__ == "__main__":
    new_dims = dim_norm([16, 4096, 5, 511], factor=16)
    print(new_dims)
    for dim in new_dims:
        print(block_range(dim, min_block=16))
    print(dim_analysis('GEMM', new_dims, [1, 128, 64, 256]))
