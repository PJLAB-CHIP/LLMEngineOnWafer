# LLMEngineOnWafer

Large Language Model Inference Engine on Wafer-Scale Chip

# Setup

```bash
conda create -n splitwise-sim python=3.11
conda activate splitwise-sim
cd LLMEngineV3
pip install -r requirements.txt
```

# DONE

1. [X] 流程跑通
2. [X] 接入德浩的静态分析器(decode修改成了batch版本)，同时HACK实现了逆天数据的时间估计
3. [X] schduler修改(已优化逻辑)
4. [X] DRAM内存分配问题，已根据输出数据画出token instance的kv cache 峰值
5. [X] 将`static_mapper()`换成了查表，避免每次都profile


# WARNING !!!
1. [] llama2-70b模型kv cache peak占用很大，可能考虑改架构

# TODO

1. [ ] 确定prefill和decode的tp粒度以及instance个数
2. [ ] prefill和decode 初始化位置(目前是按顺序简单分配)，**根据跳数计算kv cache传输用时**
3. [ ] 功耗如何计算
4. [ ] 资源交换算法设计**(优先级最后，splitwise_aa在跑code数据集的时候基本上也拼不了batch并且没出现混合池)**
5. [ ] 是否要支持可变tp

# V3

1. 重写了 `KVJSQScheduler`，增加了 `pre_sel_batch()` 函数以及重写了 `schedule()` 函数以支持我们的调度策略。使用时将  `configs/applications/solo.yaml` 中换成 `scheduler: kv_jsq`
2. 重写了 `ORCAInstance` ，增加了 `max_batch_tokens `限制以及重写了 `select_batch()` , 同时 `SplitwiseInstance` 调用 ` ORCAInstance.select_batch()` 以支持我们的策略（无抢占）目前跑code数据集有bug（已经hack过去实现功能），看代码注释
3. 优化了逻辑，乱序遍历，并且`pending_tokens=0`时直接选择