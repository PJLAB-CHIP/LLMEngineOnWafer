# LLMEngineOnWafer

Large Language Model Inference Engine on Wafer-Scale Chip

# Setup

```bash
conda create -n splitwise-sim python=3.11
conda activate splitwise-sim
cd LLMEngineV3
pip install -r requirements.txt
```
# Run command
```bash
python run.py trace.filename=AzureLLMInferenceTrace_code  
python run.py trace.filename=AzureLLMInferenceTrace_conv  
python run.py trace.filename=test_trace  
```
# DONE

1. [X] 流程跑通
2. [X] 接入德浩的静态分析器(decode修改成了batch版本)，同时HACK实现了逆天数据的时间估计
3. [X] schduler修改(已优化逻辑)
4. [X] DRAM内存分配问题，已根据输出数据画出token instance的kv cache 峰值
5. [X] 将`static_mapper()`换成了查表，避免每次都profile
6. [X] 开发了KV Cache峰值统计函数，发现decode中组batch合理，decode len太长了导致长期占据memory
7. [X] 开发了mapper中的能耗统计函数，已经接入到scheduler中
8. [X] 开发了mapper中的功耗的分解函数，已经接入到scheduler中
1. [X] 开发对应的baseline0版本，加速器架构，采用混合方式执行prefill和decode，证明加速器上需要分离式架构才能优化TTFT&TBT之类的！
1. [X] 添加了baseline0和splitwise对应的bash执行脚本
<!-- # WARNING !!!
1. [] decode中组batch太小了，导致kv占用太高了！ -->

# TODO
1. [ ] 添加llama2-7b、llama2-13b、opt-7b、opt-13b、opt-66B模型，修改对应的instance个数和tp粒度
1. [ ] 开发对应的baseline1版本，只有改加速器架构，完全复用默认splitwise的调度逻辑，证明直接迁移不合适！
1. [ ] 保存能耗及其分解功耗信息到splitwise中的csv中
1. [ ] prefill和decode 初始化位置(目前是按顺序简单分配)，**根据跳数计算kv cache传输用时**，资源交换要考虑位置
1. [ ] 资源交换算法设计**(优先级最后，splitwise_aa在跑code数据集的时候基本上也拼不了batch并且没出现混合池)**
<!-- 6. [ ] 是否要支持可变tp -->

# V3

1. 重写了 `KVJSQScheduler`，增加了 `pre_sel_batch()` 函数以及重写了 `schedule()` 函数以支持我们的调度策略。使用时将  `configs/applications/solo.yaml` 中换成 `scheduler: kv_jsq`
2. 重写了 `ORCAInstance` ，增加了 `max_batch_tokens `限制以及重写了 `select_batch()` , 同时 `SplitwiseInstance` 调用 ` ORCAInstance.select_batch()` 以支持我们的策略（无抢占）目前跑code数据集有bug（已经hack过去实现功能），看代码注释
3. 优化了逻辑，乱序遍历，并且`pending_tokens=0`时直接选择