# LLMEngineOnWafer
Large Language Model Inference Engine on Wafer-Scale Chip

# Setup
'''bash
conda create -n splitwise-sim python=3.11
conda activate splitwise-sim
cd LLMEngineV1
pip install -r requirements.txt
'''

# DONE
1. 流程跑通
2. 接入德浩的静态分析器(decode修改成了batch版本)，同时HACK实现了逆天数据的时间估计
3. schduler修改

# TODO
1. DRAM内存分配问题，确定prefill和decode的tp以及instance数量后先看KVcache峰值
2. 资源交换算法设计
3. 是否要支持可变tp
4. prefill和decode 初始化位置(目前是按顺序简单分配)

# V3
1. 重写了'KVJSQScheduler'，增加了'pre_sel_batch()'函数以及重写了'schedule()'函数以支持我们的调度策略。使用时将'configs/applications/solo.yaml'中换成'scheduler: kv_jsq'
2. 重写了'ORCAInstance'，增加了'max_batch_tokens'限制以及重写了'select_batch()', 同时'SplitwiseInstance'在使用的是'KVJSQScheduler'时会调用'ORCAInstance'的'select_batch()'以支持我们的策略（无抢占）目前跑code数据集有bug（已经hack过去实现功能），看代码注释
3. 目前默认的DRAM容量在跑conv数据集时会出现超出内存的现象，可以先将'Instance'里面'self.max_memory = sys.maxsize'