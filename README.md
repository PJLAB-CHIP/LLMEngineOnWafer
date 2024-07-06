# LLMEngineOnWafer
Large Language Model Inference Engine on Wafer-Scale Chip

# Setup
'''python
conda create -n splitwise-sim python=3.11
conda activate splitwise-sim
cd LLMEngineV1
pip install -r requirements.txt
'''

# DONE
1. 流程跑通
2. 接入德浩的静态分析器(decode修改成了batch版本)，同时HACK实现了逆天数据的时间估计

# TODO
1. schduler修改
2. 资源交换算法设计
3. DRAM内存分配问题
4. 是否要支持可变tp
5. prefill和decode 初始化位置(目前是按顺序简单分配)
