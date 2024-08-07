python run.py \
    applications.0.scheduler=kv_jsq \
    applications.0.model_architecture=llama2-7b \
    applications.0.model_size=llama2-7b-fp16 \
    cluster=half_half \
    cluster.servers.0.count=40 \
    cluster.servers.1.count=0 \
    start_state=splitwise \
    start_state.prompt.tensor_parallelism=512 \
    start_state.prompt.num_instances=8 \
    start_state.token.tensor_parallelism=32 \
    start_state.token.num_instances=32 \
    performance_model=db \
    trace.filename=AzureLLMInferenceTrace_conv \
    seed=0