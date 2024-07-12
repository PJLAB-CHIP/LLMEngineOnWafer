python run.py \
    applications.0.scheduler=kv_jsq \
    start_state=splitwise \
    trace.filename=test_trace \
    seed=0

python run.py \
    applications.0.scheduler=kv_jsq \
    start_state=splitwise \
    trace.filename=AzureLLMInferenceTrace_code \
    seed=0

    
python run.py \
    applications.0.scheduler=kv_jsq \
    start_state=splitwise \
    trace.filename=AzureLLMInferenceTrace_conv \
    seed=0

    #applications.0.scheduler=token_jsq \
    #trace.filename=rr_code_70 \
