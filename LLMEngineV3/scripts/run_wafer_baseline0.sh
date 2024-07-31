python run.py \
    applications.0.scheduler=token_jsq \
    start_state=baseline \
    trace.filename=test_trace \
    seed=0

python run.py \
    applications.0.scheduler=token_jsq \
    start_state=baseline \
    trace.filename=AzureLLMInferenceTrace_code \
    seed=0

python run.py \
    applications.0.scheduler=token_jsq \
    start_state=baseline \
    trace.filename=AzureLLMInferenceTrace_conv \
    seed=0


    #+experiment=traces_light \
