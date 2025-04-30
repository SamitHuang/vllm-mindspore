export ASCEND_RT_VISIBLE_DEVICES=7

export PYTHONPATH=/home/hyx/vllm/vllm-mindspore:$PYTHONPATH
# set path to mindone
export PYTHONPATH=/home/hyx/vllm/mindone:$PYTHONPATH

# w/o MF 
# unset vLLM_MODEL_BACKEND

# use MF
# export vLLM_MODEL_BACKEND=MindFormers
# export MINDFORMERS_MODEL_CONFIG="tests/st/python/config/predict_qwen2_5_7b_instruct.yaml"

# use MindONE
export vLLM_MODEL_BACKEND=MindOne

# export VLLM_TRACE_FUNCTION=1
# export VLLM_LOGGING_LEVEL=DEBUG

python offline_inference.py --model_path /home/hyx/models/Qwen/Qwen2.5-VL-3B-Instruct

