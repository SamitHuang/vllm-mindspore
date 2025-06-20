export ASCEND_RT_VISIBLE_DEVICES=3

export PYTHONPATH=$(pwd):$PYTHONPATH
# set path to mindone
export PYTHONPATH=/home/hyx/vllm/mindone:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# w/o MF 
unset vLLM_MODEL_BACKEND

# use MF
# export vLLM_MODEL_BACKEND=MindFormers
# export MINDFORMERS_MODEL_CONFIG="tests/st/python/config/predict_qwen2_5_7b_instruct.yaml"

# use MindONE
# export vLLM_MODEL_BACKEND=MindOne

# LLM
python test_vllm.py
# MLLM
# python tool/offline_inference_blip2.py
