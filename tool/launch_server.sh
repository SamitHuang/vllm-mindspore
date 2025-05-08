export ASCEND_RT_VISIBLE_DEVICES=5
export PYTHONPATH=$(pwd):$PYTHONPATH

# backend 
# unset vLLM_MODEL_BACKEND
# export vLLM_MODEL_BACKEND=MindFormers
export vLLM_MODEL_BACKEND=MindOne

export vLLM_MODEL_MEMORY_USE_GB=50
export ASCEND_TOTAL_MEMORY_GB=64
# export MS_ENABLE_LCCL=off
# export HCCL_OP_EXPANSION_MODE=AIV
# export HCCL_SOCKET_IFNAME=enp189s0f0
# export GLOO_SOCKET_IFNAME=enp189s0f0
# export TP_SOCKET_IFNAME=enp189s0f0
# export HCCL_CONNECT_TIMEOUT=3600
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# use MindONE
# export vLLM_MODEL_BACKEND=MindOne


python3 -m vllm_mindspore.entrypoints vllm.entrypoints.openai.api_server \
    --model /home/mikecheung/model/Qwen2.5-VL-3B-Instruct \
    --port 9529 \
    --max-num-seqs 12 \
    --max-model-len 32768 

    # --max-num-batched-tokens 2048 \
