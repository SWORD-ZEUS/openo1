cd /zhuangkai
source /zhuangkai/anaconda3/etc/profile.d/conda.sh
conda activate openo1


# Set the number of nodes and GPUs per node
NUM_NODES=1
GPUS_PER_NODE=4

export NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT="12320"

# Launch the script with the specified number of nodes and GPUs per node
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /zhuangkai/openo1/scripts/run_sft.py