#!/bin/bash

cd /zhuangkai
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openo1

# 函数：从YAML文件中提取指定键的值
get_yaml_value() {
    local yaml_file=$1
    local key=$2
    grep "^$key:" "$yaml_file" | sed 's/.*: *//'
}

# 设置配置文件路径
CONFIG_FILE="/zhuangkai/openo1/configs/verifier_config_adapter.yaml"

# 从配置文件中读取gpus_per_node的值
GPUS_PER_NODE=$(get_yaml_value "$CONFIG_FILE" "gpus_per_node")

# 如果无法从配置文件中读取值，则使用默认值
if [ -z "$GPUS_PER_NODE" ]; then
    echo "警告：无法从配置文件中读取gpus_per_node，使用默认值2"
    GPUS_PER_NODE=2
fi
echo "使用配置文件: $CONFIG_FILE"
# 设置其他变量
NUM_NODES=1
export NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT="12320"

echo "使用 GPUS_PER_NODE: $GPUS_PER_NODE"

# 使用torchrun启动脚本
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /zhuangkai/openo1/scripts/run_verifier.py \
    --config $CONFIG_FILE
