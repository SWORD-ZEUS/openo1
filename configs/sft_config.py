class SFTConfig:
    def __init__(self):
        self.model_name = "llama3-7b"  # 或其他预训练模型
        self.learning_rate = 5e-5
        self.batch_size = 16
        self.num_epochs = 3
        self.warmup_steps = 100
        self.total_steps = 10000  # 根据数据集大小和 epoch 数调整
        self.max_length = 512
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
