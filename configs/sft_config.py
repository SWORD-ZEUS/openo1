class SFTConfig:
    def __init__(self):
        self.download_model_dir = "/storage/zhuangkai/data/openo1/download_model_weights"
        self.model_name =  "Meta-Llama-3.1-8B-converted"  # 指向新的模型路径
        self.learning_rate = 5e-5
        self.batch_size = 1
        self.num_epochs = 3
        self.warmup_steps = 100
        self.total_steps = 10000  # 根据数据集大小和 epoch 数调整
        self.max_length = 512
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.val_interval = 0.25  # Validate every 1/4 epoch
        self.weight_save_path = "/storage/zhuangkai/data/openo1/weights/sft"
        self.train_data_path = "/zhuangkai/openo1/outputs/prm800k/phase2_train_gt_pre_generated_solution_to_step_list.jsonl"
        self.test_data_path = "/zhuangkai/openo1/outputs/prm800k/phase2_test_gt_pre_generated_solution_to_step_list.jsonl"

        # Wandb 配置
        self.use_wandb = True
        self.wandb_project = "openo1_sft"
        self.wandb_entity = "sota"  # 替换为Wandb 用户名
        self.wandb_api_key = "local-dcbced16c903a79158b4ccce5f5b2af5b2d53b75"  # 替换为Wandb API 密钥
        self.wandb_name = "sft_run"  # 为本次运行指定一个名称
        self.wandb_tags = ["sft", "llama3", "8b"]  # 添加标签
        self.wandb_notes = "Supervised fine-tuning of Llama 3 8B model"  # 添加描述
        self.wandb_group = "sft_experiments"  # 指定分组
        self.wandb_job_type = "training"  # 指定作业类型
        self.wandb_dir = "/storage/zhuangkai/data/openo1/wandb"  # 指定wandb文件存储目录

