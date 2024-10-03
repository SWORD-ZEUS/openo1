class SFTConfig:
    def __init__(self):
        self.download_model_dir = "/storage/zhuangkai/data/openo1/download_model_weights"
        self.model_name =  "Meta-Llama-3.1-8B-converted"  # 指向新的模型路径
        self.learning_rate = 5e-5
        self.batch_size = 2
        self.num_epochs = 3
        self.warmup_steps = 100
        self.total_steps = 10000  # 根据数据集大小和 epoch 数调整
        self.max_length = 512
        self.lora_r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.weight_save_path = "/Storage/zhuangkai/openo1/weights/sft"
        self.train_data_path = "/zhuangkai/openo1/outputs/prm800k/phase2_train_gt_pre_generated_solution_to_step_list.jsonl"
        self.test_data_path = "/zhuangkai/openo1/outputs/prm800k/phase2_test_gt_pre_generated_solution_to_step_list.jsonl"

