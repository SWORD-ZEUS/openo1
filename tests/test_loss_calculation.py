import torch
import torch.nn as nn
import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestLossCalculation(unittest.TestCase):
    def setUp(self):
        # 初始化模型和tokenizer
        self.model_name = "/storage/zhuangkai/data/openo1/download_model_weights/Meta-Llama-3.1-8B-Instruct"  
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # 确保模型处于评估模式
        self.model.eval()

    def test_loss_calculation_with_ignored_labels(self):
        # 构造两组不同的logits，但在非-100标签位置保持相同
        logits_1 = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [3.0, 4.0, 5.0, 6.0, 7.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [7.0, 8.0, 9.0, 10.0, 11.0],
            [8.0, 9.0, 10.0, 11.0, 12.0]
        ]).unsqueeze(0)  # 添加batch维度

        logits_2 = torch.tensor([
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [20.0, 30.0, 40.0, 50.0, 60.0],
            [30.0, 40.0, 50.0, 60.0, 70.0],
            [4.0, 5.0, 6.0, 7.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
            [7.0, 8.0, 9.0, 10.0, 11.0],
            [80.0, 90.0, 100.0, 110.0, 120.0]
        ]).unsqueeze(0)  # 添加batch维度

        labels = torch.tensor([[-100, -100, -100, 3, 2, 1, 0, -100]])

        # 计算损失
        # loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss_fn = nn.CrossEntropyLoss()
        loss_1 = loss_fn(logits_1.view(-1, 5), labels.view(-1))
        loss_2 = loss_fn(logits_2.view(-1, 5), labels.view(-1))

        # 打印损失值
        print(f"Loss for logits_1: {loss_1.item()}")
        print(f"Loss for logits_2: {loss_2.item()}")

        # 验证两个损失值是否相等
        self.assertAlmostEqual(loss_1.item(), loss_2.item(), places=5)

        # 验证只有非-100的标签被考虑在内
        non_ignored_labels = labels[labels != -100]
        self.assertEqual(len(non_ignored_labels), 4)  # 只有4个非-100的标签

if __name__ == '__main__':
    unittest.main()
