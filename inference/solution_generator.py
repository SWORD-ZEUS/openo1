import sys
import os
import torch
from typing import List, Dict, Optional, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.generator.generator import Generator  
from models.verifier.verifier_only_cls import Verifier
from transformers import GenerationConfig
from utils.prompts import VERIFIER_DATASET_PROMPT, GENERATOR_DATASET_PROMPT
from utils.global_funcs import load_config
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from utils.process_state_dict import process_state_dict
from dataset.dataset_verifier import VerifierModelDataset

class SolutionGenerator:
    def __init__(self, generator: Generator, verifier: Verifier):
        """
        Args:
            generator: 步骤生成器 
            verifier: 步骤验证器
        """
        self.generator = generator
        self.verifier = verifier
        self.generator.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.verifier.tokenizer.pad_token = "<|reserved_special_token_0|>"
        
        # 检测GPU设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        
        # 将模型移动到GPU
        self.generator = self.generator.to(self.device).to(self.dtype)
        self.verifier = self.verifier.to(self.device).to(self.dtype)
        
        # 验证结果到文本的映射
        self.rating_map = {
            -1: "错误",
            1: "终止", 
            0: "正确"
        }
        
        # 生成的配置
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.generator.tokenizer.pad_token_id,
            eos_token_id=self.generator.tokenizer.eos_token_id
        )
        # 复用VerifierModelDataset的patterns和end_token

        # 添加模式匹配相关的属性
        self.patterns = {
            'verifier': torch.tensor([128006, 424, 3125, 128007]),
            'step': torch.tensor([128006, 78191, 128007])
        }
        self.end_token = 128009

    def _find_pattern_positions(self, input_ids, pattern):
        """通用的模式匹配方法"""
        positions = []
        for i in range(len(input_ids) - len(pattern) + 1):
            if torch.all(input_ids[i:i+len(pattern)] == pattern):
                positions.append(i)
        return positions

    def _find_next_end_token(self, input_ids, start_pos):
        """在指定位置后查找结束标记"""
        end_positions = (input_ids[start_pos:] == self.end_token).nonzero(as_tuple=True)[0]
        return (start_pos + end_positions[0].item()) if len(end_positions) > 0 else None

    def _verify_segment(self, input_ids, pattern_key):
        """验证特定段落的完整性"""
        pattern = self.patterns[pattern_key]
        start_positions = self._find_pattern_positions(input_ids, pattern)
        
        if not start_positions:
            return None, None
            
        start_idx = start_positions[-1] + len(pattern)
        end_idx = self._find_next_end_token(input_ids, start_idx)
        
        if end_idx is None:
            return None, None
            
        return start_idx, end_idx

    def _get_step_indices_verifier(self, input_ids: torch.Tensor) -> Dict[str, int]:
        """获取step的索引"""
        start_idx, end_idx = self._verify_segment(input_ids, 'step')
        if start_idx is None:
            raise ValueError("Could not find step pattern in input_ids")
            
        return {'start': start_idx, 'end': end_idx}
    
    def generate_solution(self, question: str, max_steps: int = 20) -> Dict:
        """
        为给定问题生成解决方案
        
        Args:
            question: 输入的问题
            max_steps: 最大步骤数
            
        Returns:
            包含生成历史的字典
        """
        history = {
            "question": question,
            "steps": []
        }
        
        # 初始化messages
        messages_gen = [
            {"role": "system", "content": GENERATOR_DATASET_PROMPT},
            {"role": "user", "content": question}
        ]
        messages_verifier = [
            {"role": "system", "content": VERIFIER_DATASET_PROMPT},
            {"role": "user", "content": question}
        ]
        
        step_count = 0
        while step_count < max_steps:
            print(f"step_count:{step_count}")
            # 1. 生成下一个步骤
            next_step = self._generate_next_step(messages_gen)
            
            # 2. 验证步骤
            temp_verifier_messages = messages_verifier.copy()
            temp_verifier_messages.append({"role": "assistant", "content": next_step})
            rating, response = self._verify_step(temp_verifier_messages)
            
            # 3. 记录这一步
            step_record = {
                "step": next_step,
                "rating": self.rating_map[rating],
                "verifier_response": response
            }
            history["steps"].append(step_record)
            
            # 4. 处理验证结果
            if rating == 0:  # 正确
                messages_gen.append({"role": "assistant", "content": next_step})
                messages_verifier.append({"role": "assistant", "content": next_step})
                step_count += 1
            elif rating == 1:  # 终止
                break
            else:  # 错误，添加错误响应并重试
                messages_gen.append({"role": "assistant", "content": next_step})
                messages_gen.append({"role": "verifier", "content": f"{response}"})
                continue
                
        return history

    def _generate_next_step(self, messages: List[Dict]) -> str:
        """生成下一个步骤"""
        prompt = self.generator.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = self.generator.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)  # 移动到GPU
        inputs_id = inputs["input_ids"].squeeze()
        # print(f"inputs_id:{inputs_id}")

        with torch.no_grad():
            outputs = self.generator.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
            outputs_tokens = outputs[0]
            # print(f"outputs_tokens:{outputs_tokens}")
        generated_tokens = outputs[0][len(inputs_id):]  # 从输入中删除生成的部分
            
        generated_text = self.generator.tokenizer.decode(
            generated_tokens.cpu(),  # 转回CPU进行解码
            skip_special_tokens=True
        )
        
        # 提取新生成的部分
        return generated_text

    def _verify_step(self, messages: List[Dict]) -> Tuple[int, str]:
        """验证生成的步骤"""
        verifier_messages = messages.copy()
        inputs = self.verifier.tokenizer.encode_plus(
            self.verifier.tokenizer.apply_chat_template(verifier_messages, tokenize=False),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1280
        ).to(self.device)

        # # 确保输入数据使用正确的类型和设备
        # inputs = {k: v.to(self.device).to(self.dtype) for k, v in inputs.items()}

        # 获取step的起止位置
        # indices = self._get_step_indices(
        #     inputs['input_ids'][0].cpu(),  # 临时移到CPU计算索引
        #     verifier_messages[-1]['content']
        # )
        input_ids = inputs['input_ids'].squeeze()
        indices = self._get_step_indices_verifier(input_ids.cpu())
        
        with torch.no_grad():
            outputs = self.verifier(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                step_start_idx=torch.tensor([indices['start']], device=self.device),
                step_end_idx=torch.tensor([indices['end']], device=self.device)
            )

        # 获取分类结果
        logits = outputs['logits']
        predicted_class = torch.argmax(logits, dim=-1).item() - 1  # -1,0,1

        # 生成verifier的响应（如果步骤错误）
        response = ""
        if predicted_class == -1:
            response = "This last step is incorrect. Please revise your approach."

        return predicted_class, response

    def _get_step_indices(self, input_ids: torch.Tensor, step_content: str) -> Dict[str, int]:
        """获取step在input_ids中的起止位置"""

        step_tokens = self.verifier.tokenizer.encode_plus(
            step_content, 
            return_tensors="pt",
        )
        step_length = len(step_tokens)
        
        # 在input_ids中查找step_tokens的位置
        for i in range(len(input_ids) - step_length + 1):
            if torch.all(input_ids[i:i+step_length] == torch.tensor(step_tokens)):
                return {
                    'start': i,
                    'end': i + step_length
                }
                
        raise ValueError("Could not find step in input_ids")

def run_inference():
    # 初始化生成器配置
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_config", type=str, default="/zhuangkai/openo1/configs/sft_config.yaml")
    parser.add_argument("--verifier_config", type=str, default="/zhuangkai/openo1/configs/verifier_config_lora_cls.yaml")
    parser.add_argument("--load_verifier_trained_weights", action="store_true")
    args = parser.parse_args()
    generator_config = load_config(args.generator_config)
    model_path_gen = os.path.join(generator_config['download_model_dir'], generator_config['model_name'])
    generator_config["model_path"] = model_path_gen
    generator_config["fine_tuning"]["method"] = generator_config["test_settings"]["fine_tuning"]["method"]
    
    # 初始化验证器配置  
    verifier_config = load_config(args.verifier_config)
    model_path_verifier = os.path.join(verifier_config['download_model_dir'], verifier_config['model_name'])
    verifier_config["model_path"] = model_path_verifier
    verifier_config["fine_tuning"]["only_train_head"] = not args.load_verifier_trained_weights
    verifier_config["fine_tuning"]["method"] = verifier_config["test_settings"]["fine_tuning_method"]
    verifier_config["is_test"] = True

    # 初始化模型时指定GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化生成器
    generator = Generator(generator_config, training=False)
    generator = generator.to(device)
    
    # 初始化验证器
    verifier = Verifier(verifier_config, training="verifier")
    verifier = verifier.to(device)

    if args.load_verifier_trained_weights:
        trained_weights_path = verifier_config['test_settings']['load_trained_weights_path']
        print(f"Loading trained weights from {trained_weights_path}")
        client_sd = get_fp32_state_dict_from_zero_checkpoint(trained_weights_path)
        processed_sd = process_state_dict(client_sd)
        try:
            verifier.load_state_dict(processed_sd, strict=True)
            verifier = verifier.to(device)  # 确保权重加载后也在GPU上
            print("\n成功加载处理后的权重")
        except Exception as e:
            print(f"\n加载权重时出错: {str(e)}")
            raise e
    
    # 创建解题生成器
    solution_gen = SolutionGenerator(generator, verifier)
    
    # 生成方案
    question = "solve for x: $x^2 + 2x + 1 = 0$"
    solution = solution_gen.generate_solution(question)
    
    # 打印结果
    print(f"Question: {solution['question']}")
    for i, step in enumerate(solution['steps']):
        print(f"\nStep {i+1}:")
        print(f"Content: {step['step']}")
        print(f"Rating: {step['rating']}")
        print(f"Verifier Response: {step['verifier_response']}")

    # messages_verifier = [
    #         {"role": "system", "content": VERIFIER_DATASET_PROMPT},
    #         {
    #         "role": "user",
    #         "content": "The area of triangle $ABC$ is equal to $a^2 - (b - c)^2,$ where $a,$ $b,$ and $c$ are the sides of triangle $ABC,$ as usual.  Compute $\\tan A.$"
    #         },
    #         {
    #         "role": "assistant",
    #         "content": "It's well-known that the area of a triangle is equal to half the product of its base and height."
    #         },
    #         {
    #         "role": "assistant",
    #         "content": "So we can write the area of triangle $ABC$ as $\\frac{1}{2}bc\\sin A$."
    #         },
    #         {
    #         "role": "assistant",
    #         "content": "On the other hand, we can write the area as $a^2 - (b - c)^2$."
    #         },
    #         {
    #         "role": "assistant",
    #         "content": "So, we have the equation $\\frac{1}{2}bc\\sin A = a^2 - (b - c)^2$."
    #         },
    #         {
    #         "role": "assistant",
    #         "content": "But we want to solve for $\\tan A$."
    #         },
    #         # {
    #         # "role": "assistant",
    #         # "content": "We can use the identity $\\tan A = \\frac{\\sin A}{\\cos A}$."
    #         # },
    #         # {
    #         # "role": "assistant",
    #         # "content": "So it remains to find $\\cos A$."
    #         # },
    #         # {
    #         # "role": "assistant",
    #         # "content": "We know that in any triangle $ABC$, $\\cos A = \\frac{b^2 + c^2 - a^2}{2bc}$."
    #         # },
    #         # {
    #         # "role": "assistant",
    #         # "content": "So let's plug that into our equation."
    #         # }
    #     ]
    # rating, response = solution_gen._verify_step(messages_verifier)
    # print(f"rating:{rating}")
    # print(f"response:{response}")

if __name__ == "__main__":
    run_inference()