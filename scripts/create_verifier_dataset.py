import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mpr import MultipleProcessRunnerSimplifier
from utils.openai_access import call_chatgpt

class VerifierDatasetCreator(MultipleProcessRunnerSimplifier):
    def __init__(self, data_path, save_path, n_process=4, num_scale=None):
        # 读取数据
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading data"):
                try:
                    item = json.loads(line.strip())  # 添加strip()去除多余空白字符
                    problem = item['problem']
                    steps = item['steps']
                    
                    # 获取有wrong step的步骤
                    previous_correct_steps = []
                    for wrong_steps, correct_step in steps:
                        if wrong_steps:  # 如果有wrong steps
                            for wrong_step in wrong_steps:
                                data.append({
                                    'problem': problem,
                                    'previous_steps': previous_correct_steps.copy(),
                                    'current_wrong_step': wrong_step,
                                    'current_correct_step': correct_step
                                })
                        previous_correct_steps.append(correct_step)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON in line: {e}")
                    continue
                    
        length = len(data)
        super().__init__(
            data=data[:int(length * num_scale)] if num_scale is not None else data,
            do=self.do,
            save_path=save_path,
            n_process=n_process,
            verbose=True
        )

    def do(self, process_id, i, item, writer):
        # 构造prompt
        prompt = self._create_prompt(
            item['problem'],
            item['previous_steps'],
            item['current_wrong_step'],
            item['current_correct_step']
        )
        
        # 调用大模型
        try:
            response = call_chatgpt(prompt)
            response_json = json.loads(response)
            verifier_response = response_json.get('verifier_response', '')
            
            if verifier_response:
                #构造输出数据
                output = {
                    'problem': item['problem'],
                    'steps': item['previous_steps'] + 
                            [(item['current_wrong_step'], verifier_response)]
                }
                
                # 写入文件
                writer.write(json.dumps(output, ensure_ascii=False) + '\n')
                writer.flush()
        
        except Exception as e:
            print(f"Process {process_id}, Item {i} failed: {str(e)}")

    def _create_prompt(self, problem, previous_steps, current_wrong_step, current_correct_step):
        from utils.prompts import VERIFIER_TASK_PROMPT
        prompt = {
            "task": VERIFIER_TASK_PROMPT,
            "problem": problem,
            "previous_steps": previous_steps,
            "current_wrong_step": current_wrong_step,
            "current_correct_step": current_correct_step
        }
        return json.dumps(prompt, ensure_ascii=False)

    def _aggregate(self, final_path, sub_paths):
        """合并所有子文件的结果"""
        if final_path is not None:
            with open(final_path, 'w', encoding='utf-8') as fw:
                for sub_path in sub_paths:
                    if sub_path and os.path.exists(sub_path):
                        with open(sub_path, 'r', encoding='utf-8') as fr:
                            fw.write(fr.read())
                        os.remove(sub_path)
        return None

def main(args):
    # 创建输出目录
    os.makedirs(args.save_path, exist_ok=True)
    n_process = args.n_process
    num_scale = args.num_scale
    
    # 处理训练集、验证集和测试集
    for split in ['train', 'validation', 'test']:
        input_path = f'/zhuangkai/openo1/outputs/verifier/phase1_{split}.jsonl'
        output_path = f'{args.save_path}/phase1_{split}_verifier.jsonl'
        print(f"\nProcessing {split} set...")
        creator = VerifierDatasetCreator(
            data_path=input_path,
            save_path=output_path,
            n_process=n_process,
            num_scale=num_scale
        )
        creator.run()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--n_process', type=int, default=128)
    parser.add_argument('--save_path', type=str, default='/zhuangkai/openo1/outputs/verifier')
    parser.add_argument('--num_scale', type=float, default=None)
    args = parser.parse_args()
    main(args)