import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mpr import MultipleProcessRunnerSimplifier
from utils.openai_access import call_chatgpt

class OrderedDatasetCreator(MultipleProcessRunnerSimplifier):
    def __init__(self, data_path, save_path, n_process=4):
        # 读取数据
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                problem = item['problem']
                steps = item['steps']
                
                # 只处理rating为0的step
                for i, (step, rating) in enumerate(steps):
                    if rating == 0:
                        # 获取前置steps
                        previous_steps = [s for s, _ in steps[:i]]
                        data.append({
                            'problem': problem,
                            'previous_steps': previous_steps,
                            'current_step': step,
                            'all_steps': steps
                        })
        
        super().__init__(
            data=data,
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
            item['current_step']
        )
        
        # 调用大模型
        try:
            response = call_chatgpt(prompt)
            response_json = json.loads(response)
            rewritten_step = response_json.get('rewritten_step', '')
            
            if rewritten_step:
                # 构造输出数据
                output = {
                    'problem': item['problem'],
                    'steps': [step for step in item['previous_steps']] + 
                            [(item['current_step'], rewritten_step)]
                }
                
                # 写入文件
                writer.write(json.dumps(output, ensure_ascii=False) + '\n')
                writer.flush()
        
        except Exception as e:
            print(f"Process {process_id}, Item {i} failed: {str(e)}")

    def _create_prompt(self, problem, previous_steps, current_step):
        prompt = {
            "task": "请重写下面这个解题步骤，使其变成一个错误的步骤。要求：\n"
                   "1. 保持原始步骤的基本结构和长度\n"
                   "2. 引入合理的错误，使其看起来可能是对的，但实际是错的\n"
                   "3. 不要改变太多，只需要修改关键部分使其变得错误\n"
                   "4. 返回JSON格式，键为'rewritten_step'",
            "problem": problem,
            "previous_steps": previous_steps,
            "current_step": current_step
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

def main():
    # 创建输出目录
    os.makedirs('/zhuangkai/openo1/outputs/rm/ordered_dataset', exist_ok=True)
    n_process = 32
    
    # 处理训练集、验证集和测试集
    for split in ['train', 'validation', 'test']:
        input_path = f'/zhuangkai/openo1/dataset/prm800k/processed/rm/phase2_{split}_short.jsonl'
        output_path = f'/zhuangkai/openo1/outputs/rm/ordered_dataset/{split}_ordered.jsonl'
        
        print(f"\nProcessing {split} set...")
        creator = OrderedDatasetCreator(
            data_path=input_path,
            save_path=output_path,
            n_process=n_process  # 可以根据需要调整进程数
        )
        creator.run()

if __name__ == '__main__':
    main()
