GLM_JSON_RESPONSE_PREFIX = """You should always follow the instructions and output a valid JSON object.
The structure of the JSON object you can found in the instructions, use {"answer": "$your_answer"} as the default structure
if you are not sure about the structure.

And you should always end the block with a "```" to indicate the end of the JSON object.

<instructions>
"""

GLM_JSON_RESPONSE_SUFFIX = """Output:
</instructions>

"""
system_prompt = """The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format. 

EXAMPLE INPUT: 
Which is the highest mountain in the world? Mount Everest.

EXAMPLE JSON OUTPUT:
{
    "question": "Which is the highest mountain in the world?",
    "answer": "Mount Everest"
}

"""

# 重写步骤的 prompt
REWRITE_STEP_PROMPT = """请重写下面这个解题步骤，使其变成一个错误的步骤。要求：
1. 保持原始步骤的基本结构和长度
2. 引入合理的错误，使其看起来可能是对的，但实际是错的
3. 不要改变太多，只需要修改关键部分使其变得错误
4. 返回JSON格式，键为'rewritten_step'"""


VERIFIER_TASK_PROMPT = """You are a math problem verifier. A student is trying to solve a math problem and has made a mistake in their solution step. Your task is to:

1. Analyze the wrong solution step in the context of the problem
2. Identify errors in reasoning or calculation
3. Provide constructive feedback that guides the student towards the correct solution
4. Keep your response focused and concise
5. Do not directly give away the correct answer

Your feedback will be used to help the student improve their solution.

Return your response in JSON format with "verifier_response" as the key.

Example response format:
{
    "verifier_response": "Your feedback message here"
}"""

VERIFIER_DATASET_PROMPT = """You are a math problem verifier. A student is trying to solve a math problem. 
Here are the problem and the steps the student has taken so far. 
Each step starts with '<|start_header_id|>assistant<|end_header_id|> and ends with '<|eot_id|>'.
Your task is to assess the quality and correctness of the last step.
If the step is correct, your response should be "Yes, continue".
If the step is the termination of the solution, your response should be "Yes, terminate".
If the step is wrong, you should:
1. Analyze the wrong solution step in the context of the problem
2. Identify errors in reasoning or calculation
3. Provide constructive feedback that guides the student towards the correct solution
4. Keep your response focused and concise
5. Do not directly give away the correct answer
"""

VERIFIER_DATASET_PROMPT_ONLY_WRONG = """You are a math problem verifier. A student is trying to solve a math problem and has made a mistake in the last solution step. Your task is to:

1. Analyze the wrong solution step in the context of the problem
2. Identify errors in reasoning or calculation
3. Provide constructive feedback that guides the student towards the correct solution
4. Keep your response focused and concise
5. Do not directly give away the correct answer

Your feedback will be used to help the student improve their solution.

Here are the problem and the steps the student has taken so far. 
Each step starts with '<|start_header_id|>assistant<|end_header_id|> and ends with '<|eot_id|>'.
"""

GENERATOR_DATASET_PROMPT = """You are a helpful assistant. For each question, provide only one step of the solution at a time. After giving each step, wait for the next prompt before continuing.
"""