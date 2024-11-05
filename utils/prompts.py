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