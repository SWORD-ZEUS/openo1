import openai
import time
from openai import OpenAI
from zhipuai import ZhipuAI
import requests

def get_oai_completion(prompt):
    api_pools = [
        # ("sk-sgiRF4epx9wYhTOqFeC493E5F41d453c994fB65a7dAdC9C2", "https://api.kwwai.top/v1", "gpt-4o-mini-2024-07-18"),
        # ("sk-vrcdzbOjXs5VCoyJ3bDdBaCeC95444C7Bf471aCdC0EcCd9f", "https://api.kwwai.top/v1", "gpt-4o-mini-2024-07-18"),
        # ("699b3308c248b557cc0c2985cda9f1eb.ag3dNoEPMuXzrPKz",None,"GLM-4-FLASH"),
        ("sk-3164d554bb6f4a41a5d12821fc151035","https://api.deepseek.com","deepseek-chat"),
        # ("699b3308c248b557cc0c2985cda9f1eb.ag3dNoEPMuXzrPKz",None,"GLM-4-PLUS"),
        # ("699b3308c248b557cc0c2985cda9f1eb.ag3dNoEPMuXzrPKz",None,"GLM-4-AIR")

    ]
    api = api_pools[0]
    api_key, base_url, model = api
    if "GLM" in model:
        client = ZhipuAI(api_key=api_key)
    else:
        client = OpenAI(api_key=api_key, base_url=base_url)

    # client = OpenAI(api_key="sk-sgiRF4epx9wYhTOqFeC493E5F41d453c994fB65a7dAdC9C2", base_url="https://api.kwwai.top/v1")
    # client = OpenAI(api_key="sk-vrcdzbOjXs5VCoyJ3bDdBaCeC95444C7Bf471aCdC0EcCd9f", base_url="https://api.kwwai.top/v1")
    try: 
        # print(1)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format={ "type": "json_object" },
            stream=False
        )
        # print(2)

        res = response.choices[0].message.content
        # res = response["choices"][0]["message"]["content"]
       
        gpt_output = res
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except Exception as e:
        print(e)
        # exit()



def call_chatgpt(ins):
    success = False
    re_try_count = 5
    ans = ''
    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion(ins)
            success = True
        except Exception as e:
            print(f"Retry times: {re_try_count}; Error: {e}", flush=True)
            time.sleep(5)
    return ans