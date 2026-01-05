from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential


gpt_api_key = ""
gpt_base_url = ""

gpt_client = OpenAI(
    api_key=gpt_api_key,
    base_url=gpt_base_url
)


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def obtain_response_gpt(inputs, model='gpt-4.1-2025-04-14'):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": inputs},
            ],
        },
    ]
    chat_completion = gpt_client.chat.completions.create(messages=conversation, model=model,
                                                         temperature=0.0
                                                         )
    output = chat_completion.choices[0].message.content
    return output


qwen_api_key = ""
qwen_base_url = ""

qwen_client = OpenAI(
    api_key=qwen_api_key,
    base_url=qwen_base_url
)


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def obtain_response_qwen(inputs, model='qwen3-max'):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": inputs},
            ],
        },
    ]
    chat_completion = qwen_client.chat.completions.create(messages=conversation, model=model,
                                                          temperature=0.0
                                                          )
    output = chat_completion.choices[0].message.content
    return output
