# gpt-4.1 as an example

import json
from openai import OpenAI
import base64
import os
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_random_exponential


api_key = ""
base_url = ""

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def obtain_response(inputs):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": inputs},
            ],
        },
    ]
    chat_completion = client.chat.completions.create(messages=conversation, model='gpt-4.1-2025-04-14',
                                                     temperature=0.0
                                                     )
    output = chat_completion.choices[0].message.content
    return output


def main():
    with open(f'prompt.txt') as f:
        prompt = f.read()
    dataset_names = os.listdir('../misinformation/data')
    for dataset_name in dataset_names:
        claims = json.load(open(f'../misinformation/data/{dataset_name}'))
        evidence = json.load(open(f'../evidence/data/{dataset_name}'))
        save_path = f'data/{dataset_name}'
        if os.path.exists(save_path):
            out = json.load(open(save_path))
        else:
            out = []
        indices = list(range(len(claims)))
        for index in tqdm(indices[len(out):]):
            item = claims[index]
            each_evidence = evidence[index][3]
            each_explanation = []
            for _ in each_evidence:
                if item is None:
                    item = ' '
                if _ is None:
                    _ = ' '
                inputs = prompt.replace('<===c===>', item)
                inputs = inputs.replace('<===e===>', _)
                each_explanation.append(obtain_response(inputs))
            out.append(each_explanation)
            json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()
