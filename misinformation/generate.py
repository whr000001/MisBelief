import json
import os
from tqdm import tqdm
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


def main():
    description = json.load(open('description.json'))
    with open('prompts.txt') as f:
        prompt = f.read()
    # print(description.keys())
    dataset_names = os.listdir('../datasets')

    for dataset_name in dataset_names:
        data = json.load(open(f'../datasets/{dataset_name}'))
        topic = dataset_name.replace('.json', '')
        topic = topic.replace('_', ' ')
        data = data[:200]  # for test
        for difficulty in ['easily', 'moderately', 'highly']:
            save_path = f'data/{difficulty}_{dataset_name}'
            if os.path.exists(save_path):
                out = json.load(open(save_path))
            else:
                out = []
            for item in tqdm(data[len(out):]):
                inputs = prompt
                inputs = inputs.replace('<===article===>', item)
                inputs = inputs.replace('<===topic===>', topic)
                inputs = inputs.replace('<===difficulty===>', description[difficulty])
                out.append(obtain_response_gpt(inputs))
                json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()
