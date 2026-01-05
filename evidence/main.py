import json
import os.path
from tqdm import tqdm
from utils import obtain_response_gpt, obtain_response_qwen


def obtain_evidence(claim):
    with open('prompts/planner.txt') as f:
        prompt = f.read()
    inputs = prompt.replace('<===c===>', claim)
    evidence = obtain_response_gpt(inputs)
    evidence = evidence
    evidence = evidence.replace('```python', '')
    evidence = evidence.replace('```', '')
    try:
        evidence = json.loads(evidence)
    except:
        evidence = ['', '', '']
    return evidence


# def obtain_score(claim, evidence):
#     with open('prompts/belief.txt') as f:
#         prompt = f.read()
#     evidence_text = ''
#     for _, each in enumerate(evidence):
#         evidence_text += f'{_ + 1}: {each}\n'
#     inputs = prompt.replace('<===c===>', claim)
#     inputs = inputs.replace('<===e===>', evidence_text)
#     return obtain_response_qwen(inputs)


def refine(claim, item):
    if item is None:
        item = ' '
    with open('prompts/reviwer.txt') as f:
        judgment_prompt = f.read()
    with open('prompts/refiner.txt') as f:
        refine_prompt = f.read()
    inputs = judgment_prompt.replace('<===e===>', item)
    judgment = obtain_response_gpt(inputs)
    inputs = refine_prompt.replace('<===e===>', item)
    inputs = inputs.replace('<===j===>', judgment)
    inputs = inputs.replace('<===c===>', claim)
    # print(inputs)
    # print('========')
    refined = obtain_response_gpt(inputs)
    # print(refined)
    # print('======')
    return refined


def main():
    path = '../misinformation/data'
    dataset_names = sorted(os.listdir(path))
    for dataset_name in dataset_names:
        data = json.load(open(f'../misinformation/data/{dataset_name}'))
        save_path = f'data/{dataset_name}'
        if os.path.exists(save_path):
            out = json.load(open(save_path))
        else:
            out = []
        for misinfo in tqdm(data[len(out):]):
            if misinfo is None:
                misinfo = ' '
            each = []
            evidence = obtain_evidence(misinfo)
            each.append(evidence)
            for _ in range(5):
                temp = []
                for item in evidence:
                    temp.append(refine(misinfo, item))
                each.append(temp)
                evidence = temp
            out.append(each)
            json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()

