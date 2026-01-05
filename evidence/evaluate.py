# gpt-3.5-turbo as an example


import json
import os.path
from tqdm import tqdm
from utils import obtain_response_gpt, obtain_response_qwen


def obtain_score(claim, evidence):
    with open('prompts/belief.txt') as f:
        prompt = f.read()
    evidence_text = ''
    for _, each in enumerate(evidence):
        if each is None:
            each = ''
        evidence_text += f'{_ + 1}: {each}\n'
    if claim is None:
        claim = ''
    inputs = prompt.replace('<===c===>', claim)
    inputs = inputs.replace('<===e===>', evidence_text)
    # print(inputs)
    try:
        return obtain_response_gpt(inputs, model='gpt-3.5-turbo')
    except:
        return 'I can not generate.'


def main():
    path = '../misinformation/data'
    dataset_names = os.listdir(path)
    for dataset_name in dataset_names:
        data = json.load(open(f'../misinformation/data/{dataset_name}'))
        evidences = json.load(open(f'data/{dataset_name}'))
        save_path = f'scores/{dataset_name}'
        if os.path.exists(save_path):
            out = json.load(open(save_path))
        else:
            out = []
        indices = list(range(len(data)))
        for index in tqdm(indices[len(out):]):
            claim = data[index]
            evidence = evidences[index]
            each = []
            for _ in range(7):
                if _ == 6:
                    each.append(obtain_score(claim, []))
                else:
                    each.append(obtain_score(claim, evidence[_]))
                # print(each[-1])
                # input()
            out.append(each)
            json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()

