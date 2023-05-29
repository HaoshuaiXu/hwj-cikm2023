import pandas as pd
import json


def process_wiki80(data_filepath:str, label2id_filepath:str):
    wiki80_data = pd.read_json(data_filepath, lines=True)
    with open(label2id_filepath) as f:
        relation2id = json.load(f)
    for k, v in relation2id.items():
        wiki80_data.replace(to_replace=k, value=v, inplace=True)
    wiki80_data.rename(
        columns={'token': 'text', 'relation': 'label'},
        inplace=True
    )
    return wiki80_data[['text', 'label']]
