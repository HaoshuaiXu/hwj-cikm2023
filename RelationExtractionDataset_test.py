import torch
from torch.utils.data import DataLoader
import transformers
from transformers import RobertaTokenizer
import pandas as pd
from RelationExtractionDataset import RelationExtractionDataset


if __name__ == '__main__':
    test_data = pd.read_csv(
        "./dataset/semeval_trainingset_class-code_without-entity.tsv",
        delimiter='\t'
    )
    semeval_dataset = RelationExtractionDataset(
        dataframe=test_data,
        tokenizer=RobertaTokenizer.from_pretrained('roberta-base'),
        max_len=256
    )
    semeval_dataloader = DataLoader(semeval_dataset, batch_size=64, num_workers=4)
    print(next(iter(semeval_dataloader)))
