import torch
from torch.utils.data import Dataset
import pandas as pd
import transformers
from transformers import RobertaTokenizer


class RelationExtractionDataset(Dataset):
    def __init__(
            self, 
            dataframe:pd.DataFrame, 
            tokenizer:transformers.PreTrainedTokenizer,
            max_len:int
        ):
        """dataframe 结构：[id, text, label]"""
        self.tokenizer = tokenizer
        self.text = dataframe['text']
        self.label = dataframe['label']
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            self.text[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label': torch.tensor(self.label[idx])
        }
