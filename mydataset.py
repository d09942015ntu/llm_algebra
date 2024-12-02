import os.path
import json
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import pandas as pd
import torch
import random
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, data_dir, tokenizer, ftype='train'):
        self.data = pd.read_csv(os.path.join(data_dir,f"{ftype}.csv"))
        tokens = json.load(open(os.path.join(data_dir,"tokens.json"),"r"))
        special_tokens_dict = {'additional_special_tokens': tokens}
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer = tokenizer
        self.max_length = 9
        self.rng = np.random.RandomState(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Randomly choose a starting point
        # Match s_i as input and s_{i+1} as the target
        s1 = row['s1']
        s2 = row['s2']

        input_text = s1
        full_text = s1 + s2

        encoding = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding='max_length')
        encoding_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding='max_length')
        s1_encoded = self.tokenizer.encode(s1)
        labels = encoding_full["input_ids"].copy()
        labels[:len(s1_encoded)] = [-100]*len(s1_encoded)  # Mask `s1` tokens
        labels[len(s1_encoded)+1:] = [-100]*len(labels[len(s1_encoded)+1:])  # Mask `s1` tokens

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(labels)
        }


class EvalDataset(TrainDataset):
    def __init__(self, file_path, tokenizer, ftype='test'):
        super().__init__(file_path, tokenizer, ftype=ftype)
        self.max_length = 8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Randomly choose a starting point
        # Match s_i as input and s_{i+1} as the target
        s1 = row['s1']
        s2 = row['s2']

        encoding = self.tokenizer.encode(s1) #, truncation=True, max_length=s1.count('+')*2+2, padding='max_length')
        encoding_label = self.tokenizer.encode(s2) #, truncation=True, max_length=1, padding='max_length')

        return {
            'input_ids': torch.tensor(encoding),
            'attention_mask': torch.ones(len(encoding)),
            'labels': torch.tensor(encoding_label)
        }

if __name__ == '__main__':
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    data_name='data/ide_41_5_1'
    dataset = TrainDataset(data_name, tokenizer)
    #dataset = EvalDataset(data_name, tokenizer)
    for item in dataset:
        for key, value in item.items():
            if key == 'input_ids' or key == 'labels':
                raw = dataset.tokenizer.decode(value[value>0])
            else:
                raw = "0"
            print(f"{key}: {raw} : {value}")  # Adjust processing logic as necessary
