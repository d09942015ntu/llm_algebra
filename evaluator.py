from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from mydataset import EvalDataset
from torch.utils.data import DataLoader
import glob
import os

import argparse

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np

# Custom Dataset

def repr_to_int(x):
    try:
        x = int(''.join(filter(str.isdigit, x)))
    except:
        x = 0
    return x

def generate_one_cycle(input_ids, model, tokenizer):
    pass
    #batch_text = np.array(batch['input_text']).transpose()

    #print(input_ids)
    # Generate text continuation

# Example usage


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    incorrect = 0
    with torch.no_grad():
        for j,batch in enumerate(dataloader):
            print(f'eval_step:{j}')
            device = model.device  # 模型所在設備
            input_ids = batch['input_ids']
            labels = batch['labels']
            input_ids = input_ids.to(device)  # GPU(移至)
            labels = labels.to(device)
            with torch.no_grad():
                # input_ids = tokenizer.encode(batch_text[i] + " >", return_tensors='pt')
                outputs = model.generate(
                    input_ids,
                    num_beams=10,
                    max_length=dataloader.dataset.max_length+1,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    top_p=0.95,
                    temperature=0.7,
                    do_sample=True,  # (隨機生成模式)因為有設置temperature 和 top_p
                    # force_words_ids = [tokenizer.additional_special_tokens_ids]
                )
                equal_sign_id = dataloader.dataset.tokenizer.encode(['[=]']) #50258
                labels_m1 = np.array([label.cpu().numpy()[0] for label in labels])
                outputs_m1 = np.zeros(labels_m1.shape, dtype=np.int64)
                for k, output_seq in enumerate(outputs):
                    for i in range(len(output_seq)-1):
                        current_val = output_seq[i].cpu().numpy()
                        if current_val == equal_sign_id:
                            outputs_m1[k]=output_seq[i+1].cpu().numpy()
                            break
                decoded = dataloader.dataset.tokenizer.decode(outputs_m1)
                correct+=np.sum(labels_m1 == outputs_m1)
                incorrect+=np.sum(labels_m1 != outputs_m1)
                #if j > limit:
                #    break
    return correct/(correct+incorrect)

def T_evaluate(model, dataloader, Log, limit=5):
    model.eval()
    correct = 0
    incorrect = 0

    with torch.no_grad():
        for j,batch in enumerate(dataloader):
            print(f'eval_step:{j}')
            device = model.device
            input_ids = batch['input_ids']
            labels = batch['labels']
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                # input_ids = tokenizer.encode(batch_text[i] + " >", return_tensors='pt')
                outputs = model.generate(
                    input_ids,
                    num_beams=10,
                    max_length=dataloader.dataset.max_length+1,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    top_p=0.95,
                    temperature=0.7,
                    do_sample=True,
                    # force_words_ids = [tokenizer.additional_special_tokens_ids]
                )
                equal_sign_id = dataloader.dataset.tokenizer.encode(['[=]']) #50258
                labels_m1 = np.array([label.cpu().numpy()[0] for label in labels])
                outputs_m1 = np.zeros(labels_m1.shape, dtype=np.int64)
                for k, output_seq in enumerate(outputs):
                    for i in range(len(output_seq)-1):
                        current_val = output_seq[i].cpu().numpy()
                        if current_val == equal_sign_id:
                            outputs_m1[k]=output_seq[i+1].cpu().numpy()
                            break
                correct+=np.sum(labels_m1 == outputs_m1)
                incorrect+=np.sum(labels_m1 != outputs_m1)
                #if j > limit:
                #    break
    return correct/(correct+incorrect)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with checkpoint and dataset paths.')
    parser.add_argument('--ckpt_path', type=str, default="./results/checkpoints", #"./results/checkpoint-245"
                        help='Path to the checkpoint file.')
    parser.add_argument('--dataset_path', type=str, default='./data/ide_41_11_9',
                        help='Path to the dataset file.')

    args = parser.parse_args()

    # Access the parsed arguments
    checkpoint_path = sorted(glob.glob(os.path.join(args.ckpt_path,"*")))[-1]
    dataset_path = args.dataset_path

    #GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assuming EvalDataset is defined elsewhere in your code

    model = GPT2LMHeadModel.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.to(device)
    dataloader_train = DataLoader(EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('./models/gpt2'), ftype='train'), batch_size=1, shuffle=False) # Multiple Batch Size requires paddings
    dataloader_eval_com = DataLoader(EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('./models/gpt2'), ftype='test_com'), batch_size=1, shuffle=False) # Multiple Batch Size requires paddings
    dataloader_eval_ide = DataLoader(EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('./models/gpt2'), ftype='test_ide'), batch_size=1, shuffle=False) # Multiple Batch Size requires paddings
    print(f"{args.dataset_path},Train accuracy: {evaluate(model, dataloader_train)}")
    print(f"{args.dataset_path}, Test_com accuracy: {evaluate(model, dataloader_eval_com)}")
    print(f"{args.dataset_path}, Test_ide accuracy: {evaluate(model, dataloader_eval_ide)}")


if __name__ == '__main__':
    main()
