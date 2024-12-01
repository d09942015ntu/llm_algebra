import json

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from mydataset import TrainDataset, EvalDataset
from evaluator  import T_evaluate
import argparse
import os
import glob
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch.nn as nn
import sys

import logging
from datetime import datetime




def setup_logger(name, log_file, level=logging.DEBUG):
    """Sets up a logger to output to both terminal and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def reinitialize_weights(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

class StringOutputEvaluator(TrainerCallback):
    def __init__(self, model, tokenizer,ckpt_path,dataset_dir, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.ckpt_path = ckpt_path
        self.dataset_dir = dataset_dir
        self.logger = logger
        self.wait = 0
        self.batch_size = 1

        self.Tdataloader = DataLoader(EvalDataset(self.dataset_dir, self.tokenizer, ftype='train'),
                                      batch_size=self.batch_size, shuffle=False)  # Multiple Batch Size requires paddings
        self.Edataloader0 = DataLoader(EvalDataset(self.dataset_dir, self.tokenizer, ftype='test'),
                                       batch_size=self.batch_size, shuffle=False)  # Multiple Batch Size requires paddings
        self.Edataloader1 = DataLoader(EvalDataset(self.dataset_dir, self.tokenizer, ftype='test_com'),
                                       batch_size=self.batch_size, shuffle=False)  # Multiple Batch Size requires paddings
        self.Edataloader2 = DataLoader(EvalDataset(self.dataset_dir, self.tokenizer, ftype='test_ide'),
                                       batch_size=self.batch_size, shuffle=False)  # Multiple Batch Size requires paddings
        self.Edataloader3 = DataLoader(EvalDataset(self.dataset_dir, self.tokenizer, ftype='test_inv'),
                                       batch_size=self.batch_size, shuffle=False)  # Multiple Batch Size requires paddings

    def on_log(self, args, state, control, **kwargs):
        #model = kwargs.get('model')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        #checkpoint_path = self.ckpt_path

        #ckpt_paths = glob.glob(os.path.join(checkpoint_path, "*"))
        #if len(ckpt_paths) == 0:
        #    return
        #checkpoint_path = sorted(ckpt_paths)[-1]
        self.model.to(device)
        self.model.eval()
        Taccuracy = T_evaluate(self.model, self.Tdataloader, state.log_history)
        #Eaccuracy0 = T_evaluate(model, self.Edataloader0, state.log_history)
        Eaccuracy1 = T_evaluate(self.model, self.Edataloader1, state.log_history)
        Eaccuracy2 = T_evaluate(self.model, self.Edataloader2, state.log_history)
        #Eaccuracy3 = T_evaluate(model, self.Edataloader3, state.log_history)
        epoch=state.log_history[-1]['epoch']
        loss=state.log_history[-1]['loss']
        step=state.log_history[-1]['step']
        acc_train = Taccuracy
        log_str = json.dumps({'step':step,
                              'epoch':epoch,
                              'loss':loss,
                              'acc_train': Taccuracy,
                              #'acc_eval_all': Eaccuracy0,
                              'acc_eval_com': Eaccuracy1,
                              'acc_eval_ide': Eaccuracy2,
                              #'acc_eval_inv': Eaccuracy3,
                              })
        self.logger.info(log_str)
        print(log_str)
        if  loss < 0.001:
            self.wait += 1
            if self.wait > 2:
                sys.exit()

    
def main():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--model_name', type=str, default='./mymodels/toytrans',  help='Pre-trained model name or path')
    parser.add_argument('--dataset_dir', type=str, default='./data/ide_41_5_9',  help='Path to the training dataset')
    parser.add_argument('--output_dir', type=str, default='./results',  help='Path to output directory')

    args = parser.parse_args()


    tokenizer_path = "./models/gpt2"

    model = AutoModelForCausalLM.from_pretrained(args.model_name,trust_remote_code=True,ignore_mismatched_sizes=True)
    reinitialize_weights(model)
    dataset_train = TrainDataset(args.dataset_dir, AutoTokenizer.from_pretrained(tokenizer_path))
    dataset_eval = EvalDataset(args.dataset_dir, AutoTokenizer.from_pretrained(tokenizer_path))
    
    if 'resize_token_embeddings_by_tokenizer' in dir(model):
        model.resize_token_embeddings_by_tokenizer(dataset_train.tokenizer)
    else:
        model.resize_token_embeddings(len(dataset_train.tokenizer))

    # Get the current date and time
    current_datetime = datetime.now()

    # Print the current date and time
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir,f"{formatted_datetime}-{(os.path.basename(args.dataset_dir))}")
    ckpt_path = os.path.join(output_dir,"checkpoints")

    os.makedirs(output_dir,exist_ok=True)
    logger = setup_logger("my_logger", os.path.join(output_dir,"trainer.log"))

    logging_step=1000

    training_args = TrainingArguments(
        output_dir=ckpt_path, # Directory to save the training results
        num_train_epochs=2000000,        # Total number of training epochs
        per_device_train_batch_size = 1024, #1024, # Batch size per device during training
        save_steps=logging_step,                  # Save the model every 50 steps
        save_total_limit=3,             # Keep a maximum of 3 checkpoints
        logging_steps=logging_step,               # Log(output) after every 10 steps
        learning_rate=5e-5,             # Initial learning rate
        weight_decay=0.01               # L2 weight decay (regularization)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=dataset_train.tokenizer,
        callbacks = [StringOutputEvaluator(model, dataset_train.tokenizer, ckpt_path, args.dataset_dir, logger)]
    )

    # Train the model
    trainer.train()

# Initialize Trainer
if __name__ == '__main__':
    main()
