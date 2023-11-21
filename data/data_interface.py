import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import random
import torch
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer
import os



class TrainCollater:
    def __init__(self,
                 prompt_list=None,
                 llm_tokenizer=None,
                 train=False,
                 terminator="\n",
                 max_step=1):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train=train
        self.terminator = terminator
        self.max_step = max_step
        self.cur_step = 1

    def __call__(self, batch):
        if isinstance(self.prompt_list,list):
            instruction = random.choice(self.prompt_list)
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        else:
            instruction = sample["instruction_input"] if "instruction_input" in sample else None
            inputs_text = instruction if isinstance(instruction, list) else [instruction] * len(batch)
        
        thresh_hold = self.cur_step/self.max_step
        p = random.random()
        if p > thresh_hold:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [HistoryEmb]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt=", ".join([can_title+' [CansEmb]' for can_title in sample['cans_name']])
                    input_text=input_text.replace('[CansHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = False
        else:
            for i, sample in enumerate(batch):
                input_text=inputs_text[i]
                if '[HistoryHere]' in input_text:
                    insert_prompt=", ".join([seq_title+' [PH]' for seq_title in sample['seq_name']])
                    input_text=input_text.replace('[HistoryHere]',insert_prompt)
                if '[CansHere]' in input_text:
                    insert_prompt=", ".join([can_title+' [PH]' for can_title in sample['cans_name']])
                    input_text=input_text.replace('[CansHere]',insert_prompt)    
                inputs_text[i]=input_text
            flag = True
        self.cur_step += 1
        
        targets_text = [sample['correct_answer'] for sample in batch]

        if self.train:
            targets_text=[target_text+self.terminator for target_text in targets_text]
            inputs_pair = [[p, t] for p, t in zip(inputs_text, targets_text)]
            batch_tokens = self.llm_tokenizer(
                inputs_pair,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True)
            new_batch={"tokens":batch_tokens,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "flag":flag,
                       }
        else:
            batch_tokens = self.llm_tokenizer(
                inputs_text,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=True,
                return_attention_mask=True)
            cans_name=[sample['cans_name'] for sample in batch]
            new_batch={"tokens":batch_tokens,
                       "seq":torch.stack([torch.tensor(sample['seq']) for sample in batch], dim=0),
                       "cans":torch.stack([torch.tensor(sample['cans']) for sample in batch], dim=0),
                       "len_seq":torch.stack([torch.tensor(sample['len_seq']) for sample in batch], dim=0),
                       "len_cans":torch.stack([torch.tensor(sample['len_cans']) for sample in batch], dim=0),
                       "item_id": torch.stack([torch.tensor(sample['item_id']) for sample in batch], dim=0),
                       "correct_answer": targets_text,
                       "cans_name": cans_name,
                       }
        return new_batch

class DInterface(pl.LightningDataModule):

    def __init__(self, 
                 llm_tokenizer=None,
                 num_workers=8,
                 dataset='',
                 **kwargs):
        super().__init__()
        self.num_workers = num_workers
        self.llm_tokenizer=llm_tokenizer
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.max_epochs = kwargs['max_epochs']
        self.load_data_module()
        self.load_prompt(kwargs['prompt_path'])

        self.trainset = self.instancialize(stage='train')
        self.valset = self.instancialize(stage='val')
        self.testset = self.instancialize(stage='test')
        self.max_steps = self.max_epochs*(len(self.trainset)//self.batch_size)//self.num_workers

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True,
                          drop_last=True,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list,llm_tokenizer=self.llm_tokenizer,train=True, max_step=self.max_steps))

    def val_dataloader(self):
        return DataLoader(self.valset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list,llm_tokenizer=self.llm_tokenizer,train=False))

    def test_dataloader(self):
        return DataLoader(self.testset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list,llm_tokenizer=self.llm_tokenizer,train=False))

    def load_data_module(self):
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)
    
    def load_prompt(self,prompt_path):
        if os.path.isfile(prompt_path):
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            self.prompt_list = [p.strip() for p in raw_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []