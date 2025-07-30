#!/usr/bin/env python3

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import argparse
import os
import wandb

class CodeContextTrainer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['base_model']
        self.output_dir = self.config['training']['output_dir']
        
    def setup_model_and_tokenizer(self):
        """Load and configure model with LoRA"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config['lora']['rank'],
            lora_alpha=self.config['lora']['alpha'], 
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
    def load_dataset(self):
        """Load and preprocess training data"""
        dataset = load_dataset(
            self.config['data']['dataset_path'],
            split=self.config['data']['split']
        )
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config['data']['max_length']
            )
            
        return dataset.map(tokenize_function, batched=True)
        
    def train(self):
        """Execute training loop"""
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['wandb']['project'],
                name=self.config['wandb']['run_name']
            )
        
        self.setup_model_and_tokenizer()
        train_dataset = self.load_dataset()
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation'],
            learning_rate=self.config['training']['learning_rate'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            fp16=True,
            dataloader_pin_memory=False,
            report_to="wandb" if self.config.get('wandb', {}).get('enabled', False) else None
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        trainer.save_model()
        
        # Merge and save final model
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(f"{self.output_dir}/merged")
        self.tokenizer.save_pretrained(f"{self.output_dir}/merged")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Training configuration file")
    args = parser.parse_args()
    
    trainer = CodeContextTrainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()