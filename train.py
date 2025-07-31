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
            
        # Enhanced model loading for Qwen3 with optimizations
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,  # Required for Qwen3
            "use_flash_attention_2": torch.cuda.is_available() and torch.version.cuda,
        }
        
        # Add quantization if specified
        if self.config.get('training', {}).get('use_4bit', True):
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["bnb_4bit_quant_type"] = "nf4"
            model_kwargs["bnb_4bit_use_double_quant"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # LoRA configuration optimized for Qwen3 architecture
        lora_config = LoraConfig(
            r=self.config['lora']['rank'],
            lora_alpha=self.config['lora']['alpha'], 
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            # Qwen3-specific optimizations
            use_rslora=self.config['lora'].get('use_rslora', False),
            use_dora=self.config['lora'].get('use_dora', False)
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
        
        # Enhanced training arguments for Qwen3
        training_config = self.config.get('training', {})
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=training_config.get('epochs', 3),
            per_device_train_batch_size=training_config.get('batch_size', 4),
            gradient_accumulation_steps=training_config.get('gradient_accumulation', 4),
            learning_rate=training_config.get('learning_rate', 2e-4),
            warmup_steps=training_config.get('warmup_steps', 100),
            logging_steps=training_config.get('logging_steps', 10),
            save_steps=training_config.get('save_steps', 500),
            save_total_limit=training_config.get('save_total_limit', 2),
            
            # Precision and memory optimizations
            fp16=not training_config.get('bf16', False),
            bf16=training_config.get('bf16', False),
            tf32=training_config.get('tf32', True),
            
            # Memory management
            dataloader_pin_memory=training_config.get('dataloader_pin_memory', False),
            gradient_checkpointing=training_config.get('gradient_checkpointing', True),
            
            # Optimization
            optim=training_config.get('optimizer', 'adamw_torch'),
            weight_decay=training_config.get('weight_decay', 0.01),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            
            # Evaluation
            evaluation_strategy="steps" if training_config.get('eval_steps') else "no",
            eval_steps=training_config.get('eval_steps', 500),
            
            # Reporting
            report_to="wandb" if self.config.get('wandb', {}).get('enabled', False) else None,
            run_name=self.config.get('wandb', {}).get('run_name', 'codecontext-training'),
            
            # Advanced features
            remove_unused_columns=training_config.get('remove_unused_columns', False),
            label_smoothing_factor=training_config.get('label_smoothing_factor', 0.0)
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