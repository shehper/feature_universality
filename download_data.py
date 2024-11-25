"""
Downloads tokenized openwebtext dataset.
"""

from datasets import load_dataset
dataset_path = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
train_data = load_dataset(dataset_path, split="train", streaming=False)