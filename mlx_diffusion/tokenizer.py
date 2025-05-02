from transformers import AutoTokenizer
import torch

class Tokenizer:
    def __init__(self, pretrained_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        
    def to(self, device):
        return self
    
    def encode(self, text: str):
        encoded = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.tokenizer.max_length
        )
        return encoded.input_ids
    