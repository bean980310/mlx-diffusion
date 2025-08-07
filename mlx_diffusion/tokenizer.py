import os
from typing import Optional, Union, List, Dict, Any
import mlx.core as mx
from transformers import AutoTokenizer, CLIPTokenizer
from .models import TextEncoder


class MLXTokenizerOutput:
    """MLX tokenizer output container"""
    def __init__(self, input_ids: mx.array, attention_mask: Optional[mx.array] = None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class MLXTokenizer:
    """MLX wrapper for HuggingFace tokenizers with text encoder support"""
    
    def __init__(self, 
                 tokenizer_path: str,
                 text_encoder_path: Optional[str] = None,
                 max_length: int = 77,
                 pad_token_id: int = 49407):
        
        # Load HuggingFace tokenizer
        if os.path.isdir(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # Assume it's a model name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
        # Load text encoder if provided
        if text_encoder_path:
            self.text_encoder = TextEncoder(text_encoder_path)
        else:
            self.text_encoder = None
            
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.model_max_length = getattr(self.tokenizer, 'model_max_length', max_length)
    
    def __call__(self, 
                 text: Union[str, List[str]],
                 padding: Union[bool, str] = "max_length",
                 max_length: Optional[int] = None,
                 truncation: bool = True,
                 return_tensors: str = "mlx",
                 return_attention_mask: bool = False) -> MLXTokenizerOutput:
        """Tokenize text input"""
        
        if max_length is None:
            max_length = self.model_max_length
            
        # Handle single string input
        if isinstance(text, str):
            text = [text]
            
        # Tokenize using HuggingFace tokenizer
        encoded = self.tokenizer(
            text,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors="np"  # Return numpy for MLX conversion
        )
        
        # Convert to MLX arrays
        input_ids = mx.array(encoded["input_ids"])
        
        attention_mask = None
        if return_attention_mask and "attention_mask" in encoded:
            attention_mask = mx.array(encoded["attention_mask"])
            
        return MLXTokenizerOutput(input_ids, attention_mask)
    
    def encode(self, 
               text: Union[str, List[str]], 
               add_special_tokens: bool = True,
               max_length: Optional[int] = None,
               truncation: bool = True,
               padding: Union[bool, str] = "max_length") -> mx.array:
        """Simple encoding method that returns input_ids"""
        
        result = self(
            text=text,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors="mlx"
        )
        return result.input_ids
    
    def decode(self, 
               token_ids: Union[mx.array, List[int]], 
               skip_special_tokens: bool = True,
               clean_up_tokenization_spaces: bool = True) -> Union[str, List[str]]:
        """Decode token IDs back to text"""
        
        # Convert MLX array to list if needed
        if isinstance(token_ids, mx.array):
            if len(token_ids.shape) == 1:
                token_ids = token_ids.tolist()
            else:
                # Batch decoding
                batch_texts = []
                for batch_tokens in token_ids:
                    text = self.tokenizer.decode(
                        batch_tokens.tolist(),
                        skip_special_tokens=skip_special_tokens,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces
                    )
                    batch_texts.append(text)
                return batch_texts
                
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def encode_text_embeddings(self, text: Union[str, List[str]]) -> mx.array:
        """Encode text to embeddings using text encoder"""
        if self.text_encoder is None:
            raise ValueError("Text encoder not loaded. Provide text_encoder_path during initialization.")
            
        input_ids = self.encode(text)
        return self.text_encoder(input_ids)


class CLIPMLXTokenizer(MLXTokenizer):
    """Specialized MLX tokenizer for CLIP models"""
    
    def __init__(self, 
                 tokenizer_path: str = "openai/clip-vit-base-patch32",
                 text_encoder_path: Optional[str] = None,
                 max_length: int = 77):
        
        # Use CLIP-specific tokenizer
        if os.path.isdir(tokenizer_path):
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
            
        # Load text encoder if provided
        if text_encoder_path:
            self.text_encoder = TextEncoder(text_encoder_path)
        else:
            self.text_encoder = None
            
        self.max_length = max_length
        self.model_max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id or 49407


def create_tokenizer(model_type: str = "clip", **kwargs) -> MLXTokenizer:
    """Factory function to create appropriate tokenizer"""
    
    if model_type.lower() == "clip":
        return CLIPMLXTokenizer(**kwargs)
    else:
        return MLXTokenizer(**kwargs)
    