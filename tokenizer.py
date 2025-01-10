from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

class CustomTokenizer:
    def __init__(self, vocab_size=30000):
        self.tokenizer = ByteLevelBPETokenizer()
        self.vocab_size = vocab_size
        
    def train(self, data_files):
        # Trénování tokenizeru na našich datech
        self.tokenizer.train(
            files=data_files,
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"]
        )
        
        # Uložení natrénovaného tokenizeru
        Path("tokenizer").mkdir(exist_ok=True)
        self.tokenizer.save_model("tokenizer")
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids) 
    
    def initialize_tokenizer(self, vocab_file, merges_file):
        """Initialize tokenizer from vocabulary and merges files"""
        self.tokenizer = ByteLevelBPETokenizer(
            vocab_file,
            merges_file,
            add_prefix_space=True  # Důležité pro správné tokenizování
        )
        # Přidání speciálních tokenů
        self.tokenizer.add_special_tokens(["<pad>", "<unk>", "<sos>", "<eos>"])
