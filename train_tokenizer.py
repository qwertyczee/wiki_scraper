from tokenizer import CustomTokenizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def train_tokenizer():
    # Vytvoření adresáře pro tokenizer
    Path("tokenizer").mkdir(exist_ok=True)
    
    # Inicializace tokenizeru
    tokenizer = CustomTokenizer()
    
    # Seznam všech JSONL souborů z processed_dataset jako stringy
    data_files = [str(f) for f in Path("processed_dataset").glob("*.jsonl")]
    
    if not data_files:
        raise ValueError("No training data found in processed_dataset directory")
    
    # Trénování tokenizeru
    logging.info("Training tokenizer...")
    tokenizer.train(data_files)
    
    logging.info("Tokenizer training completed")

if __name__ == "__main__":
    train_tokenizer()
