import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import logging
from pathlib import Path
import json

# Nastavení loggeru
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Konstanty (stejné jako v tréninku)
SEQ_LENGTH = 512
BATCH_SIZE = 16

# Kontrola dostupnosti GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Dataset třída (stejná jako v tréninku)
class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item['text'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

# Model třída (stejná jako v tréninku)
class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, SEQ_LENGTH, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc_out(x)

def evaluate(model, dataloader, tokenizer):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            total_loss += loss.item()
            
            # Ukázka generování textu pro první sekvenci v batchi
            predicted = outputs[0].argmax(dim=-1)
            generated_text = tokenizer.decode(predicted)
            logger.info(f"Sample generated text: {generated_text[:100]}...")  # Prvních 100 znaků
            
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Average Test Loss: {avg_loss:.4f}")
    return avg_loss

def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    with torch.no_grad():
        # Tokenizace vstupního promptu
        tokens = tokenizer(prompt, truncation=True, padding='max_length', max_length=SEQ_LENGTH, return_tensors="pt")
        input_ids = tokens['input_ids'].to(device)
        
        # Generování textu
        outputs = model(input_ids)
        predicted = outputs[0].argmax(dim=-1)
        generated_text = tokenizer.decode(predicted)
        
        return generated_text

def main():
    # Načtení tokenizeru
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Načtení testovacích dat
    logger.info("Loading test data...")
    test_dataset = JSONLDataset("processed_dataset/test.jsonl", tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Inicializace modelu
    logger.info("Initializing model...")
    vocab_size = tokenizer.vocab_size
    model = TransformerLLM(vocab_size).to(device)
    
    # Testování každé epochy
    for epoch in range(1, 6):
        model_path = f"model_epoch_{epoch}.pth"
        logger.info(f"\nTesting model from epoch {epoch}")
        
        # Načtení vah modelu
        model.load_state_dict(torch.load(model_path))
        
        # Evaluace
        test_loss = evaluate(model, test_dataloader, tokenizer)
        
        # Vyzkoušení generování textu
        test_prompt = "This is a test prompt to see how the model generates text."
        generated = generate_text(model, tokenizer, test_prompt)
        logger.info(f"\nTest prompt: {test_prompt}")
        logger.info(f"Generated text: {generated}")

if __name__ == "__main__":
    main()