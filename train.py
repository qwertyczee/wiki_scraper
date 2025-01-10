import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from model import CustomTransformer
from tokenizer import CustomTokenizer
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging

logging.basicConfig(level=logging.INFO)

class WikiDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Načtení dat
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if 'text' in entry:
                    self.data.append(entry['text'])
                elif 'input' in entry and 'output' in entry:
                    self.data.append(entry['input'])
                    self.data.append(entry['output'])
                else:
                    raise ValueError("Invalid data format")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode(text)
        
        # Oříznutí nebo padding na max_length
        input_ids = encoding.ids[:self.max_length]
        if len(input_ids) < self.max_length:
            input_ids += [0] * (self.max_length - len(input_ids))
        
        return torch.tensor(input_ids)

def train_model(
    train_path,
    val_path,
    vocab_size=30000,
    batch_size=32,
    epochs=10,
    learning_rate=3e-4,
    gradient_accumulation_steps=4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Inicializace tokenizeru a jeho trénování
    tokenizer = CustomTokenizer(vocab_size)
    tokenizer.train([train_path, val_path])
    
    # Vytvoření datasetů
    train_dataset = WikiDataset(train_path, tokenizer)
    val_dataset = WikiDataset(val_path, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Inicializace modelu
    model = CustomTransformer(vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    
    # Trénování
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # Zajišťuje, že gradienty budou akumulovány
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            # Vytvoření target dat (posunuto o jeden token)
            target = data[:, 1:]
            input_data = data[:, :-1]
            
            output = model(input_data)
            loss = criterion(output.view(-1, vocab_size), target.contiguous().view(-1))
            
            loss.backward()  # Zpětné šíření gradientu
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:  # Pokud dosáhneme počtu kroků pro akumulaci
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()  # Resetování gradientů po kroku aktualizace
                
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validace
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                target = data[:, 1:]
                input_data = data[:, :-1]
                output = model(input_data)
                val_loss += criterion(output.view(-1, vocab_size), target.contiguous().view(-1)).item()
        
        val_loss /= len(val_loader)
        logging.info(f'Epoch: {epoch}, Training Loss: {total_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')
        
        # Uložení checkpointu
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f'checkpoints/model_epoch_{epoch}.pt')

if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    
    train_model(
        train_path='processed_dataset/train.jsonl',
        val_path='processed_dataset/val.jsonl',
        vocab_size=30000,
        batch_size=32,
        epochs=10,
        learning_rate=3e-4
    )
