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
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Kontrola dostupnosti GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Konstanty
SEQ_LENGTH = 512
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# Dataset
class JSONLDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer

        # Načtení dat
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

# Model
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

# Funkce trénování
def train(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss

# Načtení dat
logger.info("Loading data...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Použijte svůj tokenizer
train_dataset = JSONLDataset("processed_dataset/train.jsonl", tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Inicializace modelu
logger.info("Initializing model...")
vocab_size = tokenizer.vocab_size
model = TransformerLLM(vocab_size).to(device)

# Optimizer a scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 10, gamma=0.1)

# Ztrátová funkce
criterion = nn.CrossEntropyLoss()

# Trénovací smyčka
logger.info("Starting training...")
for epoch in range(NUM_EPOCHS):
    logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss = train(model, train_dataloader, optimizer, criterion, scheduler)
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
    logger.info(f"Model saved for epoch {epoch + 1}")

logger.info("Training completed.")
