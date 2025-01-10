import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import logging
from pathlib import Path

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
SEQ_LENGTH = 128  # Sníženo pro lepší paměťovou náročnost
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# Dataset
class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Přidáme speciální tokeny pro začátek a konec sekvence
        text = "<s> " + text + " </s>"
        tokens = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=SEQ_LENGTH,
            return_tensors="pt"
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0)
        }

# Model
class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, SEQ_LENGTH, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=True),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x) + self.position_encoding[:, :x.size(1), :]
        if attention_mask is not None:
            # Vytvoření padding masky pro transformer
            padding_mask = ~attention_mask.bool()
            x = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer(x)
        return self.fc_out(x)

# Funkce pro generování textu
def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            predictions = outputs[:, -1, :]
            predicted_id = torch.argmax(predictions, dim=-1)
            
            if predicted_id[0] == tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, predicted_id.unsqueeze(-1)], dim=-1)
    
    return tokenizer.decode(input_ids[0])

# Funkce trénování
def train(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Vytvoříme targets posunutím input_ids o jednu pozici
        targets = input_ids.clone()
        targets = torch.roll(targets, shifts=-1, dims=1)
        targets[:, -1] = tokenizer.pad_token_id

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
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

# Načtení datasetu
logger.info("Loading dataset...")
dataset = load_dataset("karmiq/wikipedia-embeddings-cs-seznam-mpnet", trust_remote_code=True)  # Přidáno trust_remote_code=True
texts = dataset["train"]["chunks"]

# Inicializace tokenizeru
logger.info("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Používáme GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Vytvoření dataloaderu
train_dataset = ShakespeareDataset(texts, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Inicializace modelu
logger.info("Initializing model...")
vocab_size = tokenizer.vocab_size
model = TransformerLLM(vocab_size).to(device)

# Optimizer a scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=max(1, total_steps // 10),  # changed code
    gamma=0.1
)

# Ztrátová funkce
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Trénovací smyčka
logger.info("Starting training...")
for epoch in range(NUM_EPOCHS):
    logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train_loss = train(model, train_dataloader, optimizer, criterion, scheduler)
    
    # Uložení modelu
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, f"model_epoch_{epoch + 1}.pth")
    
    # Vyzkoušení generování textu
    test_prompt = "To be or not to be"
    generated_text = generate_text(model, tokenizer, test_prompt)
    logger.info(f"Sample generation: {generated_text}")
    
    logger.info(f"Model saved for epoch {epoch + 1}")

logger.info("Training completed.")