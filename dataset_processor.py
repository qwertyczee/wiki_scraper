import json
from pathlib import Path
import random
import logging

class DatasetProcessor:
    def __init__(self, input_dir='dataset', conversation_dir='conversation_dataset', output_dir='processed_dataset'):
        self.input_dir = Path(input_dir)
        self.conversation_dir = Path(conversation_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_text(self, text):
        """Process and normalize text."""
        text = text.encode('utf-8').decode('utf-8')
        return text
    
    def process_conversation(self, conversation):
        """Process conversation into training pairs."""
        processed = []
        for i in range(len(conversation) - 1):
            processed.append({
                'input': conversation[i]['text'],
                'output': conversation[i + 1]['text']
            })
        return processed
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split dataset into train/val/test sets."""
        all_data = []
        
        # Load Wikipedia articles
        for file in self.input_dir.glob('*.jsonl'):
            with file.open('r', encoding='utf-8') as f:
                for line in f:
                    all_data.append(self.process_text(line.strip()))
        
        # Load conversations
        for file in self.conversation_dir.glob('*.jsonl'):
            with file.open('r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if 'conversation' in entry:
                        processed_conv = self.process_conversation(entry['conversation'])
                        for pair in processed_conv:
                            all_data.append(json.dumps(pair, ensure_ascii=False))
        
        # Shuffle and split data
        random.shuffle(all_data)
        total = len(all_data)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        splits = {
            'train': all_data[:train_end],
            'val': all_data[train_end:val_end],
            'test': all_data[val_end:]
        }
        
        # Save splits
        for split_name, data in splits.items():
            output_file = self.output_dir / f'{split_name}.jsonl'
            with output_file.open('w', encoding='utf-8') as f:
                for item in data:
                    f.write(item + '\n')
        
        return splits