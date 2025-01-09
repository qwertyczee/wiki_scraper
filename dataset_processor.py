import json
from pathlib import Path
import random

class DatasetProcessor:
    def __init__(self, input_dir='dataset', output_dir='processed_dataset'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Rozdělí dataset na trénovací, validační a testovací část."""
        all_articles = []
        
        # Načtení všech článků
        for file in self.input_dir.glob('*.jsonl'):
            with file.open('r', encoding='utf-8') as f:
                all_articles.extend([line.strip() for line in f])
        
        # Zamíchání článků
        random.shuffle(all_articles)
        
        # Rozdělení na části
        total = len(all_articles)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        # Uložení jednotlivých částí
        splits = {
            'train': all_articles[:train_end],
            'val': all_articles[train_end:val_end],
            'test': all_articles[val_end:]
        }
        
        for split_name, articles in splits.items():
            output_file = self.output_dir / f'{split_name}.jsonl'
            with output_file.open('w', encoding='utf-8') as f:
                for article in articles:
                    f.write(article + '\n')
                    
        return splits