import wikipediaapi
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import re
from config import MIN_ARTICLE_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

class WikiScraper:
    def __init__(self, language='cs'):
        # Vytvoření custom user agent headers
        headers = {
            'User-Agent': 'LLMTrainingDataCollector/1.0'
        }
        
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            headers=headers  # Použití headers místo user_agent
        )
        self.output_dir = Path('dataset')
        self.output_dir.mkdir(exist_ok=True)
        
    def clean_text(self, text):
        """Vyčistí text od nežádoucích znaků a formátování."""
        # Odstranění referencí
        text = re.sub(r'\[\d+\]', '', text)
        # Odstranění více mezer
        text = re.sub(r'\s+', ' ', text)
        # Odstranění prázdných řádků
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def get_page_content(self, title):
        """Získá obsah článku podle názvu."""
        try:
            page = self.wiki.page(title)
            if page.exists():
                return {
                    'title': title,
                    'text': self.clean_text(page.text),
                    'url': page.fullurl,
                    'categories': list(page.categories.keys())
                }
        except Exception as e:
            logging.error(f"Chyba při získávání stránky {title}: {str(e)}")
        return None

    def save_batch(self, articles, batch_num):
        """Uloží dávku článků do JSONL souboru."""
        """Uloží dávku článků do JSONL souboru."""
        output_file = self.output_dir / f'wiki_batch_{batch_num}.jsonl'
        with output_file.open('w', encoding='utf-8') as f:
            for article in articles:
                if article and len(article['text']) > MIN_ARTICLE_LENGTH:  # Ukládáme pouze delší články
                    # Přidání dodatečných metadat
                    article['metadata'] = {
                        'length': len(article['text']),
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'language': 'cs'
                    }
                    json.dump(article, f, ensure_ascii=False)
                    f.write('\n')

    def scrape_category(self, category_name, max_articles=1000, batch_size=100):
        """Stáhne články z dané kategorie."""
        category = self.wiki.page(f"Kategorie:{category_name}")  # Změna na český název kategorie
        if not category.exists():
            logging.error(f"Kategorie {category_name} neexistuje")
            return

        members = list(category.categorymembers.keys())
        logging.info(f"Nalezeno {len(members)} článků v kategorii {category_name}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            for i in range(0, min(len(members), max_articles), batch_size):
                batch_titles = members[i:i+batch_size]
                articles = list(tqdm(
                    executor.map(self.get_page_content, batch_titles),
                    total=len(batch_titles),
                    desc=f"Batch {i//batch_size + 1}"
                ))
                self.save_batch(articles, i//batch_size + 1)
                time.sleep(1)  # Přestávka mezi dávkami