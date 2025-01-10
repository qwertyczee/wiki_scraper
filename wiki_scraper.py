import wikipediaapi
import json
import time
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
import logging
from pathlib import Path
import re
from config import MIN_ARTICLE_LENGTH, BATCH_SIZE
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import queue
import threading

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
        self.headers = {
            'User-Agent': 'LLMTrainingDataCollector/1.0'
        }
        
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            headers=self.headers
        )
        self.output_dir = Path('dataset')
        self.output_dir.mkdir(exist_ok=True)
        
        # Buffer pro ukládání článků
        self.article_buffer = queue.Queue(maxsize=1000)
        self.save_thread = None
        self.is_running = True
        self.batch_size = BATCH_SIZE

    def clean_text(self, text):
        """Vyčistí text od nežádoucích znaků a formátování."""
        # Odstranění referencí
        text = re.sub(r'\[\d+\]', '', text)
        # Odstranění více mezer
        text = re.sub(r'\s+', ' ', text)
        # Odstranění prázdných řádků
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
    
    def split_into_sentences(self, text):
        """Rozdělí text na věty."""
        # Základní rozdělení podle tečky, otazníku a vykřičníku
        # Bere v úvahu zkratky (např., atd., př.n.l.) a čísla (3.14)
        sentences = []
        current = []
        
        # Rozdělení na potenciální věty
        for part in re.split(r'([.!?]+(?=\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]))', text):
            current.append(part)
            if re.search(r'[.!?]+(?=\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])', part):
                sentence = ''.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []
        
        # Přidání poslední věty, pokud existuje
        last_sentence = ''.join(current).strip()
        if last_sentence:
            sentences.append(last_sentence)
            
        return [s for s in sentences if len(s) > 10]  # Ignorování příliš krátkých vět

    async def get_page_content_async(self, session, title):
        """Asynchronně získá obsah článku."""
        api_url = "https://cs.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts|categories|info",
            "explaintext": "1",
            "inprop": "url",
            "redirects": "1"
        }

        try:
            async with session.get(api_url, params=params) as response:
                data = await response.json()
                pages = data["query"]["pages"]
                page = next(iter(pages.values()))
                
                if "extract" in page:
                    return {
                        'title': page.get("title", title),
                        'text': self.clean_text(page["extract"]),
                        'url': page.get("fullurl", ""),
                        'categories': [cat.get("title", "") for cat in page.get("categories", [])]
                    }
        except Exception as e:
            logging.error(f"Chyba při získávání stránky {title}: {str(e)}")
        return None

    def save_worker(self):
        """Background worker pro ukládání článků."""
        current_batch = []
        batch_num = 1

        while self.is_running or not self.article_buffer.empty():
            try:
                article = self.article_buffer.get(timeout=1.0)
                if article:
                    current_batch.append(article)
                
                if len(current_batch) >= self.batch_size or (
                    not self.is_running and 
                    self.article_buffer.empty() and 
                    current_batch
                ):
                    self.save_batch(current_batch, batch_num)
                    batch_num += 1
                    current_batch = []
                    
            except queue.Empty:
                if current_batch:
                    self.save_batch(current_batch, batch_num)
                    current_batch = []
                if not self.is_running:
                    break

    async def scrape_category_async(self, category_name, max_articles=1000):
        """Asynchronně stáhne články z kategorie."""
        category = self.wiki.page(f"Kategorie:{category_name}")
        if not category.exists():
            logging.error(f"Kategorie {category_name} neexistuje")
            return 0

        members = list(category.categorymembers.keys())[:max_articles]
        logging.info(f"Nalezeno {len(members)} článků v kategorii {category_name}")

        # Spuštění save_worker threadu, pokud ještě neběží
        if not self.save_thread or not self.save_thread.is_alive():
            self.save_thread = threading.Thread(target=self.save_worker)
            self.save_thread.start()

        articles_count = 0
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = []
            for title in members:
                tasks.append(self.get_page_content_async(session, title))

            # Použití tqdm pro progress bar
            for result in await tqdm_asyncio.gather(*tasks):
                if result and len(result['text']) > MIN_ARTICLE_LENGTH:
                    self.article_buffer.put(result)
                    articles_count += 1

        return articles_count

    def scrape_category(self, category_name, max_articles=1000, batch_size=100):
        """Hlavní metoda pro stahování kategorie."""
        try:
            articles_count = asyncio.run(self.scrape_category_async(category_name, max_articles))
            
            # Počkáme chvíli, aby se buffer stihl vyprázdnit
            time.sleep(0.5)
            
            # Ukončení save_worker threadu
            self.is_running = False
            if self.save_thread:
                self.save_thread.join(timeout=30)  # Přidán timeout
                
            return articles_count
        finally:
            # Zajistíme, že is_running bude resetován
            self.is_running = True

    def split_into_chunks(self, sentences, max_length=2000):
        """Rozdělí seznam vět do chunků s maximální délkou."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Pokud je samotná věta delší než max_length, rozdělíme ji
            if sentence_length > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Rozdělení dlouhé věty na části
                words = sentence.split()
                temp_chunk = []
                temp_length = 0
                
                for word in words:
                    word_len = len(word) + 1  # +1 pro mezeru
                    if temp_length + word_len > max_length:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = word_len
                    else:
                        temp_chunk.append(word)
                        temp_length += word_len
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                continue
            
            # Pokud by přidání věty překročilo limit, uložíme současný chunk
            if current_length + sentence_length + 1 > max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 pro mezeru
        
        # Přidání posledního chunku
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def save_batch(self, articles, batch_num):
        """Uloží dávku článků do JSONL souboru, každý chunk jako samostatný záznam."""
        output_file = self.output_dir / f'wiki_batch_{batch_num}.jsonl'
        with output_file.open('a', encoding='utf-8') as f:
            for article in articles:
                if article and len(article['text']) > MIN_ARTICLE_LENGTH:
                    sentences = self.split_into_sentences(article['text'])
                    chunks = self.split_into_chunks(sentences)
                    
                    for chunk_num, chunk in enumerate(chunks, 1):
                        chunk_entry = {
                            'title': article['title'],
                            'text': chunk,
                            'url': article['url'],
                            'categories': article['categories'],
                            'metadata': {
                                'length': len(chunk),
                                'chunk_number': chunk_num,
                                'total_chunks': len(chunks),
                                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                                'language': 'cs',
                                'original_article_length': len(article['text'])
                            }
                        }
                        json.dump(chunk_entry, f, ensure_ascii=False)
                        f.write('\n')