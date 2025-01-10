import logging
import argparse
from wiki_scraper import WikiScraper
from reddit_scraper import RedditScraper
from dataset_processor import DatasetProcessor
from config import CATEGORIES, MAX_ARTICLES_PER_CATEGORY, BATCH_SIZE, SUBREDDITS
import asyncio

def main():
    parser = argparse.ArgumentParser(description='Stažení a zpracování dat pro jazykový model')
    parser.add_argument('action', choices=['scrape', 'scrape_conversations', 'process'],
                    help='Akce ke spuštění: "scrape" pro wiki, "scrape_conversations" pro konverzace, "process" pro zpracování')
    args = parser.parse_args()

    if args.action == 'scrape':
        if not CATEGORIES:
            logging.error("Nepodařilo se načíst kategorie z Wikipedia API")
            return
            
        # Inicializace wiki scraperu
        scraper = WikiScraper()
        total_articles = 0
        
        # Stažení článků z každé kategorie
        for category in CATEGORIES:
            logging.info(f"Začínám stahovat kategorii: {category}")
            articles_count = scraper.scrape_category(
                category,
                max_articles=MAX_ARTICLES_PER_CATEGORY,
                batch_size=BATCH_SIZE
            )
            total_articles += articles_count
            logging.info(f"Staženo {articles_count} článků z kategorie {category}")
            
        logging.info("Stahování wiki dokončeno!")
        logging.info(f"Celkem staženo {total_articles} článků")

    elif args.action == 'scrape_conversations':
        # Inicializace reddit scraperu
        reddit_scraper = RedditScraper()
        
        # Asynchronní stažení konverzací
        asyncio.run(reddit_scraper.scrape_all_subreddits(SUBREDDITS))
        logging.info("Stahování konverzací dokončeno!")

    elif args.action == 'process':
        # Zpracování všech stažených dat
        logging.info("Začínám zpracovávat dataset...")
        processor = DatasetProcessor()
        splits = processor.split_dataset()
        
        logging.info("Zpracování dokončeno!")
        logging.info(f"Trénovací set: {len(splits['train'])} záznamů")
        logging.info(f"Validační set: {len(splits['val'])} záznamů")
        logging.info(f"Testovací set: {len(splits['test'])} záznamů")

if __name__ == "__main__":
    main()