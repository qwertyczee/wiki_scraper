import logging
from wiki_scraper import WikiScraper
from dataset_processor import DatasetProcessor
from config import CATEGORIES, MAX_ARTICLES_PER_CATEGORY, BATCH_SIZE

def main():
    # Inicializace scraperu
    scraper = WikiScraper()
    
    # Stažení článků z každé kategorie
    for category in CATEGORIES:
        logging.info(f"Začínám stahovat kategorii: {category}")
        scraper.scrape_category(
            category,
            max_articles=MAX_ARTICLES_PER_CATEGORY,
            batch_size=BATCH_SIZE
        )
    
    # Zpracování datasetu
    processor = DatasetProcessor()
    splits = processor.split_dataset()
    
    logging.info("Hotovo!")
    logging.info(f"Trénovací set: {len(splits['train'])} článků")
    logging.info(f"Validační set: {len(splits['val'])} článků")
    logging.info(f"Testovací set: {len(splits['test'])} článků")

if __name__ == "__main__":
    main()