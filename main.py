import logging
import argparse
from wiki_scraper import WikiScraper
from dataset_processor import DatasetProcessor
from config import (
    CATEGORIES, CATEGORIES_1, CATEGORIES_2, CATEGORIES_3, 
    CATEGORIES_4, CATEGORIES_5, CATEGORIES_6, CATEGORIES_7, 
    CATEGORIES_8, CATEGORIES_9, MAX_ARTICLES_PER_CATEGORY, BATCH_SIZE
)

def get_category_group(group_num):
    categories = {
        1: CATEGORIES_1,
        2: CATEGORIES_2,
        3: CATEGORIES_3,
        4: CATEGORIES_4,
        5: CATEGORIES_5,
        6: CATEGORIES_6,
        7: CATEGORIES_7,
        8: CATEGORIES_8,
        9: CATEGORIES_9,
        10: CATEGORIES
    }
    return categories.get(group_num)

def main():
    # Nastavení argumentů příkazové řádky
    parser = argparse.ArgumentParser(description='Stažení článků z Wikipedie podle kategorií')
    parser.add_argument('category_group', type=int, choices=range(1,11),
                    help='Číslo skupiny kategorií (1-10)')
    args = parser.parse_args()

    # Získání vybrané skupiny kategorií
    selected_categories = get_category_group(args.category_group)
    if not selected_categories:
        logging.error(f"Neplatná skupina kategorií: {args.category_group}")
        return

    # Inicializace scraperu
    scraper = WikiScraper()
    
    # Stažení článků z každé kategorie
    for category in selected_categories:
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