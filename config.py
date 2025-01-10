# config.py
import aiohttp
import asyncio
import logging

async def fetch_categories():
    """Asynchronně získá kategorie z Wikipedia API."""
    api_url = "https://cs.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "allcategories",
        "aclimit": 500,
        "format": "json"
    }
    
    categories = []
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extrahuje názvy kategorií z odpovědi
                    for category in data["query"]["allcategories"]:
                        categories.append(category["*"])
                else:
                    logging.error(f"Failed to fetch categories: {response.status}")
    except Exception as e:
        logging.error(f"Error fetching categories: {str(e)}")
    
    return categories

# Získání kategorií synchronně
def get_categories():
    return asyncio.run(fetch_categories())

SUBREDDITS = [
    'czech',
    'cesky',
    'realCzech',
    'Czechia',
    'Brno',
    'MenTy',
    'SirYakari'
]

MAX_CONVERSATIONS_PER_SUBREDDIT = 5000

# Konstanty pro scraping
MAX_ARTICLES_PER_CATEGORY = 20000
BATCH_SIZE = 100
MIN_ARTICLE_LENGTH = 1000  # znaků

# Dynamicky načtené kategorie
CATEGORIES = get_categories()