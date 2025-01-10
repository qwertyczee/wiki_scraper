import aiohttp
import asyncio
import json
from pathlib import Path
import logging
from tqdm.asyncio import tqdm_asyncio
import re
import time
from config import MAX_CONVERSATIONS_PER_SUBREDDIT, BATCH_SIZE

class RedditScraper:
    def __init__(self, output_dir='conversation_dataset'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headers = {
            'User-Agent': 'ConversationCollector/1.0'
        }
        
    async def scrape_subreddit(self, subreddit, limit=1000):
        """Scrape conversations from a subreddit."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"https://old.reddit.com/r/{subreddit}/top.json?limit={limit}&t=all"
            
            try:
                async with session.get(url) as response:
                    data = await response.json()
                    posts = data['data']['children']
                    
                    conversations = []
                    for post in posts:
                        post_data = post['data']
                        if post_data['num_comments'] > 5:
                            comments_url = f"https://old.reddit.com{post_data['permalink']}.json"
                            async with session.get(comments_url) as comments_response:
                                comments_data = await comments_response.json()
                                conversation = self._extract_conversation_thread(comments_data[1])
                                if conversation:
                                    conversations.append(conversation)
                    
                    return conversations
            except Exception as e:
                logging.error(f"Chyba při scrapování subredditu {subreddit}: {str(e)}")
                return []

    def _extract_conversation_thread(self, comments_data):
        """Extract meaningful conversation threads from comments."""
        conversation = []
        
        def process_comment(comment):
            if 'body' in comment['data'] and not comment['data']['body'].startswith('[deleted]'):
                text = self._clean_text(comment['data']['body'])
                # Kontrola češtiny - aspoň 40% textu musí být české znaky
                if self._is_czech_text(text) and len(text.split()) > 3:
                    conversation.append({
                        'text': text,
                        'score': comment['data'].get('score', 0)
                    })
                
                replies = comment['data'].get('replies', {})
                if replies and 'data' in replies:
                    for reply in replies['data']['children']:
                        process_comment(reply)
        
        for comment in comments_data['data']['children']:
            process_comment(comment)
            
        return conversation if len(conversation) >= 2 else None

    def _clean_text(self, text):
        """Clean and normalize conversation text."""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _is_czech_text(self, text):
        """Check if text is predominantly Czech."""
        czech_chars = set('áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ')
        text_chars = set(text.lower())
        return len(text_chars.intersection(czech_chars)) >= 2

    def save_conversations(self, conversations, subreddit, batch_num):
        """Save scraped conversations to JSONL format."""
        output_file = self.output_dir / f'conversations_{subreddit}_batch_{batch_num}.jsonl'
        with output_file.open('w', encoding='utf-8') as f:
            for conv in conversations:
                json.dump({
                    'source': 'reddit',
                    'subreddit': subreddit,
                    'conversation': conv,
                    'metadata': {
                        'format_version': '1.0',
                        'language': 'cs',
                        'quality_score': sum(msg.get('score', 0) for msg in conv)
                    }
                }, f, ensure_ascii=False)
                f.write('\n')

    async def scrape_all_subreddits(self, subreddits):
        """Scrape all specified subreddits."""
        for subreddit in subreddits:
            logging.info(f"Začínám stahovat konverzace z r/{subreddit}")
            conversations = await self.scrape_subreddit(
                subreddit,
                limit=MAX_CONVERSATIONS_PER_SUBREDDIT
            )
            self.save_conversations(conversations, subreddit, 1)
            logging.info(f"Staženo {len(conversations)} konverzací z r/{subreddit}")
            await asyncio.sleep(0.05)  # Prevent rate limiting