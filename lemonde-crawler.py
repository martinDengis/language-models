from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading

# Download necessary NLTK data
nltk.download('punkt')

#-----------------------------------------
# Data structures
#-----------------------------------------
@dataclass
class ArticleData:
    """Data structure to hold article information"""
    url: str
    content: str
    word_count: int

class ThreadSafeCounter:
    """Thread-safe counter for tracking word count"""
    def __init__(self, target: int):
        self.count = 0
        self.target = target
        self.lock = Lock()
        
    def add(self, value: int) -> bool:
        """
        Adds value to counter if it won't exceed target.
        Returns True if value was added, False if adding would exceed target.
        """
        with self.lock:
            if self.count + value <= self.target * 1.05:  # Allow 5% overflow
                self.count += value
                return True
            return False
    
    def get_count(self) -> int:
        """Get current count"""
        with self.lock:
            return self.count

#-----------------------------------------
# Utils functions
#-----------------------------------------
def fetch_article_links(url: str) -> List[str]:
    """
    Fetches article links from a given LeMonde URL.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', class_='teaser__link')
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] - Links found: {len(links)}")
        return [link['href'] for link in links]
    except Exception as e:
        print(f"Error fetching links from {url}: {str(e)}")
        return []

def fetch_article_content(url: str) -> Optional[str]:
    """
    Fetches the content of an article from the given URL.
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p', class_='article__paragraph')
        return ' '.join([p.text for p in paragraphs])
    except Exception as e:
        print(f"Error fetching content from {url}: {str(e)}")
        return None

def process_text(text: str) -> str:
    """
    Processes the input text by tokenizing, converting to lowercase,
    and removing punctuation and numbers.
    """
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()]
    return ' '.join(tokens)

def process_article(url: str) -> Optional[ArticleData]:
    """
    Processes a single article URL and returns ArticleData if successful.
    """
    try:
        content = fetch_article_content(url)
        if not content:
            return None
            
        processed_content = process_text(content)
        word_count = len(processed_content.split())
        
        thread_name = threading.current_thread().name
        print(f"[{thread_name}] Processed article: {url} ({word_count:,} words)")
        
        return ArticleData(url, processed_content, word_count)
    except Exception as e:
        print(f"Error processing article {url}: {str(e)}")
        return None

#-----------------------------------------
# Crawler functions
#-----------------------------------------
def process_date(date: datetime, word_counter: ThreadSafeCounter, 
                results_queue: queue.Queue) -> None:
    """
    Processes all articles from a specific date.
    """
    thread_name = threading.current_thread().name
    base_url_template = 'https://www.lemonde.fr/archives-du-monde/{day:02d}-{month:02d}-{year}/{page}/'
    page = 1
    
    while word_counter.get_count() < word_counter.target:
        base_url = base_url_template.format(
            year=date.year,
            month=date.month,
            day=date.day,
            page=page
        )
        
        print(f"\n[{thread_name}] Processing {base_url}")
        article_links = fetch_article_links(base_url)
        
        if not article_links:
            break
            
        for link in article_links:
            if word_counter.get_count() >= word_counter.target:
                break
                
            article_data = process_article(link)
            if article_data and word_counter.add(article_data.word_count):
                results_queue.put(article_data)
            
            # Random delay between requests (shorter due to parallel nature)
            time.sleep(random.uniform(0.5, 1.5))
            
        page += 1

def crawl_lemonde(start_date: datetime, end_date: datetime, 
                 target_words: int = 700000, max_threads: int = 4) -> Tuple[str, int]:
    """
    Crawls LeMonde website using multiple threads until reaching the target word count.
    
    Args:
        start_date: Starting date for crawling
        end_date: Ending date for crawling
        target_words: Target number of words to collect
        max_threads: Maximum number of concurrent threads
    
    Returns:
        Tuple containing the corpus text and total word count
    """
    word_counter = ThreadSafeCounter(target_words)
    results_queue = queue.Queue()
    
    print(f"Starting multi-threaded crawl with target of {target_words:,} words")
    print(f"Using maximum of {max_threads} threads")
    
    # Create a list of dates to process
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Process dates using thread pool
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(process_date, date, word_counter, results_queue)
            for date in dates
        ]
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in thread: {str(e)}")
    
    # Collect all results from the queue
    all_articles = []
    while not results_queue.empty():
        article = results_queue.get()
        all_articles.append(article.content)
    
    final_text = ' '.join(all_articles)
    final_word_count = len(final_text.split())
    
    return final_text, final_word_count

#-----------------------------------------
# Main
#-----------------------------------------
if __name__ == '__main__':
    # Usage
    start_date = datetime(2012, 1, 1)  # Starting date
    end_date = datetime(2023, 12, 31)  # Ending date
    target_words = 700000
    max_threads = 7  # Adjust based on your needs and server limitations
    
    corpus, word_count = crawl_lemonde(
        start_date, 
        end_date, 
        target_words,
        max_threads
    )
    
    print(f"\nFinal corpus statistics:")
    print(f"- Total words collected: {word_count:,}")
    print(f"- Target word count: {target_words:,}")
    print(f"- Difference from target: {abs(word_count - target_words):,} words")
    
    # Save the corpus to a file
    with open('lemonde_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(corpus)
        print("\nCorpus successfully saved to lemonde_corpus.txt")