from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
import time
import random

# Download necessary NLTK data
nltk.download('punkt_tab')

#-----------------------------------------
# Utils functions
#-----------------------------------------
def fetch_article_links(url):
    """
    Fetches article links from a given Reuters URL.

    This function sends a GET request to the specified URL, parses the HTML content,
    and extracts all article links that belong to the 'world' section of the Reuters website.

    Args:
        url (str): The URL of the Reuters page to fetch article links from.

    Returns:
        list: A list of full URLs to the articles in the 'world' section.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', class_='teaser__link')
    print("- Links found: ", len(links))
    return [link['href'] for link in links]


def fetch_article_content(url):
    """
    Fetches the content of an article from the given URL.

    This function sends a GET request to the specified URL, parses the HTML content,
    and extracts the text from all paragraph elements with a specific class.

    Args:
        url (str): The URL of the article to fetch.

    Returns:
        str: The concatenated text content of the article's paragraphs.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p', class_='article__paragraph')
    return ' '.join([p.text for p in paragraphs])


def process_text(text):
    """
    Processes the input text by tokenizing, converting to lowercase, and removing punctuation and numbers.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The processed text with tokens joined by spaces.
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]
    return ' '.join(tokens)


#-----------------------------------------
# Crawler functions
#-----------------------------------------
def crawl_lemonde(start_date, end_date, num_articles=100):
    """
    Crawls Reuters website to fetch and process a specified number of articles.

    Args:
        num_articles (int): The number of articles to fetch and process. Default is 100.

    Returns:
        str: A single string containing the processed text of all fetched articles.

    The function performs the following steps:
    1. Initializes the base URL for Reuters world news and an empty list to store article texts.
    2. Fetches article links from the base URL until the desired number of articles is reached.
    3. For each article link, fetches the article content, processes the text, and appends it to the list.
    4. Waits for a random interval between 1 to 3 seconds between requests to avoid overloading the server.
    5. Handles exceptions that may occur during the fetching and processing of articles.
    6. Returns the concatenated processed text of all articles as a single string.
    """
    base_url_template = 'https://www.lemonde.fr/archives-du-monde/{day:02d}-{month:02d}-{year}/{page}/'
    all_text = []
    article_count = 0

    while article_count < num_articles and start_date <= end_date:
        page = 1
        while article_count < num_articles:
            base_url = base_url_template.format(year=start_date.year, month=start_date.month, day=start_date.day, page=page)
            print("\n----------")
            print(f"Fetching articles from {base_url}...")
            print("----------")
            article_links = fetch_article_links(base_url)

            if not article_links:
                break  # No more articles on this page, move to the next date

            for link in article_links:
                if article_count >= num_articles:
                    print("----------")
                    print(f"Reached {num_articles} articles.")
                    break

                try:
                    content = fetch_article_content(link)
                    processed_content = process_text(content)
                    all_text.append(processed_content)
                    article_count += 1
                    print(f"Processed article {article_count}: {link}")

                    time.sleep(random.uniform(1, 3))  # avoid overloading the server
                except Exception as e:
                    print(f"Error processing {link}: {str(e)}")

            page += 1

        start_date += timedelta(days=1)

    return ' '.join(all_text)


#-----------------------------------------
#Main
#-----------------------------------------
if __name__ == '__main__':
    # Usage
    start_date = datetime(2023, 1, 1)  # Starting date
    end_date = datetime(2023, 12, 31)  # Ending date
    corpus = crawl_lemonde(start_date, end_date, num_articles=100)
    print(f"Corpus length: {len(corpus.split())}")

    # Save the corpus to a file
    with open('lemonde_corpus.txt', 'w', encoding='utf-8') as f:
        f.write(corpus)
        print("Corpus successfully saved.")
