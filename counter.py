#!/usr/bin/env python3
import sys
from pathlib import Path
import re
from collections import Counter
from multiprocessing import Pool, cpu_count
import itertools
from typing import List, Dict, Tuple, Iterator
import math

def chunk_text(text: str, num_chunks: int) -> List[str]:
    """
    Split text into roughly equal chunks for parallel processing.
    """
    chunk_size = math.ceil(len(text) / num_chunks)
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        # Adjust chunk boundary to not split words
        if i + chunk_size < len(text):
            last_space = chunk.rfind(' ')
            if last_space != -1:
                chunk = chunk[:last_space]
                i = i + last_space
        chunks.append(chunk)
    
    return chunks

def map_words(text: str) -> List[Tuple[str, int]]:
    """
    Map function that splits text into words and returns (word, 1) pairs.
    """
    # Split text into words and clean them
    words = text.lower().split()
    result = []
    # Filter out empty strings and strings containing only punctuation
    for word in words:
        word = re.sub(r'[^a-z]', '', word)
        if word:  # Only emit non-empty words
            result.append((word, 1))
    return result

def reduce_words(mapped_values: List[Tuple[str, int]]) -> Dict[str, int]:
    """
    Reduce function that counts word occurrences from mapped values.
    """
    counter = Counter()
    for word, count in mapped_values:
        counter[word] += count
    return dict(counter)

def process_chunk(chunk: str) -> List[Tuple[str, int]]:
    """
    Process a single chunk of text. This function will be used with Pool.map().
    """
    return map_words(chunk)

def filter_and_replace_words(text: str, word_counts: Dict[str, int], min_occurrences: int = 3) -> str:
    """
    Filter out words that appear less than min_occurrences times.
    """
    # Create set of words to remove
    words_to_remove = {word for word, count in word_counts.items() 
                      if count < min_occurrences}
    
    # Process text word by word
    words = text.split()
    filtered_words = []
    
    for word in words:
        word_lower = re.sub(r'[^a-z]', '', word.lower())
        if word_lower not in words_to_remove:
            filtered_words.append(word)
    
    return ' '.join(filtered_words)

def process_file(file_path: str, min_occurrences: int = 3) -> Tuple[Dict[str, int], int, int]:
    """
    Process a text file using MapReduce to count and filter words.
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            original_line_count = len(text.splitlines())
        
        # Determine number of processes to use (one per CPU core)
        num_processes = cpu_count()
        
        # Split text into chunks for parallel processing
        chunks = chunk_text(text, num_processes)
        
        # Create process pool and process chunks
        with Pool(processes=num_processes) as pool:
            # Map phase - process all chunks in parallel
            mapped_values = pool.map(process_chunk, chunks)
            # Flatten the list of lists
            mapped_values = list(itertools.chain.from_iterable(mapped_values))
            
            # Reduce phase
            word_counts = reduce_words(mapped_values)
            
            # Filter and create new text
            filtered_text = filter_and_replace_words(text, word_counts, min_occurrences)
            
            # Count words in filtered text
            filtered_word_count = len(filtered_text.split())
            
            # Save filtered text
            output_path = file_path.parent / 'cleaned_lemonde_corpus.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(filtered_text)
            
            return word_counts, filtered_word_count, original_line_count
            
    except UnicodeDecodeError:
        print(f"Error: Unable to read file {file_path}. Please ensure it's a valid text file with UTF-8 encoding.")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def main():
    """
    Main function to process lemonde_corpus.txt using MapReduce.
    """
    file_path = "lemonde_corpus.txt"
    min_occurrences = 3
    
    try:
        print(f"\nProcessing {file_path} using MapReduce...")
        print(f"Using {cpu_count()} CPU cores")
        print(f"Minimum word occurrences: {min_occurrences}")
        
        word_counts, filtered_word_count, line_count = process_file(
            file_path, 
            min_occurrences
        )
        
        # Calculate statistics
        total_unique_words = len(word_counts)
        removed_words = sum(1 for count in word_counts.values() if count < min_occurrences)
        
        print("\nFile Statistics:")
        print("-" * 50)
        print(f"Original unique words: {total_unique_words:,}")
        print(f"Words removed (< {min_occurrences} occurrences): {removed_words:,}")
        print(f"Remaining unique words: {total_unique_words - removed_words:,}")
        print(f"Words in cleaned corpus: {filtered_word_count:,}")
        print(f"Original line count: {line_count:,}")
        print(f"Average words per line: {filtered_word_count/line_count:.1f}")
        print("\nTop 10 most frequent words:")
        print("-" * 50)
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{word}: {count:,}")
            
        print(f"\nCleaned corpus saved to: cleaned_lemonde_corpus.txt")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()