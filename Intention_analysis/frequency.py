import pandas as pd
import re, os
from collections import Counter
import numpy as np
from pathlib import Path
from math import log

# ========================= CONFIG =========================
DATA_DIR = Path('/Users/vscalar/project/M.I.N.D/ds_cleaned_dataset')

MIN_WORD_LEN = 2
TOP_N = 30

# Basic stopwords (expand as needed)
STOPWORDS = {
    'the', 'and', 'to', 'of', 'in', 'my', 'for', 'is', 'this', 'be', 'on', 'it', 'can',
    'by', 'with', 'a', 'an', 'i', 'want', 'that', 'have', 'will',
    'would', 'could', 'me', 'at', 'or', 'are', 'there', 'as', 'about', 'also', 'am', 'if'
}

def tokenize(text, remove_stopwords=True):
    if pd.isna(text) or not isinstance(text, str):
        return []
    text = text.lower().strip()
    # Keep Korean words as single tokens + English words
    tokens = re.findall(r'[\uac00-\ud7a3]+|\b\w+\b', text)
    
    if remove_stopwords:
        return [t for t in tokens if len(t) >= MIN_WORD_LEN and t not in STOPWORDS]
    else:
        # For 2-grams with stopwords, we might want to keep single-letter stopwords like 'a' or 'i'
        return [t for t in tokens]

def generate_bigrams(tokens):
    """Generates 2-gram pairs from a list of tokens."""
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

# ========================= LOAD DATA =========================
def load_csv(file_path):
    df = pd.read_csv(file_path)

    # Clean
    for col in ['Q1', 'Q2']:
        if col in df.columns:
            # Mark as NaN if cell contains the word "noisy"
            df[col] = df[col].where(~df[col].astype(str).str.contains('noisy', case=False, na=False), 
                                    other=pd.NA)
            
    # Clean and create cleaned columns
    df['Q1_clean'] = df['Q1'].fillna('').astype(str)
    df['Q2_clean'] = df['Q2'].fillna('').astype(str)
    
    # 1. Tokenize for Unigrams (w/o stopwords)
    q1_tokens_list = [tokenize(text, remove_stopwords=True) for text in df['Q1_clean']]
    q2_tokens_list = [tokenize(text, remove_stopwords=True) for text in df['Q2_clean']]
    all_q1_tokens = [t for sublist in q1_tokens_list for t in sublist]
    all_q2_tokens = [t for sublist in q2_tokens_list for t in sublist]

    # 2. Tokenize and Generate 2-Grams (with stopwords)
    q1_bigrams_list = [generate_bigrams(tokenize(text, remove_stopwords=False)) for text in df['Q1_clean']]
    q2_bigrams_list = [generate_bigrams(tokenize(text, remove_stopwords=False)) for text in df['Q2_clean']]
    all_q1_bigrams = [bg for sublist in q1_bigrams_list for bg in sublist]
    all_q2_bigrams = [bg for sublist in q2_bigrams_list for bg in sublist]

    return (q1_tokens_list, q2_tokens_list, all_q1_tokens, all_q2_tokens, 
            q1_bigrams_list, q2_bigrams_list, all_q1_bigrams, all_q2_bigrams, df)

# ========================= TF-IDF (Unique Words) =========================
def compute_tfidf(tokens_list):
    all_tokens = [t for sublist in tokens_list for t in sublist]
    vocab = list(set(all_tokens))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    N = len(tokens_list)
    
    # Document Frequency
    df_count = np.zeros(len(vocab))
    for tokens in tokens_list:
        seen = set(tokens)
        for word in seen:
            if word in word_to_idx:
                df_count[word_to_idx[word]] += 1
    
    # TF-IDF
    word_importance = np.zeros(len(vocab))
    for tokens in tokens_list:
        if not tokens:
            continue
        tf = Counter(tokens)
        for word, count in tf.items():
            if word in word_to_idx:
                idx = word_to_idx[word]
                tf_val = count / len(tokens)
                idf_val = log(N / (df_count[idx] + 1)) + 1
                word_importance[idx] += tf_val * idf_val
    
    word_importance /= N
    tfidf_scores = [(vocab[i], round(score, 4)) for i, score in enumerate(word_importance) if score > 0.001]
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    return tfidf_scores


def list_csv_files(root_dir):
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

csv_files = list_csv_files("/Users/vscalar/project/M.I.N.D/ds_cleaned_dataset")
csv_files.sort()
print(csv_files)

i = 1
for csv_file in csv_files:
    (q1_tokens_list, q2_tokens_list, all_q1_tokens, all_q2_tokens,
     q1_bigrams_list, q2_bigrams_list, all_q1_bigrams, all_q2_bigrams, df) = load_csv(csv_file)
     
    output_file_q1 = "/Users/vscalar/project/M.I.N.D/Intention_analysis/" + f"discussion{i}_q1"
    output_file_q2 = "/Users/vscalar/project/M.I.N.D/Intention_analysis/" + f"discussion{i}_q2"
    i += 1
    
    # Write Q1 results
    with open(output_file_q1, 'w', encoding='utf-8') as f:
        f.write("=== Q1: Top Frequent Words ===\n")
        for word, count in Counter(all_q1_tokens).most_common(TOP_N):
            f.write(f"{word:18} : {count:3d}\n")

        f.write("\n=== Q1: Top Unique Words (TF-IDF) ===\n")
        for word, score in compute_tfidf(q1_tokens_list)[:TOP_N]:
            f.write(f"{word:20} : {score:.4f}\n")
            
        f.write("\n=== Q1: Top 2-Gram Pairs (with Stop Words) ===\n")
        for bigram, count in Counter(all_q1_bigrams).most_common(TOP_N):
            f.write(f"{bigram:25} : {count:3d}\n")

    # Write Q2 results
    with open(output_file_q2, 'w', encoding='utf-8') as f:
        f.write("\n=== Q2: Top Frequent Words ===\n")
        for word, count in Counter(all_q2_tokens).most_common(TOP_N):
            f.write(f"{word:18} : {count:3d}\n")

        f.write("\n=== Q2: Top Unique Words (TF-IDF) ===\n")
        for word, score in compute_tfidf(q2_tokens_list)[:TOP_N]:
            f.write(f"{word:20} : {score:.4f}\n")
            
        f.write("\n=== Q2: Top 2-Gram Pairs (with Stop Words) ===\n")
        for bigram, count in Counter(all_q2_bigrams).most_common(TOP_N):
            f.write(f"{bigram:25} : {count:3d}\n")