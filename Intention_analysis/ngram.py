import pandas as pd
import re
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

# ========================= CONFIG =========================
DATA_DIR = Path('/home/workdir/attachments')
N_GRAM = 2          # Change to 3 for trigram, etc.
MIN_WORD_LEN = 2
TOP_K = 10

STOPWORDS = {
    'the', 'and',  'of', 'in', 'my', 'for', 'is', 'this', 'be', 'on', 'it', 'can',
    'by', 'with', 'a', 'an', 'i', 'want', 'that', 'have', 'will',
    'would', 'could', 'me', 'at', 'or', 'are', 'there', 'as', 'about', 'also', 'am', 'if'
}

def tokenize(text):
    if pd.isna(text) or not str(text).strip():
        return []
    text = str(text).lower().strip()
    # Korean + English tokens
    tokens = re.findall(r'[\uac00-\ud7a3]+|\b\w+\b', text)
    return [t for t in tokens if len(t) >= MIN_WORD_LEN and t not in STOPWORDS]

# ========================= LOAD & PREPARE CORPUS =========================
df = pd.read_csv("/Users/vscalar/project/M.I.N.D/ds_cleaned_dataset/discussion_1.csv")

# Combine Q1 and Q2 (you can separate them later if needed)

# Tokenize all documents
all_tokens = []
for text in df['Q1']:
    all_tokens.extend(tokenize(text))

print(f"Total tokens in corpus: {len(all_tokens)}")
print(f"Unique tokens: {len(set(all_tokens))}\n")

# ========================= BUILD N-GRAM MODEL =========================
def build_ngram_model(tokens, n=N_GRAM):
    ngram_counts = defaultdict(Counter)
    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i:i+n-1])
        next_token = tokens[i+n-1]
        ngram_counts[context][next_token] += 1
    return ngram_counts

ngram_model = build_ngram_model(all_tokens, N_GRAM)

# ========================= NEXT TOKEN PROBABILITY =========================
def next_token_probabilities(context_tokens, model, top_k=TOP_K):
    context = tuple(context_tokens[-(N_GRAM-1):])  # last (n-1) tokens
    if context not in model:
        print(f"No data for context: {context}")
        return []
    
    counts = model[context]
    total = sum(counts.values())
    
    probs = [(token, count / total) for token, count in counts.most_common(top_k)]
    return probs

# ========================= INTERACTIVE PREDICTION =========================
print("=== Next Token Probability Predictor ===")
print(f"Using {N_GRAM}-gram model\n")

while True:
    user_input = input("Enter some tokens (space separated) or 'quit': ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    context_tokens = tokenize(user_input)
    if not context_tokens:
        print("Please enter valid tokens.\n")
        continue
    
    probs = next_token_probabilities(context_tokens, ngram_model)
    
    print(f"\nContext: {' '.join(context_tokens[-3:])}")
    print(f"Top {len(probs)} next token probabilities:")
    for token, prob in probs:
        print(f"  {token:18} : {prob:.4f} ({int(prob*100)}%)")
    print("-" * 50)