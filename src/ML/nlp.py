import re
import os
import nltk
from oncotrail.utils.env_loader import load_env
load_env()
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.data.path.append(os.environ.get("NLTK_DATA_DIR", ""))

def get_wordnet_pos(treebank_tag):
    """Map POS tags to WordNet POS tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def lemmatization_func(text):
    """Lemmatize text using NLTK with POS tagging."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]
    return ' '.join(lemmatized_words)

def filter_stop_words(text):
    """Remove default + domain-specific stopwords and common DEID terms."""
    # Standard English stopwords + DEID placeholders
    stop_words = set(stopwords.words('english'))
    deid_terms = [
        'PATIENT', 'STAFF', 'AGE', 'DATE', 'PHONE', 'ID', 'EMAIL', 'PATORG',
        'LOC', 'HOSP', 'OTHERPHI', 'OTHERISSUE'
    ]
    stop_words.update([s.lower() for s in deid_terms])

    # Remove exact stopwords
    filtered_report = ' '.join(word for word in text.split() if word not in stop_words)

    # Remove common meaningless multi-word phrases (regex-based)
    phrases_to_remove = [
        'dictated read', 'clinic note', 'cc cccce', 'md frcpc', 'date visit',
        'next week', 'see back', 'clinic today', 'transcribed nt',
        'princess margaret', 'plan mr', 'see week', 'last week', 'clinic week',
        'consultation note', 'dictated dr', 'service dr', 'week time',
        'dr job', 'time dictated', 'today show', 'dear', 'dr', 'thank',
        'transcribed', 'letter', 'visit', 'md', 'fracp', 'medical record report',
        'mr', 'ms', 'mrs', 'cc'
    ]
    # add months of the year and their abbreviation to phrases_to_remove
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    for month in months:
        phrases_to_remove.append(month)
        phrases_to_remove.append(month[:3])
        
    pattern = r'(?i)\b(?:' + '|'.join(re.escape(p) for p in phrases_to_remove) + r')\b'
    filtered_report = re.sub(pattern, '', filtered_report)
    filtered_report = re.sub(r'\s+', ' ', filtered_report).strip()

    return filtered_report

def process_df(df, note_col):
    """Full preprocessing pipeline: lowercasing, cleaning, lemmatization."""
    df[note_col] = df[note_col].astype(str).str.lower()
    df[note_col] = df[note_col].apply(lambda x: re.sub(r'[^\w\s.]', '', x))  # keep periods
    step1_cleaned = df[note_col].apply(filter_stop_words)
    lemmatized = step1_cleaned.apply(lemmatization_func)
    final_cleaned = lemmatized.apply(filter_stop_words)
    df[f'{note_col}_lemmatized_note'] = final_cleaned
    return df.copy()

def sentence_aware_analyzer(text):
    """Generates unigrams and bigrams while respecting sentence boundaries."""
    sentences = re.split(r'[\.?!]+', text)
    grams = []
    for sent in sentences:
        tokens = re.findall(r'\b\w+\b', sent.lower())
        grams.extend(tokens)  # unigrams
        grams.extend([" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)])  # bigrams
    return grams

def extract_top_ngrams(df, text_col, top_k=300):
    """
    Extracts top-k TF-IDF unigrams and bigrams from a column of preprocessed text.
    Filters out digits and 1-character unigrams.
    Returns a list of vocabulary terms.
    """
    vectorizer = TfidfVectorizer(
        analyzer=sentence_aware_analyzer,
        lowercase=False,
        token_pattern=None,
        max_features=None
    )
    
    tfidf_matrix = vectorizer.fit_transform(df[text_col])
    tfidf_sums = tfidf_matrix.sum(axis=0).A1
    feature_names = vectorizer.get_feature_names_out()

    # Build DataFrame of terms and scores
    tfidf_df = pd.DataFrame({
        'ngram': feature_names,
        'tfidf_sum': tfidf_sums
    })

    # Filter out:
    tfidf_df = tfidf_df[~tfidf_df['ngram'].str.contains(r'\d')]  # n-grams with digits
    tfidf_df = tfidf_df[~tfidf_df['ngram'].str.match(r'^[a-zA-Z]$')]  # single-letter unigrams

    # Rank by cumulative TF-IDF and return top_k n-grams
    tfidf_df = tfidf_df.sort_values(by='tfidf_sum', ascending=False).head(top_k)
    vocab = tfidf_df['ngram'].tolist()

    return vocab

def build_tfidf_matrix(df, text_col, vocabulary):
    """
    Constructs a TF-IDF feature matrix using a fixed vocabulary.
    
    Parameters:
        df (pd.DataFrame): DataFrame with preprocessed notes
        text_col (str): Column name of cleaned/lemmatized notes
        vocabulary (List[str]): Top n-grams to use as features
    
    Returns:
        X (csr_matrix): TF-IDF feature matrix (n_notes x len(vocab))
        vectorizer (TfidfVectorizer): Fitted vectorizer for reference
    """
    vectorizer = TfidfVectorizer(
        analyzer=sentence_aware_analyzer,
        lowercase=False,
        token_pattern=None,
        vocabulary=vocabulary
    )

    X = vectorizer.fit_transform(df[text_col])  # you can .transform() later on new sets
    return X, vectorizer

def build_count_matrix(df, text_col, vocabulary):
    """
    Constructs a CountVectorizer feature matrix using a fixed vocabulary.

    Parameters:
        df (pd.DataFrame): DataFrame with preprocessed notes
        text_col (str): Column name of cleaned/lemmatized notes
        vocabulary (List[str]): Top n-grams to use as features

    Returns:
        X (csr_matrix): Count-based feature matrix (n_notes x len(vocab))
        vectorizer (CountVectorizer): Fitted vectorizer for reference
    """
    vectorizer = CountVectorizer(
        analyzer=sentence_aware_analyzer,
        lowercase=False,
        token_pattern=None,
        vocabulary=vocabulary
    )

    X = vectorizer.fit_transform(df[text_col])  # you can .transform() later on new sets
    return X, vectorizer