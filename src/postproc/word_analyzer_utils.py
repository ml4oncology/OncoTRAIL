from adjustText import adjust_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score
import re
import seaborn as sns
from ml_common.constants import CANCER_CODE_MAP
from scipy.stats import ranksums
from sklearn.metrics import roc_auc_score
# Download the stop words from NLTK
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
import os
nltk_data_path = '/cluster/home/t127556uhn/nltk_data'
nltk.data.path.append(nltk_data_path)

from rpy2 import robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
stats = importr('stats')

# Stop_word removal for the whole dataset
def filter_stop_words(text):
  stop_words = set(stopwords.words('english'))
  deid_terms = ['PATIENT', 'STAFF', 'AGE', 'DATE', 'PHONE', 'ID', 'EMAIL', 'PATORG', 'LOC', 'HOSP', 'OTHERPHI', 'OTHERISSUE']
  deid_terms = [s.lower() for s in deid_terms]

  stop_words = stop_words.union(deid_terms)
    
  # Split the report into words, filter out stop words, and rejoin into a string
  filtered_report = ' '.join(word for word in text.split() if word not in stop_words)

  # remove multiple words
  words_to_remove = ['dictated read', 'clinic note', 'cc cccce', 'md frcpc', 'date visit',
                     'next week', 'see back', 'clinic today', 'transcribed nt', 'princess margaret',
                     'plan mr', 'see week', 'last week', 'clinic week', 'consultation note', 'dictated dr',
                     'service dr', 'week time', 'dr job', 'dr job', 'time dictated', 'today show', 'dear', 'dr', 
                      'thank', 'transcribed', 'letter', 'visit' ]
  pattern = r'(?i)\b(?:' + '|'.join(re.escape(w) for w in words_to_remove) + r')\b'
  # pattern = '|'.join(re.escape(string) for string in words_to_remove)
  # Compile the regex pattern
  regex = re.compile(pattern)
  # Use sub method to replace all occurrences with an empty string
  filtered_report = regex.sub('', filtered_report)
  filtered_report = re.sub(r'\s+', ' ', filtered_report).strip()

  return filtered_report

def normalize_medical_terms(text):
    # Define domain-specific synonyms
    replacements = {
        'chemo': 'chemotherapy',
    }

    # Tokenize and replace if in mapping
    tokens = text.split()
    normalized_tokens = [replacements.get(tok, tok) for tok in tokens]
    return ' '.join(normalized_tokens)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

def lemmatization_func(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)

    lemmatized_words = []
    for word in tokens:
        # Try noun lemma first (for plural reduction)
        lemma = lemmatizer.lemmatize(word, wordnet.NOUN)
        # Then try verb lemma to reduce past/present tense variants
        lemma = lemmatizer.lemmatize(lemma, wordnet.VERB)
        lemmatized_words.append(lemma)

    return ' '.join(lemmatized_words)

def process_df(df, note_col):
    df[note_col] = df[note_col].apply(lambda x: str(x).lower())
    # remove special characters except for periods
    df[note_col] = df[note_col].apply(lambda x: re.sub(r'[^\w\s.]','', x)) 
    temp_stop_words = df[note_col].apply(lambda x: filter_stop_words(x))
    temp_lemma = temp_stop_words.apply(lemmatization_func)
    temp_lemma = temp_lemma.apply(normalize_medical_terms)
    df[f'{note_col}_lemmatized_note'] = temp_lemma.apply(lambda x: filter_stop_words(x))

    return df.copy()

def sentence_aware_analyzer(text, n=2):
    # 1. Split at sentence enders (., ?, !).  
    #    This gives you chunks you won’t cross when making n-grams.
    sentences = re.split(r'[\.?!]+', text)
    
    grams = []
    for sent in sentences:
        # 2. Tokenize each sentence into “words” (alphanumeric only):
        tokens = re.findall(r'\b\w+\b', sent.lower())
        
        # 3. Emit only the within-sentence n-grams:
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))
    return grams

def gen_bow(df, gram_val, text_data):
    # vectorizer = CountVectorizer(analyzer='word', ngram_range=(gram_val, gram_val))

    vectorizer = CountVectorizer(
      analyzer=lambda txt: sentence_aware_analyzer(txt, n=gram_val),
      lowercase=False,      # we handle lowercasing ourselves
      token_pattern=None    # disable default token-pattern/tokenizer
    )

    X_sum = vectorizer.fit_transform(df[text_data]).sum(axis=0)

    vocab = vectorizer.vocabulary_
    reverse_vocab = dict((v,k) for k,v in vocab.items())

    X_df_total_counts = pd.DataFrame({'total': np.asarray(X_sum)[0]})
    X_df_total_counts['word'] = (X_df_total_counts.index).map(reverse_vocab)

    # X_df_total_counts['has_char'] = X_df_total_counts['word'].apply(lambda input_string: any( char.isalpha() for char in input_string) )
    X_df_total_counts['has_number'] = X_df_total_counts['word'].apply(lambda input_string: any( char.isnumeric() for char in input_string) )
    X_df_total_counts = X_df_total_counts.loc[~X_df_total_counts['has_number'], ['total','word']]
    X_df_total_counts['percentage'] = X_df_total_counts['total']*100/X_df_total_counts['total'].sum()
    X_df_total_counts.sort_values(by='percentage', ascending=False, inplace=True)

    return X_df_total_counts

def calculate_pair_data(bow_df, df_notes, text_name_col, target_name_col, x_axis_metric, y_axis_metric):
    """
    Calculate metric changes and p-values for word presence/absence in notes.
    
    Parameters:
        bow_df: DataFrame with 'word' column
        df_notes: DataFrame with text and target columns
        text_name_col: Column name containing text
        target_name_col: Column name containing target variable
        x_axis_metric: Metric type ('auc', 'fdr', or 'average')
        y_axis_metric: Comparison type ('predictions' or 'residuals')
    
    Returns:
        DataFrame with calculated metrics
    """
    # Pre-tokenize all notes once (moved outside loop for efficiency)
    df_notes = df_notes.copy()
    df_notes['tokens'] = df_notes[text_name_col].astype(str).str.lower().str.findall(r"[a-zA-Z0-9']+")
    
    results = {
        'change_in_metric': [],
        'p_value': [],
        'n_metric_samples': [],
        'frequency': []
    }
    
    for word in bow_df['word']:
        word_lower = word.lower()
        df_notes['word_in_note'] = df_notes['tokens'].apply(lambda toks: int(word_lower in toks))
        
        df_present = df_notes[df_notes['word_in_note'] == 1]
        df_absent = df_notes[df_notes['word_in_note'] == 0]
        
        # Store metrics
        results['n_metric_samples'].append(min(len(df_present), len(df_absent)))
        results['frequency'].append(len(df_present) / len(df_notes))
        
        # Calculate change in metric
        change = _calculate_metric_change(df_present, df_absent, target_name_col, x_axis_metric)
        results['change_in_metric'].append(change)
        
        # Calculate p-value
        p_val = _calculate_p_value(df_present, df_absent, target_name_col, y_axis_metric)
        results['p_value'].append(p_val)
    
    # Add results to bow_df
    bow_df = bow_df.copy()
    for key, values in results.items():
        bow_df[key] = values
    
    # Filter out rows with NaN change_in_metric
    bow_df_plotting = bow_df.loc[bow_df['change_in_metric'].notna()].copy()
    
    return bow_df_plotting


def _calculate_metric_change(df_present, df_absent, target_col, metric_type):
    """Calculate the change in metric between present and absent groups."""
    if metric_type == 'auc':
        auc_present = _safe_auc(df_present, target_col)
        auc_absent = _safe_auc(df_absent, target_col)
        return auc_present - auc_absent
    
    elif metric_type == 'fdr':
        fdr_present = _calculate_fdr(df_present, target_col)
        fdr_absent = _calculate_fdr(df_absent, target_col)
        return fdr_present - fdr_absent
    
    elif metric_type == 'average':
        avg_present = df_present['Probability'].mean()
        avg_absent = df_absent['Probability'].mean()
        return avg_present - avg_absent
    
    return np.nan


def _safe_auc(df, target_col):
    """Safely calculate AUC, returning NaN on error."""
    try:
        return roc_auc_score(df[target_col], df['Probability'])
    except:
        return np.nan


def _calculate_fdr(df, target_col):
    """Calculate False Discovery Rate (1 - precision)."""
    predictions = (df['Probability'] >= 0.5).astype(int)
    precision = precision_score(df[target_col], predictions, zero_division=np.nan)
    return 1 - precision


def _calculate_p_value(df_present, df_absent, target_col, comparison_type):
    """Calculate p-value using rank-sum test."""
    if comparison_type == 'predictions':
        stat_present = df_present['Probability']
        stat_absent = df_absent['Probability']
    elif comparison_type == 'residuals':
        stat_present = df_present['Probability'] - df_present[target_col]
        stat_absent = df_absent['Probability'] - df_absent[target_col]
    else:
        return np.nan
    
    return ranksums(stat_present, stat_absent).pvalue

def process_data_for_volcano_plot(df_path, fname, target, note_type, n_gram_val, x_axis_metric, y_axis_metric, df_notes):
    
    df = pd.read_csv(f'{df_path}/{fname}')
    df = df.loc[df['Probability'].notna()].copy()
    df['treatment_date'] = pd.to_datetime(df['treatment_date']).dt.strftime('%Y-%m-%d')

    if len(df_notes) != 0:
        df = pd.merge(df, df_notes[['mrn', 'treatment_date', 'note']], on=['mrn', 'treatment_date'], how='left')

    df_proc = process_df(df, note_type)
    df_bow = gen_bow(df_proc, n_gram_val, f'{note_type}_lemmatized_note')
    df_bow = df_bow.loc[df_bow['total'] >= np.quantile(df_bow['total'], 0.9)].copy()
    df_bow = calculate_pair_data(df_bow, df_proc, f'{note_type}_lemmatized_note', target, x_axis_metric, y_axis_metric)

    # pvals = FloatVector(df_bow['p_value'].tolist())
    # adj_p = stats.p_adjust(pvals, method = adjust_method)
    # df_bow['p_adj'] = np.array(adj_p)
    # df_bow['neg_log_p'] = -np.log10(df_bow['p_adj'])

    return df_bow, df_proc

def adjust_p_values(df, adjust_method):
    pvals = FloatVector(df['p_value'].tolist())
    adj_p = stats.p_adjust(pvals, method=adjust_method)
    df['p_adj'] = np.array(adj_p)
    df['neg_log_p'] = -np.log10(df['p_adj'])

    return df

def generate_popularity_plot(df, title, note_type):
    # Filter to significant 1-grams
    df = df[df['p_adj'] < 0.05].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    # -----------------------
    # Select words to highlight
    # -----------------------
    top_pos = df.nlargest(10, 'change_in_metric')
    top_neg = df.nsmallest(10, 'change_in_metric')

    highlight_words = set(top_pos['word']).union(set(top_neg['word']))

    df['highlight'] = df['word'].isin(highlight_words)

    df_highlight = df[df['highlight']]
    df_background = df[~df['highlight']]

    # -----------------------
    # Marker sizes (constant for all points)
    # -----------------------
    size_bg = 50
    size_hi = 50

    # -----------------------
    # Background points (grey, no labels)
    # -----------------------
    ax.scatter(
        df_background['change_in_metric'],
        df_background['frequency'],
        s=size_bg,
        color='#B0B0B0',          # color-blind friendly neutral gray
        alpha=0.5,
        edgecolors='none',
        zorder=1
    )

    # -----------------------
    # Highlighted points (blue, labeled)
    # -----------------------
    ax.scatter(
        df_highlight['change_in_metric'],
        df_highlight['frequency'],
        s=size_hi,
        color="#0072B2",
        alpha=0.8,
        edgecolors='k',
        zorder=2
    )

    # -----------------------
    # Text labels (highlight only)
    # -----------------------
    texts = [
        ax.text(
            row['change_in_metric'],
            row['frequency'],
            row['word'],
            fontsize=12
        )
        for _, row in df_highlight.iterrows()
    ]

    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        force_points=0.5,
        force_text=0.5,
        lim=500,
        only_move={'points': 'y', 'text': 'xy'}
    )

    # -----------------------
    # Labels and title
    # -----------------------
    ax.set_xlabel('average Δ LLM risk prediction', fontsize=16)

    if note_type == 'note':
        y_label_string = 'clinical notes'
    elif note_type == 'Reason':
        y_label_string = 'LLM responses'

    ax.set_ylabel(f'proportion of {y_label_string} with 1-gram', fontsize=16)
    ax.set_title(title, fontsize=17)
    ax.grid(True)

    plt.tight_layout()
    plt.show()
    return fig


def plot_input_output_alignment(df, title):
    """
    Plot alignment between influential words from input (clinical notes)
    and output (LLM reasoning), colored by frequency ratio.
    
    Parameters
    ----------
    df : pd.DataFrame
        Columns: ["word", "change_in_metric_input", "change_in_metric_output", "frequency_ratio"]
    target : str
        Name of the adverse event, used in the title.
    """
    
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # ---- Normalize color scale around 1 ----
    norm = plt.Normalize(vmin=0.5, vmax=2.0)
    cmap = plt.get_cmap("coolwarm")  # diverging, colorblind-safe alternative: 'RdBu_r'
    
    # ---- Scatter plot ----
    sc = ax.scatter(
        df["change_in_metric_input"],
        df["change_in_metric_output"],
        c=df["frequency_ratio"],
        cmap=cmap,
        norm=norm,
        s=120,
        alpha=0.9,
        edgecolor="k",
        linewidth=0.4
    )
    
    # ---- Add word labels ----
    texts = []
    for _, row in df.iterrows():
        texts.append(
            ax.text(
                row["change_in_metric_input"],
                row["change_in_metric_output"],
                row["word"],
                fontsize=9,
                color="black"
            )
        )
    
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        expand_points=(1.2, 1.4),
        expand_text=(1.2, 1.4),
        force_points=0.5,
        force_text=0.5,
        lim=500,
        only_move={'points': 'y', 'text': 'xy'}
    )
    
    # ---- Reference lines ----
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=1)
    
    # ---- Labels and title ----
    ax.set_xlabel("Δ predicted risk (input clinical notes)")
    ax.set_ylabel("Δ predicted risk (LLM reasoning output)")
    ax.set_title(f"{title}", loc="left")
    
    # ---- Correlation ----
    corr = df[["change_in_metric_input", "change_in_metric_output"]].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {corr:.2f}", transform=ax.transAxes, fontsize=11)
    
    # ---- Colorbar ----
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("frequency ratio (output / input)")
    cbar.set_ticks([0.5, 1.0, 2.0])
    cbar.ax.axhline(1.0, color='k', lw=0.8)  # visual center
    
    # ---- Clean aesthetic ----
    sns.despine(ax=ax)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    return fig