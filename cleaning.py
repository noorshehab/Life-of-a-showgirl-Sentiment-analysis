#%% [code]
#Imports and Cleaning functions
import re
import html
import pandas as pd
from langdetect import detect, LangDetectException

# Cleaning functions (customized)
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    # HTML unescape
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@\w+', '', text)
    # Remove “#” but keep the word
    text = re.sub(r'#', '', text)

    # Remove “RT”
    text = re.sub(r'\bRT\b', '', text)
    # Remove punctuation (except ! & ?)
    text = re.sub(r"[^\w\s\!\?]", " ", text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def remove_terms(text, terms):
    """
    Remove whole-word occurrences of each term in `terms` from `text`.
    Uses word boundaries and is case-insensitive.
    """
    import re
    if not isinstance(text, str):
        return text
    t = text
    for term in terms:
        # Lowercase the term (for case-insensitive matching)
        term_esc = re.escape(term.lower())
        # pattern: word boundary + term + word boundary
        pattern = rf"\b{term_esc}\b"
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)
    # Clean up extra whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_minimal(text, min_words=5):
    if not text:
        return False
    return len(text.split()) >= min_words

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
    
def clean_search_term(text):
    #map the search term column to the terms in the search term list
    text = text.lower()
    for term in SEARCH_TERMS:
        if term in text:
            return term 

def read_csv(input_csv):
    df = pd.read_csv(input_csv)
    df = df[COLUMNS].copy()
    df['searchTerms'] = df['searchTerms'].astype(str).apply(clean_search_term)
    return df

def clean_text(df):
    df = df.dropna(subset=['author/description'])
    df['cleaned'] = df[text_col].apply(lambda t: clean_tweet(t))
    df['is_minimal'] = df['cleaned'].apply(is_minimal)
    df['is_english'] = df['cleaned'].apply(is_english)
    df['masked'] = df['cleaned'].apply(lambda txt: remove_terms(txt, SEARCH_TERMS))
    return df

def create_dataset(dfs):
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    combined = combined.dropna(subset=['author/description'])
    return combined






# %% [code]
#Constants
COLUMNS=['author/description','author/screen_name','author/name','author/location', 'full_text','searchTerms','favorite_count','retweet_count','reply_count','quote_count','created_at']
text_col = 'full_text'  
CSV_NAMES=[r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-10_10-39-15-248.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-06_20-30-50-967.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-06_21-05-15-811.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-06_21-03-08-626.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-06_21-22-29-835.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-07_17-40-58-412.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-08_11-03-27-968.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-08_11-41-46-517.csv',
r'C:\Users\Hatem\OneDrive\Desktop\Swiftieeraoverinshallah\CSVs\dataset_twitter-x-scraper_2025-10-06_19-33-50-822.csv']

SEARCH_TERMS=['life of a showgirl',
              'taylor swift',
              'taylor',
              'new album',
              'album',
              'actually romantic',
              'fate of ophelia',
              'opalite',
              'wood',
              'elizabeth taylor',
              'eldest daughter',
              'wish list',
              'ruin the friendship',
              'father figure',
              'swiftie',
              'cancelled']

# %% [code]
# Read and clean data
dataframes = [read_csv(name) for name in CSV_NAMES]
cleaned_dfs = [clean_text(df) for df in dataframes]
dataset = create_dataset(cleaned_dfs)

#export cleaned dataset
dataset.to_csv('dataset_cleaned.csv', index=False)


