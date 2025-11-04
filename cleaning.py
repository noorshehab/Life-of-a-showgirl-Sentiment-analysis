# %% [code]
#Imports and Cleaning functions
import re
import html
import pandas as pd
from langdetect import detect, LangDetectException
from datetime import timezone
import constants


# Cleaning functions (customized)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # HTML unescape
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+|https\S+', '', text)
    #remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags 
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation (except ! & ?)
    text = re.sub(r"[^\w\s\!\?]", " ", text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_mentions(text):
    lst = re.findall(r'@\w+', text)
    return " ".join(lst)

def extract_hashtags(text):
    lst = re.findall(r'#\w+', text)
    return " ".join(lst)

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
    for term in constants.SEARCH_TERMS:
        if term in text:
            return term 

def combine_fields(row):
        return (
            f"{row.get('author/description', '')} "
            f"{row.get('author/screen_name', '')} "
            f"{row.get('author/name', '')} "
            f"{row.get('author/location', '')}"
        ).strip().lower()

def account_age(timestamp):
    # parse with UTC awareness
    created_at = pd.to_datetime(timestamp, errors="coerce", utc=True)
    if pd.isna(created_at):
        return None
    now = pd.Timestamp.now(tz=timezone.utc)
    age = (now - created_at).days / 365
    return round(age,3)

def account_details(df):
    df["mentions"] = df.apply(lambda row: extract_mentions(combine_fields(row)), axis=1)
    df["hashtags"] = df.apply(lambda row: extract_hashtags(combine_fields(row)), axis=1)
    df["acc_age"] = df["author/created_at"].apply(account_age)
    df['acc_details'] = df.apply(lambda row: clean_text(combine_fields(row)), axis=1)
    return df

def clean_tweets(df):
    df['tweet_mentions'] = df[constants.text_col].apply(extract_mentions)
    df['tweet_hashtags'] = df[constants.text_col].apply(extract_hashtags)
    df['cleaned_tweet'] = df[constants.text_col].apply(clean_text)
    df['is_reply'] = df['in_reply_to_status_id_str'].notnull()
    df['conversation_length'] = df.groupby('conversation_id_str')['id'].transform('count')
    df['is_minimal'] = df['cleaned_tweet'].apply(is_minimal)
    df['is_english'] = df['cleaned_tweet'].apply(is_english)
    return df

def create_dataset(dfs):
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return combined


def clean_dataframe(df):
    df['searchTerms'] = df['searchTerms'].astype(str).apply(clean_search_term)
    df= clean_tweets(df)
    df= account_details(df)
    return df




# %% [code]
# Read and clean data
id_cols=[
    "quoted_status_id_str",
    "in_reply_to_status_id_str",
    "in_reply_to_user_id_str",
    "author/id_str","id",
    "user_id_str",
    "conversation_id_str",
]
dataframes = [pd.read_csv(name,low_memory=False, usecols=constants.COLUMNS,dtype={col: str for col in id_cols}) for name in constants.CSV_NAMES]
cleaned_dfs = [clean_dataframe(df) for df in dataframes]
dataset = create_dataset(cleaned_dfs)


dataset.to_csv('CSVs/clean.csv', index=False, encoding='UTF-8')


# %%
