# %% [code]

#Imports

import tweetnlp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline


# Engagement EDA
def engagement_eda(df):
    num_cols = ['reply_count', 'retweet_count', 'quote_count', 'favorite_count']
    for col in num_cols:
        plt.figure(figsize=(6,4))
        cap = df[col].quantile(0.99)
        sns.histplot(df[df[col] <= cap][col], bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlim(0, cap)
        plt.show()



    df['tweet_length'] = df['full_text'].astype(str).apply(len)
    plt.figure(figsize=(6,4))
    cap = df['tweet_length'].quantile(0.99)
    sns.histplot(df[df['tweet_length'] <= cap]['tweet_length'], bins=30, kde=True)
    plt.title('Distribution of Tweet Lengths')
    plt.xlim(0, cap)
    plt.show()


    plt.figure(figsize=(6,4))
    sns.heatmap(df[num_cols + ['tweet_length']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Between Numeric Features')
    plt.show()

# Textual EDA

def textual_analysis(df):
    texts = df['masked'].dropna().astype(str).tolist()
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2), max_features=50)
    X = vectorizer.fit_transform(texts)
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1))


    plt.figure(figsize=(10,5))
    sns.barplot(x=list(word_freq.values()), y=list(word_freq.keys()))
    plt.title('Most Common Words & Bigrams')
    plt.xlabel('Frequency')
    plt.show()


    text_all = ' '.join(texts)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text_all)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Tweets')
    plt.show() 

# TweetNLP Analysis


def get_topic(df):
    model = tweetnlp.load_model('topic_classification', multi_label=False)
    def get_topic(text):
        try:
            result = model.topic(text)
            return result['label']
        except:
            return None
    df['topic'] = df['cleaned'].astype(str).apply(get_topic)
    # Visualize topic distribution
    plt.figure(figsize=(10,5))
    sns.countplot(y='topic', data=df, order=df['topic'].value_counts().index)
    plt.title('Topic Distribution')
    plt.xlabel('Count')
    plt.ylabel('Topic')
    plt.show()
    return df

def emotion_analysis(df):
    model=tweetnlp.load_model('emotion')
    def get_emotion(text):
        try:
            result = model.emotion(text)
            return result['label']
        except:
            return None
    df['emotion_tweetnlp'] = df['cleaned'].astype(str).apply(get_emotion)

    plt.figure(figsize=(6,4))
    sns.countplot(x='emotion_tweetnlp', data=df, order=df['emotion_tweetnlp'].value_counts().index)
    plt.title('Emotion Distribution (TweetNLP)')
    plt.show()

def sentiment_analysis(df):
    model =tweetnlp.load_model('sentiment')
    def get_sentiment(text):
        try:
            result = model.sentiment(text)
            return result['label']
        except:
            return None
    df['sentiment_tweetnlp'] = df['cleaned'].astype(str).apply(get_sentiment)

    plt.figure(figsize=(6,4))
    sns.countplot(x='sentiment_tweetnlp', data=df, order=df['sentiment_tweetnlp'].value_counts().index)
    plt.title('Sentiment Distribution (TweetNLP)')
    plt.show()

def NamedEntity_analysis(df):
    model = tweetnlp.load_model('ner')

    def extract_entities(text):
        try:
            result = model.ner(text)
            entities = [r['entity'] for r in result]
            types = [r['type'] for r in result]
            return pd.Series([entities, types])
        except Exception as e:
            return pd.Series([[], []])

    df[['entities', 'entity_types']] = df['cleaned'].astype(str).apply(extract_entities)

    # Flatten
    exploded = df.explode('entity_types').explode('entities')

    # --- Visualization: Top 20 Named Entities ---
    plt.figure(figsize=(10, 5))
    top_entities = (
        exploded['entities']
        .value_counts()
        .head(20)
        .sort_values(ascending=True)
    )
    sns.barplot(x=top_entities.values, y=top_entities.index)
    plt.title('Top 20 Named Entities (TweetNLP)')
    plt.xlabel('Count')
    plt.ylabel('Entity')
    plt.show()

    # --- Visualization: Top 20 Entity Types ---
    plt.figure(figsize=(10, 5))
    top_types = (
        exploded['entity_types']
        .value_counts()
        .head(20)
        .sort_values(ascending=True)
    )
    sns.barplot(x=top_types.values, y=top_types.index)
    plt.title('Top 20 Entity Types (TweetNLP)')
    plt.xlabel('Count')
    plt.ylabel('Entity Type')
    plt.show()

    return df



def Irony_analysis(df):
    model=tweetnlp.load_model('irony')
    def is_ironic(text):
        try:
            result = model.irony(text)
            return result['label']
        except:
            return None
    df['irony_tweetnlp'] = df['cleaned'].astype(str).apply(is_ironic)
    plt.figure(figsize=(6,4))
    sns.countplot(x='irony_tweetnlp', data=df, order=df['irony_tweetnlp'].value_counts().index)
    plt.title('Irony Distribution (TweetNLP)')
    plt.show()

def Relevance_check(df):
    model=pipeline('question-answering', model='distilbert-base-cased-distilled-squad')
    questions=["is this tweet about taylor swift yes or no?",
               "is this tweet about taylor swift's new album the life of a showgirl yes or no?",
               "is this tweet about taylor swift's music yes or no?"]
    def is_about_ts(text):
        answers = []
        for question in questions:
            result = model(question=question, context=text)
            answers.append(result['answer'].lower())
        return answers
    df['relevance'] = df['cleaned'].astype(str).apply(is_about_ts)
    #plt.figure(figsize=(6,4))
    #sns.countplot(x='relevance', data=df, order=df['relevance'].value_counts().index)
    #plt.title('Relevance Distribution (TweetNLP)')
    #plt.show()

    

# =====================
# Full EDA Pipeline
# =====================
def full_eda_pipeline(df):
    engagement_eda(df)
    textual_analysis(df)
    sentiment_analysis(df)
    get_topic(df)
    emotion_analysis(df)
    NamedEntity_analysis(df)
    Irony_analysis(df)
    Relevance_check(df)
    #examples of relevant and irrelevant tweets
    #df_relevant=df[df['relevance']=='Relevant']
    #df_irrelevant=df[df['relevance']=='Irrelevant']
    #df_relevant.to_csv('relevant_tweets.csv', index=False)
    df.to_csv('tweets.csv', index=False)
    #extract_themes(df)
    #correlation_analysis(df)
    #df.to_csv('tweets_with_sentiment_themes_labeled.csv', index=False)
    #print('Analysis complete — exported tweets_with_sentiment_themes_labeled.csv')

# %% [code]
DATASET_PATH='CSVs\dataset_cleaned.csv'
dataset=pd.read_csv(DATASET_PATH)
dataset=dataset.sample(frac=0.01, random_state=42).reset_index(drop=True)  # shuffle
full_eda_pipeline(dataset)

# %% [code]

# %%
