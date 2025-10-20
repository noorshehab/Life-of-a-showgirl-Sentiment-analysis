#from the description and location get any account with swiftie ts stan taylor etc 
#get all the other words they use and search with those too to create the keywords set
#when you have the keywords set run the scoring classifier on all the descriptions to get all the swifties

#%% [code]
import pandas as pd
import re
from collections import Counter
from itertools import chain
from nltk.util import ngrams
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import os

ACCOUNT_COLUMNS=['author/description','author/screen_name','author/name','author/location']

SWIFTIE_TERMS_INITIAL=['swift', 'swiftie', 'ts stan', 'taylor stan', 'taylor swift fan', 'taylor',
                       'taylor nation', 'taylors version', 'lover', 'reputation', 'folklore', 'evermore', 'midnights',
                       'eras','s version','ts']

STOPWORDS = set(stopwords.words('english'))

def get_swiftie_lingo(df,search_terms,base_threshold=50, max_ngram_length=3,
 decay_factor=0.8,with_stopwords=False,output_folder='Out'):
    
    df['concat_text'] = (
    df['author/description'].fillna("") + " " +
    df['author/screen_name'].fillna("") + " " +
    df['author/name'].fillna("") + " " +
    df['author/location'].fillna("")
    ).str.lower()

    # Step 2: Filter rows with any initial keyword
    pattern = "|".join(re.escape(term) for term in search_terms)
    # Use word boundaries or just substring depending on your tolerance
    mask = df['concat_text'].str.contains(pattern, na=False)
    df_filtered = df[mask]

    print(f"Filtered down to {len(df_filtered)} accounts out of {len(df)}")

    def tokenize(text):
        words = re.findall(r"\b[a-z0-9']{3,}\b", text)
        if not with_stopwords:
            words = [w for w in words if w not in STOPWORDS]
        return words
        
    counters = {n: Counter() for n in range(1, max_ngram_length+1)}
    
    for text in df_filtered['concat_text']:
        tokens = tokenize(text)
        for n in range(1, max_ngram_length+1):
            if n == 1:
                for tok in tokens:
                    if any(init in tok for init in search_terms):
                        continue
                    counters[1][tok] += 1
            else:
                for gram in ngrams(tokens, n):
                    if any(init in w for init in search_terms for w in gram):
                        continue
                    counters[n][gram] += 1
    
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    for n, counter in counters.items():
        effective_threshold = base_threshold * (decay_factor ** (n-1))
        # Convert to integer
        effective_threshold_int = int(effective_threshold)
        
        # Filter and build list
        items = [(gram, cnt) for gram, cnt in counter.items() if cnt >= effective_threshold_int]
        # Sort by descending count
        items.sort(key=lambda x: -x[1])
        
        # Build a DataFrame for saving
        if n == 1:
            phrases = [gram for gram, cnt in items]
        else:
            phrases = [" ".join(gram) for gram, cnt in items]
        counts = [cnt for gram, cnt in items]
        
        import pandas as pd
        df_out = pd.DataFrame({
            "phrase": phrases,
            "count": counts
        })
        
        # Save CSV
        filename = f"associated_keywords_{n}gram.csv"
        path = os.path.join(output_folder, filename)
        df_out.to_csv(path, index=False)
        print(f"Saved top {n}-grams to {path} (threshold ≥ {effective_threshold_int})")
        
        results[n] = df_out
    
    return results
        


#def swiftie_classifier(df):


#%% [code]
df =pd.read_csv('CSVs/dataset_cleaned.csv')
associated_keywords = get_swiftie_lingo(df,SWIFTIE_TERMS_INITIAL,50,2,0.95,with_stopwords=False)

# %% [code]
phrase_1gram = pd.read_csv('Out/pruned_associated_keywords/associated_keywords_1gram.csv')
phrase_bigram = pd.read_csv('Out/pruned_associated_keywords/associated_keywords_2gram.csv')

from_1grams=phrase_1gram['phrase'].tolist()
from_2grams=phrase_bigram['phrase'].tolist()

SWIFTIE_TERMS_EXTENDED = SWIFTIE_TERMS_INITIAL + from_1grams + from_2grams

# %%
df =pd.read_csv('CSVs/dataset_cleaned.csv')
associated_keywords_extended = get_swiftie_lingo(df,SWIFTIE_TERMS_EXTENDED,50,3,0.95,with_stopwords=False)
# %%
