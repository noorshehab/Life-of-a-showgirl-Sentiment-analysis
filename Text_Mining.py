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
from nltk.corpus import stopwords
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import constants
import numpy as np
from sklearn.model_selection import StratifiedKFold
import itertools


SWIFTIE_TERMS_INITIAL=['swift', 'swiftie', 'ts', 'taylor swift',
                       'taylor nation', 'taylors version''eras','s version']

STOPWORDS = set(stopwords.words('english'))
#For the given search terms find all the associated phrases the function can be run for a list of terms or just one
def get_lingo(df,search_terms,column_name,output_folder,base_threshold=50, max_ngram_length=3,
 decay_factor=0.8,with_stopwords=False):
    
   
    # Step 2: Filter rows with any initial keyword
    pattern = "|".join(re.escape(term) for term in search_terms)
    # Use word boundaries or just substring depending on your tolerance
    mask = df[column_name].str.contains(pattern, na=False)
    df_filtered = df[mask]

    print(f"Filtered down to {len(df_filtered)} rows out of {len(df)}")

    def tokenize(text):
        words = re.findall(r"\b[a-z0-9']{3,}\b", text)
        if not with_stopwords:
            words = [w for w in words if w not in STOPWORDS]
        return words
        
    counters = {n: Counter() for n in range(1, max_ngram_length+1)}
    
    for text in df_filtered[column_name]:
        tokens = tokenize(text)
        for n in range(1, max_ngram_length+1):
            if n == 1:
                for tok in tokens:
                    counters[1][tok] += 1
            else:
                for gram in ngrams(tokens, n):
                    counters[n][gram] += 1
    
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    filenames={}
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
        filenames[n] = filename
        path = os.path.join(output_folder, filename)
        df_out.to_csv(path, index=False)
        print(f"Saved top {n}-grams to {path} (threshold ≥ {effective_threshold_int})")
        
        results[n] = df_out
    
    return results

def get_tags_mentions(
    df,
    search_terms,
    text_col,          # the column where you match search terms (e.g. tweet text)
    hashtag_col=None,  # column containing list or string of hashtags per tweet
    mention_col=None,  # column containing list or string of mentions per tweet
    output_folder="Out",
    hashtag_thresh=10,
    mention_thresh=10
):
    """
    For tweets matching any of the search_terms, count hashtags & mentions
    that co-occur in those tweets. Save counts to CSV in output_folder.
    """
    # 1. Filter tweets by search terms
    pattern = "|".join(re.escape(term) for term in search_terms)
    mask = df[text_col].str.contains(pattern, na=False, case=False)
    df_sub = df[mask]
    print(f"Filtered {len(df_sub)} / {len(df)} tweets containing search terms")

    # Prepare counters
    hashtag_counter = Counter()
    mention_counter = Counter()

    for idx, row in df_sub.iterrows():
        # process hashtags
        if hashtag_col:
            h = row.get(hashtag_col)
            if isinstance(h, str):
                # e.g. a string like "#tag1 #tag2", split by whitespace or commas
                tags = re.findall(r"#\w+", h)
            else:
                tags = []
            for tag in tags:
                hashtag_counter[tag.lower()] += 1

        # process mentions
        if mention_col:
            m = row.get(mention_col)
            if isinstance(m, str):
                mentions = re.findall(r"@\w+", m)
            else:
                mentions = []
            for mention in mentions:
                mention_counter[mention.lower()] += 1

    os.makedirs(output_folder, exist_ok=True)

    results = {}
    # Save hashtag counts
    if hashtag_col:
        items = [(tag, cnt) for tag, cnt in hashtag_counter.items() if cnt >= hashtag_thresh]
        items.sort(key=lambda x: -x[1])
        df_ht = pd.DataFrame({"hashtag": [t for t, c in items], "count": [c for t, c in items]})
        path_ht = os.path.join(output_folder, "cooccur_hashtags.csv")
        df_ht.to_csv(path_ht, index=False)
        print(f"Saved co-occurring hashtags to {path_ht}")
        results["hashtags"] = df_ht

    if mention_col:
        items = [(m, cnt) for m, cnt in mention_counter.items() if cnt >= mention_thresh]
        items.sort(key=lambda x: -x[1])
        df_m = pd.DataFrame({"mention": [m for m, c in items], "count": [c for m, c in items]})
        path_m = os.path.join(output_folder, "cooccur_mentions.csv")
        df_m.to_csv(path_m, index=False)
        print(f"Saved co-occurring mentions to {path_m}")
        results["mentions"] = df_m

    return results


def build_corpus_and_vectorizer(df, phrases_list,column_name):
    # Build concatenated text per account
    corpus = df[column_name].tolist()
    
    # Use TfidfVectorizer. We want to capture these phrases (so use ngram_range)
    vec = TfidfVectorizer(vocabulary=phrases_list,
                          ngram_range=(1, 2),  # allow unigrams, bigrams
                          tokenizer=lambda x: re.findall(r"\b[a-z0-9']{2,}\b", x),
                          lowercase=True)
    tfidf_matrix = vec.fit_transform(corpus)   # shape: (num_accounts × num_features)
    
    return df, vec, tfidf_matrix

def compute_score(
    tfidf_vector_row,
    feature_names,
    phrases_list,
    #phrase_to_length_weight,
    alpha=1.0,
    bias=0.0
):
    """
    Given one row (sparse vector) of TF–IDF scores,
    sum over matched phrases in phrases_list, apply weights,
    then compute score via sigmoid or normalized log formula.
    """
    total = 0.0
    # iterate over each phrase in your list
    for phrase in phrases_list:
        if phrase not in feature_names:
            continue
        idx = feature_names.index(phrase)
        w = tfidf_vector_row[0, idx]  # TF–IDF score
        if w > 0:
            #length_wt = phrase_to_length_weight.get(phrase, 1.0)
            total += w #* length_wt
    
    # Option A: sigmoid with log
    s = 1/(1 + np.exp(- (bias + alpha * np.log1p(total))))
    return s

def compute_scores(df, phrases_list,pred_column_name,col_name, alpha=1.0,
                    #gamma=1.5,
                     bias=-1.5):
    """
    Wrapper: builds vectorizer, computes phrase length weights,
    then compute scores for all accounts.
    Returns df with a new column pred_column_name.
    """
    df2, vec, tfidf_mat = build_corpus_and_vectorizer(df, phrases_list,col_name)
    feature_names = vec.get_feature_names_out().tolist()
    
    # Build length weights: e.g. unigrams = 1.0, bigrams = gamma^1, trigrams = gamma^2, etc.
   # phrase_to_length_weight = {}
   # for ph in phrases_list:
    #    n = len(ph.split())
     #   phrase_to_length_weight[ph] = gamma ** (n - 1)
    
    scores = []
    for i in range(tfidf_mat.shape[0]):
        row = tfidf_mat[i : i+1]  # keep as sparse row
        s = compute_score(row, feature_names, phrases_list,
                                       #phrase_to_length_weight,
                                       alpha=alpha, bias=bias)
        scores.append(s)
    
    df2[pred_column_name] = scores
    return df2


def read_csv_with_fallback(path, encodings=None, **kwargs):
    """Try reading a CSV using a list of encodings, falling back to a permissive read.

    Returns a DataFrame. kwargs are passed to pd.read_csv.
    """
    if encodings is None:
        encodings = ['utf-8', 'cp1252', 'latin-1']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
        except Exception:
            # other errors (e.g. parser errors) should be raised
            raise

    # Last resort: open with errors='replace' to avoid decode errors
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return pd.read_csv(f, **kwargs)

def tune_parameters(threshold_range,biases,alphas,phrases_list,labeled_df,column_name,label_name,probability_column_name,pred_column_name):
    """
    Function tunes the parameters for the simple text classification from keywords
    probability threshold,bias based on how distribution of examples,weights for length of phrases,alpha scaling factor
    takes the labeled set and splits to test and tune
    returns the best parameters based on f1 score
    label_name:name of the label column from the original csv
    probability_column_name:name of the probability column to be created
    pred_column_name:name of the predicted label column to be created
    
    """
    #split into tune and test
    tuning_set,test_set=train_test_split(labeled_df,test_size=0.45,random_state=42,stratify=labeled_df[label_name])
    print(f"Tuning set size: {len(tuning_set)}, Test set size: {len(test_set)}")
    f1_scores=[]
    accuracy_scores=[]
    precision_scores=[]
    recall_scores=[]  

    for threshold in threshold_range:
        for bias in biases:

            for alpha in alphas:
                #compute probabilities
                scored_df=compute_scores(tuning_set,phrases_list,probability_column_name,column_name,alpha=alpha,bias=bias)
                #apply threshold to get predictions
                scored_df[pred_column_name]=(scored_df[probability_column_name]>=threshold).astype(int)
                #compute metrics
                f1=f1_score(tuning_set[label_name],scored_df[pred_column_name])
                accuracy=accuracy_score(tuning_set[label_name],scored_df[pred_column_name])
                precision=precision_score(tuning_set[label_name],scored_df[pred_column_name])
                recall=recall_score(tuning_set[label_name],scored_df[pred_column_name])
                #save to lists
                f1_scores.append((f1,threshold,bias,alpha))
                accuracy_scores.append((accuracy,threshold,bias,alpha))
                precision_scores.append((precision,threshold,bias,alpha))
                recall_scores.append((recall,threshold,bias,alpha))

    #get best parameters with best scores
    best_parameters_f1=f1_scores[np.argmax([x[0] for x in f1_scores])]
    print(f"Best F1 Score: {best_parameters_f1[0]} with Threshold: {best_parameters_f1[1]}, Bias: {best_parameters_f1[2]},Alpha: {best_parameters_f1[3]}")
    best_parameters_accuracy=accuracy_scores[np.argmax([x[0] for x in accuracy_scores])]
    print(f"Best Accuracy Score: {best_parameters_accuracy[0]} with Threshold: {best_parameters_accuracy[1]}, Bias: {best_parameters_accuracy[2]},  Alpha: {best_parameters_accuracy[3]}")
    best_parameters_precision=precision_scores[np.argmax([x[0] for x in precision_scores])]
    print(f"Best Precision Score: {best_parameters_precision[0]} with Threshold: {best_parameters_precision[1]}, Bias: {best_parameters_precision[2]}, Alpha: {best_parameters_precision[3]}")
    best_parameters_recall=recall_scores[np.argmax([x[0] for x in recall_scores])]
    print(f"Best Recall Score: {best_parameters_recall[0]} with Threshold: {best_parameters_recall[1]}, Bias: {best_parameters_recall[2]}, Alpha: {best_parameters_recall[3]}")
    
    #test set with best f1 parameters
    scored_test_set=compute_scores(test_set,phrases_list,probability_column_name,column_name,alpha=best_parameters_f1[3],bias=best_parameters_f1[2])
    scored_test_set[pred_column_name]=(scored_test_set[probability_column_name]>=best_parameters_f1[1]).astype(int)
    f1_test=f1_score(test_set[label_name],scored_test_set[pred_column_name])
    accuracy_test=accuracy_score(test_set[label_name],scored_test_set[pred_column_name])
    precision_test=precision_score(test_set[label_name],scored_test_set[pred_column_name])
    recall_test=recall_score(test_set[label_name],scored_test_set[pred_column_name])
    print(f"Test Set Performance with Best F1 Parameters - F1: {f1_test}, Accuracy: {accuracy_test}, Precision: {precision_test}, Recall: {recall_test}")
    # confusion matrix (rows=true, cols=predicted) - specify label order explicitly
    labels_order = [1, 0]
    cm = confusion_matrix(test_set[label_name], scored_test_set[pred_column_name], labels=labels_order)
    print("Confusion Matrix:")
    print(cm)

    # Visualize with ticks corresponding to class labels and annotated counts
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')

    # Set axis labels and ticks to class labels (not raw pixel coordinates)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    ax.set_xticks(np.arange(len(labels_order)))
    ax.set_yticks(np.arange(len(labels_order)))
    ax.set_xticklabels(labels_order)
    ax.set_yticklabels(labels_order)

    # Annotate each cell with the count
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    return best_parameters_f1


def _predict_probs_with_vectorizer(train_df, val_df, phrases_list, col_name, alpha, bias):
    """Fit TF-IDF on train, compute probabilities on val using existing pipeline.

    Returns a numpy array of probabilities aligned with val_df.
    """
    # Fit vectorizer on training fold
    _, vec, _ = build_corpus_and_vectorizer(train_df, phrases_list, col_name)
    feature_names = vec.get_feature_names_out().tolist()

    # Transform validation fold
    val_corpus = val_df[col_name].astype(str).tolist()
    tfidf_val = vec.transform(val_corpus)

    probs = []
    for i in range(tfidf_val.shape[0]):
        row = tfidf_val[i : i+1]
        p = compute_score(row, feature_names, phrases_list, alpha=alpha, bias=bias)
        probs.append(p)
    return np.asarray(probs)


def tune_parameters_cv(
    threshold_range,
    biases,
    alphas,
    phrases_list,
    labeled_df,
    column_name,
    label_name='swiftie',
    n_splits=5,
    random_state=42,
):
    """Grid search with Stratified K-Fold cross-validation.

    Returns (best_params_dict, results_df) where results_df contains mean metrics per combo.
    """
    # validate label
    if label_name not in labeled_df.columns:
        raise ValueError(f"Label column '{label_name}' not found in labeled_df")

    # adjust n_splits if necessary
    class_counts = labeled_df[label_name].value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError("Need at least 2 examples in each class for CV")
    if min_class_count < n_splits:
        n_splits = int(min_class_count)
        print(f"Warning: lowering n_splits to {n_splits} due to limited class counts")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    combos = list(itertools.product(threshold_range, biases, alphas))
    results = []

    X = labeled_df.reset_index(drop=True)
    y = X[label_name].values

    for threshold, bias, alpha in combos:
        fold_metrics = {'f1': [], 'accuracy': [], 'precision': [], 'recall': []}

        for train_idx, val_idx in skf.split(X, y):
            train_df = X.iloc[train_idx].reset_index(drop=True)
            val_df = X.iloc[val_idx].reset_index(drop=True)

            probs_val = _predict_probs_with_vectorizer(train_df, val_df, phrases_list, column_name, alpha, bias)
            preds = (probs_val >= threshold).astype(int)

            y_val = val_df[label_name].values
            # compute metrics (zero_division=0 to avoid exceptions)
            f1 = f1_score(y_val, preds, zero_division=0)
            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)

            fold_metrics['f1'].append(f1)
            fold_metrics['accuracy'].append(acc)
            fold_metrics['precision'].append(prec)
            fold_metrics['recall'].append(rec)

        results.append({
            'threshold': threshold,
            'bias': bias,
            'alpha': alpha,
            'mean_f1': float(np.mean(fold_metrics['f1'])),
            'mean_accuracy': float(np.mean(fold_metrics['accuracy'])),
            'mean_precision': float(np.mean(fold_metrics['precision'])),
            'mean_recall': float(np.mean(fold_metrics['recall'])),
        })

    results_df = pd.DataFrame(results).sort_values('mean_f1', ascending=False).reset_index(drop=True)
    best = results_df.iloc[0]
    best_params = {'threshold': best['threshold'], 'bias': best['bias'], 'alpha': best['alpha']}
    return best_params, results_df

#%% [code]
df =pd.read_csv(constants.DATASET_PATH)
keywords=pd.read_csv('Out/associated_keywords_1gram.csv')
keywords=keywords[keywords['type']=='tsr']
keywords_list=np.asarray(keywords['phrase'])

threshold_range = np.arange(0.5, 0.9, 0.1)
biases = np.arange(-2.0, 2.0, 0.5)
alphas = np.arange(0.5, 2.0, 0.5)

labeled_set = read_csv_with_fallback('labeled.csv')
# ensure dropna uses list for subset
labeled_set = labeled_set.dropna(subset=['swiftie'])

best_parameters ,results_df= tune_parameters_cv(
    threshold_range=threshold_range,
    biases=biases,
    alphas=alphas,
    phrases_list=keywords_list,
    labeled_df=labeled_set,
    column_name='acc_details',
    label_name='swiftie',  # your label column
      # where to store predictions
)

# %% [code]

best_parameters ,results_df= tune_parameters_cv(
    threshold_range=threshold_range,
    biases=biases,
    alphas=alphas,
    phrases_list=keywords_list,
    labeled_df=labeled_set,
    column_name='acc_details',
    label_name='swiftie',  # your label column
      # where to store predictions
)



# %%
print(results_df.head())
print(best_parameters)
# %%
threshold=0.5
bias=0
alpha=0.5
df_labeled=compute_scores(df,keywords_list,'swiftie_prb','acc_details',alpha,bias)
df_labeled['swiftie']=(df_labeled['swiftie_prb']>=threshold).astype(int)
swifties=df_labeled[df_labeled['swiftie']==1]
swiftie_sample=swifties.sample(n=200, random_state=3)
swiftie_sample.to_csv('swifties.csv',index=False)


# %%

print(len(swifties))
print(keywords_list)