#%%
import constants
import pandas as pd
import re
import unicodedata
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

swiftie_terms = [
    # general
    'taylor swift',
    'taylor',
    'tn',
    'swiftie',
    'tay',
    # album names
    'life of a showgirl',
    'evermore',
    'folklore',
    'reputation',
    'midnights',
    'tortured poets department',
    'red',
    '1989',
    # references
    'eras',
    'the fate of',
    'actually romantic',
    'eldest daughter',
    '❤️‍🔥',
    "s version",
    '⸆⸉',
    'opalite',
    'ophelia'
]

author_columns = [
    "author/name",
    "author/screen_name",
    "author/location",
    "author/description",
]

relevant_terms=[
    #general
    'taylor swift',
    'swiftie',
    #album
    'life of a showgirl',
    'showgirl',
    'tloas',
    #songs
    'actually romantic',
    'fate of ophelia',
    'opalite',
    'elizabeth taylor',
    'eldest daughter',
    'ruin the friendship',
    'father figure',
    'cancelled!'
]

text_col='full_text'

def label(row,terms,columns):
    for col in columns:
        raw = row.get(col, "")
        if raw is None:
            continue
        text = str(raw)
        # normalize
        text_norm = unicodedata.normalize("NFC", text).lower()
        for term in terms:
            term_norm = unicodedata.normalize("NFC", term).lower()
            # decide matching logic
            # if term contains non-alphanumeric (emoji etc.)
            if re.search(r"\W", term_norm):  # \W = non-word character
                # substring match
                if term_norm in text_norm:
                    return True
            else:
                # match whole word (so "red" doesn't match "reddish")
                if re.search(rf"\b{re.escape(term_norm)}\b", text_norm):
                    return True
    return False

def make_author_text(row, author_cols):
    """Concatenate author columns into one text field (normalized)"""
    parts = []
    for col in author_cols:
        v = row.get(col, "")
        if pd.isna(v):
            continue
        parts.append(str(v))
    txt = " ".join(parts)
    # Normalize Unicode (so emoji sequences are consistent)
    txt = unicodedata.normalize("NFC", txt)
    return txt

def train_and_eval_nb(sample_df, target_column,label_column, n_splits=5):
    # Prepare X, y
    sample_df = sample_df.copy()
    #sample_df["author_text"] = sample_df.apply(lambda r: make_author_text(r, author_columns), axis=1)
    X = sample_df[target_column].values
    y = sample_df[label_column].astype(int).values  # 1 = swiftie, 0 = not

    # Define vectorizer that preserves emojis (just treat them as tokens)
    # For example, use a custom token pattern that includes non-whitespace, including emojis
    vect = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b|[\U0001F000-\U0001FAFF]+"
        # Explanation: match normal word tokens OR a block for emoji codepoints
        # This pattern might need tuning depending on your emoji range.
    )

    # Build pipeline: vectorization → NB classifier
    clf = Pipeline([
        ("vectorizer", vect),
        ("nb", MultinomialNB(alpha=1.0))  # Laplace smoothing
    ])

    # Stratified k-fold so each fold keeps class balance
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("Cross-validation results:")
    # Use accuracy or other scoring
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print("  Accuracy per fold: ", scores)
    print("  Mean accuracy: %.3f ± %.3f" % (scores.mean(), scores.std()))

    # Also run full training + test split for inspection
    # (or you can pick one fold to inspect more deeply)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("\nClassification report on full data (for inspection):")
    print(classification_report(y, y_pred, digits=3))

    # Optionally inspect feature log probabilities to see which tokens (including emojis) are predictive
    vectorizer = clf.named_steps["vectorizer"]
    nb_model = clf.named_steps["nb"]
    feature_names = vectorizer.get_feature_names_out()
    # Log probabilities for class=1 (swiftie)
    log_probs = nb_model.feature_log_prob_[1]  # shape = (n_features,)
    # Get top features for Swiftie
    top_idx = np.argsort(log_probs)[-20:]  # 20 highest
    print("Top features for Positive class:")
    for idx in reversed(top_idx):
        print(f"  {feature_names[idx]}  (log prob = {log_probs[idx]:.3f})")

    cm = confusion_matrix(y, y_pred)
    print("Confusion matrix (counts):")
    print(cm)

    # Optionally, display with labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)  # you can choose other colormap
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

    return clf  # trained model
#%%
#read the data
df = pd.read_csv(constants.DATASET_PATH,low_memory=False,dtype={col: str for col in constants.id_cols})
# Sample 1500 rows
sample_df = df.sample(n=1500, random_state=7)
ids_to_remove = sample_df["id"].tolist()
unlabeled= df[~df["id"].isin(ids_to_remove)].copy()
# Keep only one row per account
sample_df = sample_df.drop_duplicates(subset="author/id_str", keep="first")


#%%
# Run the label function 
sample_df["is_swiftie"] = sample_df.apply(lambda row: label(row,swiftie_terms,author_columns), axis=1)

print("Total labeled rows:", len(sample_df))
print("Swifties:", sample_df["is_swiftie"].sum())
print("Non-swifties:", len(sample_df) - sample_df["is_swiftie"].sum())

sample_df['is_relevant']=sample_df.apply(lambda row: label(row,relevant_terms,[text_col]), axis=1)

print("Total labeled rows:", len(sample_df))
print("Relevant:", sample_df["is_relevant"].sum())
print("Not Relevant:", len(sample_df) - sample_df["is_relevant"].sum())

#%%
# Show a sample of 50 tweets that are relevant
print("\n--- Sample Relevant ---")
print(sample_df[sample_df["is_relevant"] == True].head(50)[[text_col,"is_relevant"]])

# Show a sample of 50 tweets that are not relevant
print("\n--- Sample Not Relevant ---")
print(sample_df[sample_df["is_relevant"] == False].head(50)[[text_col,"is_relevant"]])


# %%
#train the classifiers 
sample_df["author_text"] = sample_df.apply(lambda r: make_author_text(r, author_columns), axis=1)
print("--------Swiftie Classifier----------")
swiftie_clf = train_and_eval_nb(sample_df,'author_text','is_swiftie', n_splits=5)
print("--------Relevance Classifier----------")
relevance_clf=train_and_eval_nb(sample_df,text_col,'is_relevant', n_splits=5)

# %%
#label swiftie or not
unlabeled['author_text']=unlabeled.apply(lambda r: make_author_text(r, author_columns), axis=1)
X_unlabeled=unlabeled['author_text'].values
Y_pred=swiftie_clf.predict(X_unlabeled)
unlabeled["swiftie"]=Y_pred

#%%
#label is relevant or not
X_unlabeled=unlabeled[text_col].values
Y_pred=relevance_clf.predict(X_unlabeled)
unlabeled['relevant']=Y_pred

# %%
df2 = unlabeled 
ct = pd.crosstab(df2["swiftie"], df2["relevant"],
                 rownames=["Swiftie"], colnames=["Relevant"], margins=True, margins_name="Total")
print("Cross-tabulation (counts):")
print(ct)

ct_norm = pd.crosstab(df2["swiftie"], df2["relevant"],
                      normalize="index",  # proportions within each Swiftie status
                      rownames=["Swiftie"], colnames=["Relevant"], margins=False)
print("\nProportions within Swiftie rows:")
print(ct_norm)

# 3. Summary statistics: counts, percentages, etc.
total = len(df2)
n_swiftie = df2["swiftie"].sum()
n_relevant = df2["relevant"].sum()
print("\nTotal examples:", total)
print("Number of Swifties:", n_swiftie, f"({n_swiftie/total:.2%})")
print("Number of Relevant:", n_relevant, f"({n_relevant/total:.2%})")

# 4. Detailed breakdown from the cross tab
# For example: number of Swifties who are relevant, etc.
n_swiftie_and_relevant = ct.loc[True, True]
n_swiftie_not_relevant = ct.loc[True, False]
n_not_swiftie_but_relevant = ct.loc[False, True]
n_not_swiftie_not_relevant = ct.loc[False, False]

print("\nBreakdown:")
print(" Swiftie & Relevant:", n_swiftie_and_relevant)
print(" Swiftie & Not relevant:", n_swiftie_not_relevant)
print(" Not Swiftie & Relevant:", n_not_swiftie_but_relevant)
print(" Not Swiftie & Not relevant:", n_not_swiftie_not_relevant)

# 5. Optionally visualize with a heatmap or bar chart
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
sns.heatmap(ct.iloc[:-1, :-1], annot=True, fmt="d", cmap="Blues")
plt.title("Swiftie × Relevant counts")
plt.show()

# %%
#export
unlabeled.to_csv('labeled_swiftie_relevant.csv',encoding='UTF-8')
# %%
