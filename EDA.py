#%%[code]
#Imports

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import seaborn as sns
import json
import os
import constants 

def summary_metrics(df):
    metrics = {}
    metrics['how_many_tweets'] = len(df)
    metrics['how_many_accounts'] = df['author/id_str'].nunique()
    metrics['average_tweets_by_author'] = df.groupby('author/id_str').size().mean()
    metrics['how_many_quotes'] = len(df[df['is_quote_status'] == True])
    metrics['how_many_replies'] = len(df[df['is_reply'] == True])
    metrics['how_many_conversations'] = df['conversation_id_str'].nunique()
    metrics['length_of_conversations'] = df['conversation_length'].mean()
    metrics['average_retweets'] = df['retweet_count'].mean()
    metrics['average_replies'] = df['reply_count'].mean()
    metrics['average_quotes'] = df['quote_count'].mean()
    metrics['average_likes'] = df['favorite_count'].mean()

    all_ids = set(df['id'].astype(str))
    # internal replies: those where conversation_id_str matches some tweet id
    metrics['internal_replies'] = df["conversation_id_str"].astype(str).isin(all_ids).sum()
    # external replies = replies minus internal ones
    metrics['external_replies'] = len(df[df['is_reply'] == True]) - metrics['internal_replies']
    # internal quotes
    metrics['internal_quotes'] = df["quoted_status_id_str"].astype(str).isin(all_ids).sum()
    metrics['external_quotes'] = len(df[df['is_quote_status'] == True]) - metrics['internal_quotes']
    # tweets in same conversation (i.e. conversation_length > 1)
    metrics['how_many_tweets_in_same_conversation'] = len(df[df['conversation_length'] > 1])
    # total volume of retweets
    metrics['total_volume_of_retweets'] = df['retweet_count'].sum()

    with open("summary_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=int)

    return metrics

def plot_percentile_cdf(data, p_low=25, p_high=75, label=None):
    """
    Split `data` by percentiles and plot cumulative distributions:
      - Lower tail (<= p_low percentile)
      - Middle between (p_low, p_high)
      - Upper tail (>= p_high percentile)
    label: label for x-axis / plot
    """
    arr = np.asarray(data)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        print("No data to plot.")
        return

    low_val = np.percentile(arr, p_low)
    high_val = np.percentile(arr, p_high)

    lower = arr[arr <= low_val]
    middle = arr[(arr > low_val) & (arr < high_val)]
    upper = arr[arr >= high_val]

    n = len(arr)
    label_safe = label.replace("/", "_") 

    # === Plot 1: lower + middle together ===
    plt.figure(figsize=(8, 6))
    # Lower
    if len(lower) > 0:
        lower_sorted = np.sort(lower)
        y_low = np.arange(1, len(lower_sorted) + 1) / n
        plt.plot(lower_sorted, y_low, marker='.', linestyle='none', label=f"Lower ≤ {p_low}th pct")

    # Middle
    if len(middle) > 0:
        mid_sorted = np.sort(middle)
        y_mid = np.arange(1, len(mid_sorted) + 1) / n
        plt.plot(mid_sorted, y_mid, marker='.', linestyle='none', label=f"Middle {p_low}–{p_high} pct")

    plt.xlabel(label)
    plt.ylabel("Fraction ≤ x")
    plt.title(f"CDF: Lower & Middle Percentiles ({p_low}-{p_high})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname1 =f"{constants.PLOTS_PATH}/cdf_lower_middle_{label_safe}.png"
    os.makedirs(os.path.dirname(fname1), exist_ok=True)
    plt.savefig(fname1, bbox_inches='tight')
    plt.close()

    # === Plot 2: upper percentile only ===
    if len(upper) > 0:
        plt.figure(figsize=(6, 5))
        up_sorted = np.sort(upper)
        y_up = np.arange(1, len(up_sorted) + 1) / n
        plt.plot(up_sorted, y_up, marker='.', linestyle='none', label=f"Upper ≥ {p_high}th pct")
        plt.xlabel(label)
        plt.ylabel("Fraction ≤ x")
        plt.title(f"CDF: Upper Percentile ≥ {p_high}th")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        fname2 =f"{constants.PLOTS_PATH}/cdf_upper_{label_safe}.png"
        os.makedirs(os.path.dirname(fname2), exist_ok=True)
        plt.savefig(fname2, bbox_inches='tight')
        plt.close()


def Distributions(df, save_folder='plots'):
    os.makedirs(save_folder, exist_ok=True)
    fields=["retweet_count",
    "reply_count",
    "quote_count",
    "favorite_count",
    "bookmark_count","author/followers_count",
    "author/friends_count",
    "author/statuses_count",]

    for field in fields:
        plot_percentile_cdf(df[field],label=f"{field} distribution")


def correlations(df, threshold=0.3, save_folder='plots'):
    os.makedirs(save_folder, exist_ok=True)

    cols = ['retweet_count', 'reply_count', 'quote_count', 'favorite_count',
            'author/followers_count', 'author/friends_count',
            'author/statuses_count','conversation_length']
    pearson_corr = df[cols].corr(method='pearson')
    spearman_corr = df[cols].corr(method='spearman')

    # 1. Heatmaps
    for corr_mat, name in [(pearson_corr, "Pearson"), (spearman_corr, "Spearman")]:
        plt.figure(figsize=(8,6))
        sns.heatmap(
            corr_mat, annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, center=0,
            square=True, linewidths=0.5
        )
        plt.title(f"Correlation Heatmap ({name})")
        fname = f"{constants.PLOTS_PATH}/heatmap_{name}.png"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()
    def extract_pairs(corr_mat, thr_low, thr_high):
        pairs = set()
        for i in corr_mat.index:
            for j in corr_mat.columns:
                if i == j:
                    continue
                val = corr_mat.loc[i, j]
                if pd.isna(val):
                    continue
                av = abs(val)
                if av >= thr_low and av < thr_high:
                    pairs.add((i, j, val))
        return pairs

    p_pairs = extract_pairs(pearson_corr, threshold, 1.0)
    s_pairs = extract_pairs(spearman_corr, threshold, 1.0)

    def plot_scatter(x, y, corr_val):
        plt.figure(figsize=(5, 4))
        sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
        sns.regplot(data=df, x=x, y=y, scatter=False, color='red', ci=None)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{x} vs {y}, r≈{corr_val:.2f}")
        ax = plt.gca()
        ax.ticklabel_format(style='plain', axis='both', useOffset=False)
        x_safe=x.replace("/", "_")
        y_safe=y.replace("/", "_")
        fname = f"{constants.PLOTS_PATH}/scatter_{x_safe}_vs_{y_safe}.png"
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

    for (x, y, val) in s_pairs:
        plot_scatter(x, y, val)

   
def prolific_accounts(df, percentile=90):
    total=len(df)
    # 1. Tweets by accounts whose statuses_count is in the top percentile
    # first find the cutoff on statuses_count
    cutoff_statuses = df['author/statuses_count'].dropna().quantile(percentile / 100.0)
    # select which authors qualify
    prolific_authors = df.loc[df['author/statuses_count'] > cutoff_statuses, 'author/id_str']
    # count how many tweets in dataset by those authors
    by_big_posters = df[df['author/id_str'].isin(prolific_authors)].shape[0]

    # 2. Repeat posters (by dataset tweet counts)
    counts = df.groupby("author/id_str")["id"].count().reset_index(name="tweet_count")
    cutoff_tweetcount = counts["tweet_count"].quantile(percentile / 100.0)
    repeat_authors = counts[counts["tweet_count"] > cutoff_tweetcount]["author/id_str"]
    by_repeat_posters = repeat_authors.shape[0]

    # 3. Tweets by popular accounts (top percentile by followers)
    cutoff_followers = df['author/followers_count'].dropna().quantile(percentile / 100.0)
    popular_authors = df.loc[df['author/followers_count'] > cutoff_followers, 'author/id_str']
    by_popular_posts = df[df['author/id_str'].isin(popular_authors)].shape[0]

    print(f"tweets by prolific posters (top statuses_count): {by_big_posters} proportion {(by_big_posters/total)*100} %")
    print(f"number of repeat posters (top dataset tweet_count): {by_repeat_posters} proportion {(by_repeat_posters/total)*100} %")
    print(f"tweets by popular accounts (top follower_count): {by_popular_posts} proportion {(by_popular_posts/total)*100} %")

    return {
        "by_big_posters_tweets": by_big_posters,
        "num_repeat_posters": by_repeat_posters,
        "by_popular_posts": by_popular_posts,
        "cutoff_statuses": cutoff_statuses,
        "cutoff_tweetcount": cutoff_tweetcount,
        "cutoff_followers": cutoff_followers,
    }

    return
# %%
df=pd.read_csv(constants.DATASET_PATH)
prolific_accounts(df,80)


# %%
