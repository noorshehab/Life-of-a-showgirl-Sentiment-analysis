To judge public reception of ts newest release i scraped 14k top tweets corresponding to relevant search terms from the first 3 days after the initial release


Scoring
    a score s∈[0,1] such that:
    Presence of a phrase / keyword contributes positively to the score.
    Phrases (bigrams/trigrams) count more (stronger signal) than single keywords.
    Additional matches increase the score, but with diminishing returns (e.g. logarithmically).
    TF–IDF weighting is used so that a rare but distinctive term gives more boost than a common one.
    The final result is normalized (e.g. via logistic / sigmoid / dividing by max possible) to lie in [0,1].

Formula: s=σ(α⋅log(1+t∈T∑​wt​))
    Where:
        T = set of matched terms/phrases in that account (from your list)
        wt​=tfidf(t,account)×length_weight(t)
        length_weight(t) 
        α is a scaling constant to control steepness
        log(1+⋅) ensures diminishing returns
        σ(x)= is the logistic sigmoid, mapping R → (0,1)