
## Sklearn TfidfVectorizer

In CodeBERT, every input is compressed into a fixed-size vector (e.g., 768) regardless of length—that's where the "averaging" (mean pooling) happens in your code. In TF-IDF, if a word isn't in your top max_features, it is simply ignored.

```python
vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english",
    ngram_range=(1, 2),  # Capture phrases like "fix bug"
)
```

Here is the breakdown of what those parameters actually do under the hood:

1. **max_features (The Global Filter)**
- Think of max_features as a popularity contest for words across your entire dataset.
- How it works: The vectorizer looks at every commit message in your training set. It counts how often every word appears. It then picks the top $N$ (e.g., 100) most frequent words and builds a "dictionary" from them.
- The Result: If a commit message is 1,000 words long but only 5 of those words are in your "top 100" dictionary, the resulting vector will only have 5 non-zero values. The other 995 words are discarded.
- Why use it? In JIT bug prediction, thousands of unique variable names or hex codes appear once and never again. Including them creates a "curse of dimensionality" where your model gets lost in noise.

2. **ngram_range=(1, 2) (Contextual Windows)**

- By default, "Bag-of-Words" models are "blind" to word order. ngram_range gives them a tiny bit of sight.

    - Unigrams (1, 1): Treats every word individually. "not" and "working" are separate.
    - Bigrams (2, 2): Treats pairs of words as a single feature. "not working" becomes its own column.
    - Combined (1, 2): Your vectorizer will create features for single words AND pairs.
        - Example: "Fix bug" results in three potential features: tfidf_fix, tfidf_bug, and tfidf_fix_bug.
    - Impact: This is crucial for commit messages because "fix" and "bug" appear in many contexts, but the bigram "fix bug" is a very strong indicator of a fault-correcting commit.

3. **stop_words="english" (Noise Reduction)**
- This tells the vectorizer to automatically ignore common English words like "the", "is", "at", and "which".
- Why it matters: These words appear in almost every commit message. Because they are so common, their IDF score (Inverse Document Frequency) would be near zero anyway, but removing them early saves memory and prevents them from taking up a slot in your max_features "popularity contest."

### Creating fixed-sized vectors

When you set max_features=100, the TfidfVectorizer first scans your entire dataset to find the top 100 most important words (e.g., "fix", "bug", "error", "null", etc.).

It then creates a fixed "Dictionary" where every word is assigned a specific "column" index:

When the vectorizer processes a specific commit message, it asks: "How much of word #0 is in here? How much of word #1?"

- Message A: "Fix bug in login" $\rightarrow$ It has a value for tfidf_fix and tfidf_bug. The other 98 columns are set to 0.0.
- Message B: "Update documentation" $\rightarrow$ It has a value for tfidf_update. The other 99 columns are set to 0.0.

Even though Message A only "used" 2 words, it still produces a vector of length 100. It's just a sparse vector (a vector filled mostly with zeros).

### Why TF-IDF might improve your Precision
In your JIT bug prediction, CodeBERT might be "over-generalizing." Because CodeBERT groups similar concepts together, it might see a commit message about a "minor refactor" and a "major bug fix" as somewhat similar because they both deal with "code changes."

TF-IDF is "brutal" and literal. If you have a high TF-IDF score for the word NullPointer, it is an extremely strong signal. By adding TF-IDF, you are giving the XGBoost model hard keywords to look for. This often helps the model create clearer decision boundaries, which should help reduce the number of False Positives (improving your Precision).

## CodeBert

CodeBERT is a "bimodal" model, meaning it was trained to understand two "modes" of communication: Natural Language (English comments, documentation) and Programming Language (Python, Java, etc.).

When you use CodeBERT to generate embeddings, you are essentially asking a pre-trained neural "brain" to summarize a piece of code based on everything it learned from reading millions of functions on GitHub.

Unlike TF-IDF, which is just a count of words, CodeBERT embeddings are generated through a multi-stage neural process.

Knowledge utilized: The model has a fixed vocabulary of 50,265 tokens. It knows that def is a keyword and calculate is a semantic concept.