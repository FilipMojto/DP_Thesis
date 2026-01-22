from sklearn.feature_extraction.text import TfidfVectorizer
import re


def advanced_clean_msg(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove SVN metadata (extremely common in Pandas/NumPy history)
    text = re.sub(r"git-svn-id:.*", "", text)

    # 3. Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # 4. Remove Hex Hashes (4+ chars) and PR numbers (e.g., #1234)
    text = re.sub(r"\b[0-9a-f]{4,}\b", "", text)
    text = re.sub(r"#\d+", "", text)

    # 5. Remove file extensions (keep the name, lose the .py/.cy)
    text = re.sub(r"\.py|\.c|\.cpp|\.h", " ", text)

    # 6. Remove non-alphabetic noise
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    return text.strip()

DEF_MAX_FEATURES = 100

sklearn_tfidf_vectorizer = TfidfVectorizer(
    max_features=DEF_MAX_FEATURES,
    stop_words="english",
    ngram_range=(1, 2),  # Capture phrases like "fix bug"
    preprocessor=advanced_clean_msg,
)

class DenseTfidf(TfidfVectorizer):
    def transform(self, X):
        return super().transform(X).toarray()
    

