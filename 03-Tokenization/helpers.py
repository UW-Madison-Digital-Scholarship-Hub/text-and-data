"""Functions from the What's in a Word notebook, collected in one place.

The notebook's catch-up cell imports these so the "Your Turn" activities still
work after a session restarts. They are copies of the functions defined in the
notebook itself, so if you change one there, change it here too.
"""
import re
from collections import Counter

import gutenbergpy.textget
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

for package in ["punkt_tab", "wordnet", "averaged_perceptron_tagger_eng", "stopwords"]:
    nltk.download(package, quiet=True)

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm", disable=["parser"])

# blue, orange, aqua: easy to tell apart, including for colorblind readers
BOOK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_book(book_id):
    """Download a Project Gutenberg book (or load the saved copy) and return its text without the header and license."""
    raw = gutenbergpy.textget.get_text_by_id(book_id)
    return gutenbergpy.textget.strip_headers(raw).decode("utf-8")


def penn_to_wordnet(tag):
    """Convert a Penn Treebank tag (e.g., 'VBD') to the WordNet code the lemmatizer expects."""
    if tag.startswith('J'):
        return 'a' # adjective
    elif tag.startswith('V'):
        return 'v' # verb
    elif tag.startswith('R'):
        return 'r' # adverb
    else:
        return 'n' # noun (WordNet's default)


def word_counts(text):
    """Return a DataFrame of unique words and their counts, most frequent first."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    text = text.lower().replace("’", "'")
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    counts = Counter(words)
    return pd.DataFrame(counts.most_common(), columns=["word", "count"])


def remove_stop_words(text, stop_words=None):
    """Return the text as a lowercase string of words with stop words removed."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    text = text.lower().replace("’", "'")
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    kept = [word for word in words if word not in stop_words]
    return " ".join(kept)


def tag_text(text):
    """Run spaCy over the text one paragraph at a time and return a list of processed Docs."""
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return list(nlp.pipe(paragraphs, batch_size=50))


def pos_table(docs):
    """Return a DataFrame with one row per word: the word, its part of speech, its detailed tag, and its lemma."""
    rows = []
    for doc in docs:
        for token in doc:
            if token.is_alpha:
                rows.append({
                    "word": token.text,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "lemma": token.lemma_,
                })
    return pd.DataFrame(rows)


def pos_counts(docs, pos):
    """Return a DataFrame of words with the given part(s) of speech and their counts, most frequent first."""
    if isinstance(pos, str):
        pos = [pos]
    words = [token.text.lower() for doc in docs for token in doc
             if token.pos_ in pos and token.is_alpha]
    counts = Counter(words)
    return pd.DataFrame(counts.most_common(), columns=["word", "count"])


def entity_counts(docs, labels=("GPE", "LOC", "FAC")):
    """Return a DataFrame of named entities with the given labels and their counts, most frequent first."""
    entities = [ent.text for doc in docs for ent in doc.ents if ent.label_ in labels]
    counts = Counter(entities)
    return pd.DataFrame(counts.most_common(), columns=["entity", "count"])


def lemmatize(docs):
    """Return the text as a lowercase string of lemmas, ready for remove_stop_words or word_counts."""
    lemmas = [token.lemma_.lower() for doc in docs for token in doc if token.is_alpha]
    return " ".join(lemmas)


def analyze_book(book_id):
    """Download a Project Gutenberg book and return a dictionary of word frequency tables."""
    text = load_book(book_id)
    docs = tag_text(text)
    return {
        "words": word_counts(text),
        "words_no_stops": word_counts(remove_stop_words(text)),
        "lemmas_no_stops": word_counts(remove_stop_words(lemmatize(docs))),
        "pronouns": pos_counts(docs, "PRON"),
        "verbs": pos_counts(docs, "VERB"),
        "nouns": pos_counts(docs, "NOUN"),
        "places": entity_counts(docs),
    }


def per_1000_words(counts, words, n=20):
    """Return the top n rows of a counts table with each count converted to a rate per 1,000 words in the book."""
    total_words = words["count"].sum()
    table = counts.head(n).copy()
    table["per_1000_words"] = (table["count"] / total_words * 1000).round(1)
    return table


def plot_pronoun_rates(books, pronouns=("i", "we", "you", "he", "she", "they", "his", "her")):
    """Draw a bar chart comparing how often up to three books use the given pronouns (per 1,000 words)."""
    rates = {}
    for name, (pronoun_counts, words) in books.items():
        table = per_1000_words(pronoun_counts, words, n=len(pronoun_counts))
        rates[name] = table.set_index("word")["per_1000_words"].reindex(list(pronouns)).fillna(0)
    rates = pd.DataFrame(rates)

    ax = rates.plot.barh(figsize=(8, 6), width=0.8, color=BOOK_COLORS[:len(books)], edgecolor="white", linewidth=1)
    ax.invert_yaxis()
    ax.set_title("How often does each book use these pronouns?", loc="left")
    ax.set_xlabel("Times per 1,000 words")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    plt.show()
