import re


# ============================================================
# TEXT CLEANING
# ============================================================

DEVANAGARI_RANGE = r"ऀ-ॿ"

URL_RE = re.compile(
    r"http\S+|www\.\S+"
)

MENTION_RE = re.compile(
    r"@\w+"
)

KEEP_CHARS_RE = re.compile(
    rf"[^{DEVANAGARI_RANGE}a-zA-Z0-9\s]"
)

MULTISPACE_RE = re.compile(
    r"\s+"
)


def clean_text(text):

    text = str(text)

    text = URL_RE.sub(" ", text)

    text = MENTION_RE.sub(" ", text)

    text = KEEP_CHARS_RE.sub(" ", text)

    text = text.lower()

    text = MULTISPACE_RE.sub(
        " ",
        text
    ).strip()

    return text


# ============================================================
# NEAR-DUPLICATE KEY
# ============================================================

# The dataset contains augmented copies of reviews: the same review
# with a stock English phrase inserted ("to be honest, ...",
# "i would like to mention that ...") and/or its sentences shuffled.
# Removing exact duplicates does not catch these copies, so
# deduplication uses this key instead.

STOCK_PHRASES = [
    "i would like to mention that",
    "it s worth noting that",
    "i must say that",
    "after using this product",
    "after using this",
    "based on my purchase",
    "from my experience",
    "honestly speaking",
    "to be honest",
    "to be frank",
    "as a customer",
    "in my opinion",
    "personally",
    "basically",
    "actually",
]

STOCK_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(STOCK_PHRASES) + r")\b"
)


def has_stock_phrase(clean):
    """True if a cleaned review contains one of the stock phrases."""
    return bool(STOCK_PHRASE_RE.search(clean))


def dedup_key(clean):
    """
    Key for near-duplicate detection, applied to CLEANED text:
    stock phrases are removed and the remaining words are sorted,
    so copies that differ only by an inserted phrase or by sentence
    order get the same key.
    """
    words = STOCK_PHRASE_RE.sub(" ", clean).split()
    return " ".join(sorted(words))
