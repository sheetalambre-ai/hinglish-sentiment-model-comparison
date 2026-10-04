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
