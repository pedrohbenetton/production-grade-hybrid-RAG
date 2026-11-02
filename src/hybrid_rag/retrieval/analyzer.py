"""Lexical analyzer shared by BM25 indexing and querying (must stay symmetric)."""

from __future__ import annotations

import re
import unicodedata

import snowballstemmer

ENGLISH_STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been
    before being below between both but by can did do does doing down during each few
    for from further had has have having he her here hers herself him himself his how i
    if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such
    than that the their theirs them themselves then there these they this those through
    to too under until up very was we were what when where which while who whom why
    will with you your yours yourself yourselves""".split()  # noqa: SIM905
)

_WORD = re.compile(r"\w+")


class Analyzer:
    """NFKC → casefold → word tokens → stopword removal → Snowball stemming."""

    def __init__(
        self, language: str = "english", stopwords: frozenset[str] = ENGLISH_STOPWORDS
    ) -> None:
        self._stemmer = snowballstemmer.stemmer(language)
        self._stopwords = stopwords

    def __call__(self, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        words = [w for w in _WORD.findall(normalized) if w not in self._stopwords]
        return [str(stem) for stem in self._stemmer.stemWords(words)]
