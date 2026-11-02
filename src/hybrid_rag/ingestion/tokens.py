from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+|[^\w\s]")


class RegexTokenCounter:
    """Words + punctuation. Dependency-free, deterministic, and a slight
    underestimate of BPE counts — budgets should leave ~20% headroom."""

    def count(self, text: str) -> int:
        return len(_TOKEN.findall(text))
