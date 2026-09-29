"""Kinyarwanda text normalization: turn digits, money, percentages, dates and
abbreviations into the words a native speaker would actually say.

Every TTS model is only as good as its input text. MMS, VITS, etc. were trained
on spelled-out words, so "5,000 Frw" must become "amafaranga ibihumbi bitanu".

NOTE: number agreement in Kinyarwanda depends on the noun class being counted.
The rules below cover the common cases and MUST be reviewed by a native speaker
(see tests/test_normalize.py -- extend it whenever you find a wrong reading).
"""

from __future__ import annotations

import re

# Unit numerals 1..9 per agreement pattern.
#   count : bare counting ("rimwe, kabiri, gatatu ...")
#   a     : class 6 (ama-/ma-)      e.g. amafaranga atanu, magana abiri
#   i     : class 4 (imi-/mi-)      e.g. mirongo itatu
#   bi    : class 8 (ibi-)          e.g. ibihumbi bibiri
#   zi    : class 10 (in-/im-)      e.g. miliyoni ebyiri
UNITS: dict[str, list[str]] = {
    "count": ["rimwe", "kabiri", "gatatu", "kane", "gatanu", "gatandatu", "karindwi", "umunani", "icyenda"],
    "a": ["rimwe", "abiri", "atatu", "ane", "atanu", "atandatu", "arindwi", "umunani", "icyenda"],
    "i": ["umwe", "ibiri", "itatu", "ine", "itanu", "itandatu", "irindwi", "umunani", "icyenda"],
    "bi": ["kimwe", "bibiri", "bitatu", "bine", "bitanu", "bitandatu", "birindwi", "umunani", "icyenda"],
    "zi": ["imwe", "ebyiri", "eshatu", "enye", "eshanu", "esheshatu", "zirindwi", "umunani", "icyenda"],
}

MONTHS = ["Mutarama", "Gashyantare", "Werurwe", "Mata", "Gicurasi", "Kamena",
          "Nyakanga", "Kanama", "Nzeri", "Ukwakira", "Ugushyingo", "Ukuboza"]

ABBREVIATIONS = {
    "Dr.": "Dogiteri",
    "Prof.": "Porofeseri",
    "Nyak.": "Nyakubahwa",
    "No.": "nimero",
    "n°": "nimero",
    "km": "kilometero",
    "kg": "kilogarama",
    "%": " ku ijana",
}


def _join(parts: list[str]) -> str:
    """Join number components with 'na' (the Kinyarwanda 'and')."""
    return " na ".join(p for p in parts if p)


def _below_100(n: int, cls: str) -> str:
    tens, unit = divmod(n, 10)
    unit_word = UNITS[cls][unit - 1] if unit else ""
    if tens == 0:
        return unit_word
    if tens == 1:
        tens_word = "icumi" if not unit else "cumi"
    elif tens == 2:
        tens_word = "makumyabiri"
    else:
        tens_word = "mirongo " + UNITS["i"][tens - 1]
    return _join([tens_word, unit_word])


def _below_1000(n: int, cls: str) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _below_100(rest, cls)
    h = "ijana" if hundreds == 1 else "magana " + UNITS["a"][hundreds - 1]
    return _join([h, _below_100(rest, cls) if rest else ""])


def number_to_words(n: int, cls: str = "count") -> str:
    """Spell out a non-negative integer. `cls` sets agreement of the final units."""
    if n == 0:
        return "zeru"
    if n < 0:
        return "munsi ya zeru " + number_to_words(-n, cls)
    parts: list[str] = []
    for value, sing, plur, mcls in ((10**9, "miliyari", "miliyari", "zi"),
                                    (10**6, "miliyoni", "miliyoni", "zi"),
                                    (1000, "igihumbi", "ibihumbi", "bi")):
        big, n = divmod(n, value)
        if big == 1:
            parts.append(sing if value == 1000 else f"{sing} {UNITS[mcls][0]}")
        elif big > 1:
            parts.append(f"{plur} {_below_1000(big, mcls)}")
    if n:
        parts.append(_below_1000(n, cls))
    return _join(parts)


def digits_to_words(s: str) -> str:
    """Read a string of digits one by one (phone numbers, IDs)."""
    return " ".join(number_to_words(int(d)) for d in s if d.isdigit())


def _to_int(s: str) -> int:
    return int(s.replace(",", "").replace(" ", "").replace(".", ""))


_NUM = r"\d{1,3}(?:[ ,.]\d{3})+|\d+"


def normalize(text: str) -> str:
    # Abbreviations
    for abbr, full in ABBREVIATIONS.items():
        if abbr == "%":
            continue
        text = re.sub(rf"(?<!\w){re.escape(abbr)}(?!\w)", full, text)

    # Money: "5,000 Frw", "RWF 5000", "Frw 5.000"
    text = re.sub(rf"(?:RWF|Rwf|Frw|FRW)\s*({_NUM})",
                  lambda m: "amafaranga " + number_to_words(_to_int(m[1]), "a"), text)
    text = re.sub(rf"({_NUM})\s*(?:RWF|Rwf|Frw|FRW|frw)\b",
                  lambda m: "amafaranga " + number_to_words(_to_int(m[1]), "a"), text)

    # Percent: "15%" -> "cumi na gatanu ku ijana"
    text = re.sub(rf"({_NUM})\s*%", lambda m: number_to_words(_to_int(m[1])) + " ku ijana", text)

    # Dates: 05/06/2024 or 5-6-2024 (day/month/year, Rwandan order)
    def _date(m: re.Match) -> str:
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m[0]
        return f"tariki ya {number_to_words(d)} {MONTHS[mo - 1]} {number_to_words(y)}"
    text = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", _date, text)

    # Phone numbers: 07xxxxxxxx, +250 7xx xxx xxx -> digit by digit
    text = re.sub(r"(?:\+250\s?)?\b0?7\d(?:\s?\d){7}\b", lambda m: digits_to_words(m[0]), text)

    # Remaining plain numbers
    text = re.sub(_NUM, lambda m: number_to_words(_to_int(m[0])), text)

    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    import sys
    print(normalize(" ".join(sys.argv[1:]) or sys.stdin.read()))


if __name__ == "__main__":
    main()
