"""Readings a native speaker has checked. Add a case every time you fix one."""
import pytest

from kintts.normalize import normalize, number_to_words

@pytest.mark.parametrize("n, words", [
    (1, "rimwe"), (5, "gatanu"), (10, "icumi"), (11, "cumi na rimwe"),
    (20, "makumyabiri"), (25, "makumyabiri na gatanu"), (30, "mirongo itatu"),
    (100, "ijana"), (200, "magana abiri"), (1000, "igihumbi"),
    (2000, "ibihumbi bibiri"), (5000, "ibihumbi bitanu"),
    (2024, "ibihumbi bibiri na makumyabiri na kane"),
    (1_000_000, "miliyoni imwe"), (2_000_000, "miliyoni ebyiri"),
])
def test_numbers(n, words):
    assert number_to_words(n) == words

@pytest.mark.parametrize("text, spoken", [
    ("Ni 5,000 Frw.", "Ni amafaranga ibihumbi bitanu."),
    ("Yishyuye RWF 500", "Yishyuye amafaranga magana atanu"),
    ("Byazamutseho 15%", "Byazamutseho cumi na gatanu ku ijana"),
    ("Yavutse 05/06/2001", "Yavutse tariki ya gatanu Kamena ibihumbi bibiri na rimwe"),
    ("Dr. Mukamana", "Dogiteri Mukamana"),
])
def test_normalize(text, spoken):
    assert normalize(text) == spoken
