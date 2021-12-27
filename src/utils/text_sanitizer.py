# -*- coding: utf-8 -*-
"""Functions to work with texts.

"""
import re


def sanetize(text: str) -> str:
    """Sanitize text by the following steps:
    1. delete all new lines
    2. replace piling whitespaces with single ones
    3. deleting the unicode character `\uf075 `

    Args:
        text (str): Input text to clean.

    Returns:
        str: Cleaned text.
    """
    text = text.replace("\n", "").strip()
    text = re.sub(" +", " ", text)
    text = text.replace("\uf075 ", "")
    return text
