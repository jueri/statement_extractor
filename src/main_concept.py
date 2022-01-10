# -*- coding: utf-8 -*-
"""Module to detect main concepts and similarities.

Example:
    $ from src.main_concept import sentence_title_similarity
    $
    $ similarities = sentence_title_similarity(text, title, doc_table["sentence"].to_list())

"""
from config import SPACY_DATA_PATH
from scipy import spatial

from src import segmenter


def sentence_title_similarity(text: str, title: str, text_sentences: list[str]) -> list[float]:
    """Calculate the similarities between the sentences and the title as the cosine similarity between the sentence and topic embedding.

    Args:
        text (str): Fulltext string of the full document to efficiently calculate embeddings for all words.
        title (str): Title of the document.
        text_sentences (list[str]): List of the sentence tokenised document.

    Returns:
        list[float]: list of absolute distances.
    """
    word_vectors = segmenter.get_embeddings(text + title, SPACY_DATA_PATH)

    title_vec = segmenter.get_sentence_embeddings([title], word_vectors)[0]

    sentence_embeddings = segmenter.get_sentence_embeddings(text_sentences, word_vectors)

    similarities = []
    for emb in sentence_embeddings:
        similarities.append(abs(1 - spatial.distance.cosine(emb, title_vec)))

    return similarities
