# -*- coding: utf-8 -*-
"""Module to segment texts into parts like sentences or coherent segments.

Example:
    $ from segmenter import segment_passages, segment_sentences
    $
    $ segments = split_segments(text)
    $
    $ sentences = split_sentences(text)


"""

from typing import Optional, Any

import numpy as np  # type: ignore
import pandas as pd  # type:ignore
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer  # type:ignore
from textsplit.algorithm import (
    get_total,
    split_greedy,  # type:ignore
    split_optimal,
)
from textsplit.tools import get_penalty, get_segments

from src.utils import text_sanitizer
from config import SPACY_DATA_PATH, DEBUG


def get_embeddings(text: str, model_path: str) -> pd.DataFrame:
    """Create a patch DataFrame with embedding vectors as values and the tokens as index. Embeddings are calculated with spacy.

    Args:
        text (str): String of text to calculate the vocabulary and embeddings of.

    Returns:
        pd.DataFrame: DataFrame with tokens as index and embeddings as values.
    """

    def _get_save_vector(token: str) -> Optional[np.ndarray]:
        """Get a vector of a token. If spacy returns a vector of zeros, None is returned

        Args:
            token (str): single token to create an embedding of.

        Returns:
            Union[np.ndarray, None]: embedding array or none
        """
        vector = nlp(token).vector
        if str(vector) != str(BAD_VECTOR):
            return vector  # type: ignore
        else:
            return None

    nlp = spacy.load(model_path)
    BAD_VECTOR = nlp("hrgenbrmpf").vector

    vocabulary = list(set(word_tokenize(text)))  # unique tokens from text
    word_vectors = pd.DataFrame(vocabulary)
    word_vectors["vec"] = word_vectors[0].apply(
        lambda x: _get_save_vector(x)
    )  # get all wordvactors
    word_vectors.dropna(inplace=True)  # delete tokens without embeddings in the vocabulary
    word_vectors.set_index(0, inplace=True)
    return word_vectors


def get_sentence_embeddings(sentenced_text: list[str], word_vectors: pd.DataFrame) -> np.ndarray:
    """Claculate an embaddings for full sentences of a list of sentences. The embeddings are calculated
     ad dot product of the single token embedding.

    Args:
        sentenced_text (list[str]): List of sentences
        word_vectors (pd.DataFrame):

    Returns:
        np.ndarray:  DataFrame with tokens as index and embeddings as values.
    """
    vecr = CountVectorizer(vocabulary=word_vectors.index)
    sentence_vectors = vecr.transform(sentenced_text).dot(list(word_vectors["vec"].to_numpy()))
    return sentence_vectors


def make_segments(
    sentence_vectors: np.ndarray,
    sentenced_text: list[str],
    segment_len: int,
    seg_limit: int = 250,
    greedy: bool = False,
) -> list[list[str]]:
    """Segment a text in segments using sentence embeddings.

    Args:
        sentence_vectors (np.ndarray):
        sentenced_text (list[str]):
        seg_limit (int):
        greedy (bool):

    Returns:
        list[list[str]]:
    """
    penalty = get_penalty([sentence_vectors], segment_len)

    if DEBUG:
        print("penalty %4.2f" % penalty)

    optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=seg_limit)
    segmented_text = get_segments(sentenced_text, optimal_segmentation)

    if greedy or DEBUG:
        greedy_segmentation = split_greedy(
            sentence_vectors, max_splits=len(optimal_segmentation.splits)
        )
        greedy_segmented_text = get_segments(sentenced_text, greedy_segmentation)

    if DEBUG:
        print(
            "%d sentences, %d segments, avg %4.2f sentences per segment"
            % (len(sentenced_text), len(segmented_text), len(sentenced_text) / len(segmented_text))
        )

        lengths_optimal = [len(segment) for segment in segmented_text for sentence in segment]
        lengths_greedy = [len(segment) for segment in greedy_segmented_text for sentence in segment]
        df: pd.DataFrame = pd.DataFrame({"greedy": lengths_greedy, "optimal": lengths_optimal})
        df.plot.line(figsize=(18, 3), title="Segment lenghts over text")

        df.plot.hist(bins=30, alpha=0.5, figsize=(10, 3), title="Histogram of segment lengths")

        totals = [
            get_total(sentence_vectors, seg.splits, penalty)
            for seg in [optimal_segmentation, greedy_segmentation]
        ]
        print("optimal score %4.2f, greedy score %4.2f" % tuple(totals))
        print("ratio of scores %5.4f" % (totals[0] / totals[1]))

    if greedy:
        return greedy_segmented_text
    else:
        return segmented_text


def split_sentences(text: str, min_sentence_lenghth: int = 3) -> list[str]:
    """Split a text into sentences.

    Args:
        text (str): Text to split.
        min_sentence_lenghth (int, optional): Minimum number of words in a sentence. Defaults to 3.

    Returns:
        list[str]: List of sentences from text.
    """
    sentence_tokens = sent_tokenize(text)
    sentenced_text_clean = []
    for sentence in sentence_tokens:
        if not len(sentence.split(" ")) < min_sentence_lenghth:
            sentenced_text_clean.append(sentence.strip())
    return sentenced_text_clean


def split_segments(text: str) -> list[dict[str, str]]:
    """Split text into coherent segments using the textslpit algorythm.

    Args:
        text (str): Text to split.

    Returns:
        list[dict[str, str]]: List of dicts with sentences and segment ids.
    """

    cleaned_full_text = text_sanitizer.sanetize(text)
    word_vectors = get_embeddings(cleaned_full_text, SPACY_DATA_PATH)

    cleaned = text_sanitizer.sanetize(text)

    sentences = split_sentences(cleaned)
    sentence_vectors = get_sentence_embeddings(sentences, word_vectors)
    try:
        segments: list[list[str]] = make_segments(sentence_vectors, sentences, segment_len=5)
    except ValueError:
        segments = [sentences]  # not enough sentences in segment use all sentences as one passage

    dataset = []
    for segment in segments:
        segment_id = 0
        for sentence in segment:
            dataset.append(
                {
                    "segment_id": segment_id,
                    "sentence": sentence,
                }
            )
            segment_id += 1
    return dataset  # type: ignore
