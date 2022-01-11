# -*- coding: utf-8 -*-
"""Module to detect main concepts and similarities.

Example:
    $ from src.main_concept import sentence_title_similarity
    $
    $ similarities = sentence_title_similarity(text, title, doc_table["sentence"].to_list())

"""
import time
from typing import Any, Optional

import requests
from scipy import spatial

from src import segmenter
from config import SPACY_DATA_PATH, TAGME_TOKEN


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


def wikify(text: str, service: str, token: str, language: str = "de") -> Optional[Any]:
    """Detect Wikipedia entities mentioned in a text. Choose one of two APIs implemented at the moment.

    Args:
        text (str): Text to retrieve entities of off.
        service (str): Name of the API. Either dandaleon or tagme.
        token (str): Token for the API.
        language (str, optional): Language of the input text. Defaults to "de".

    Returns:
        Optional[Any]: Result dictionary with detected nain_concepts if available or "Error" if not enough credits are left.
    """
    if len(text.split(" ")) < 4:
        return None

    if service == "dandelion":
        api = f"https://api.dandelion.eu/datatxt/nex/v1/?lang={language}&text={text}&token={token}"
        time.sleep(0.02)  # obey rate limit of 50 calls per second
    elif service == "tagme":
        api = f"https://tagme.d4science.org/tagme/tag?lang={language}&gcube-token={token}&tweet=false&text={text}"
        time.sleep(0.02)

    r = requests.get(api)
    if r.status_code == 403:
        print("ERROR: not enough credits left.")
        return "Error"

    if r.status_code == 200 and r.json().get("annotations"):
        return r.json().get("annotations")
    else:
        return None


def detect_main_concept(
    text: str, token: str, num_entetys: int = 1, language: str = "de"
) -> Optional[Any]:
    """Detect the main concept as a Wikipedia article from a given text. By using the `top_entities` parameter of the dandaleon api,
    the extracted entities are ranked accordingly to the relevance for the input text.

    Args:
        text (str): Text to retrieve entities of off.
        token (str): Token for the API.
        num_entetys (int, optional): Number of main entities for the text to return. Defaults to 1.
        language (str, optional): Language of the input text. Defaults to "de".

    Returns:
        Optional[Any]: Result dictionary with detected nain_concepts if available or "Error" if not enough credits are left.
    """
    if len(text.split(" ")) < 4:
        return None

    if len(text) > 4000:
        print("Error: Text is longer than 4000 chars.")
        return None

    num_entetys_str: str = str(num_entetys)
    api = f"https://api.dandelion.eu/datatxt/nex/v1/?lang={language}&text={text}&token={token}&top_entities={num_entetys_str}"
    time.sleep(0.02)  # obey rate limit of 50 calls per second

    r = requests.get(api)
    if r.status_code == 403:
        print("ERROR: not enough credits left.")
        return "Error"
    if r.status_code == 200 and r.json().get("annotations"):
        return r.json().get("annotations")
    else:
        return None


def wikified_article_score(sentence: str, document_spot_ids: set) -> float:
    """Claculate a main concept score from wikipedia articles. The probabilitie scores of spots both in the sentence and in the title are summed.

    Args:
        sentence (str): Sentence to calculate the score for.
        document_spot_ids (set): List of article IDs from the document.

    Returns:
        float: Score of the sentence.
    """
    spots = wikify(text=sentence, service="tagme", token=TAGME_TOKEN)
    if spots:
        return sum(
            {article["link_probability"] for article in spots if article["id"] in document_spot_ids}
        )  # sum all probabilities for all spots also in article
    else:
        return 0.0
