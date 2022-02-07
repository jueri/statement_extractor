# -*- coding: utf-8 -*-
"""Module to detect main concepts and similarities. All options can be displayed by calling `python statementExtractor.py --help`.

Example:
    $ python statementExtractor.py -i Press-Briefing.pdf --pdf_out test.pdf --main_concept wikify_intro --intro intro.txt --main_concept_th 1.0 --claim_th 0.8

"""

import datetime
import os

import click
import pandas as pd
import transformers
from typing import Tuple, Dict, Any, List

transformers.logging.set_verbosity_error()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import MODEL_NAME, MODEL_WEIGHTS_PATH, TAGME_TOKEN, DANDELION_TOKEN
from src import detect_claims, highlight_pdf, segmenter
from src.utils import pdf_parser
from src.main_concept import sentence_title_similarity, wikified_article_score, wikify

pd.options.mode.chained_assignment = None  # default='warn'


def load_data(pdf_in: str) -> Tuple[str, str, str, Dict[str, Any]]:
    """Parse the PDF file from the given directory into the full text, title creation date and passages.

    Args:
        pdf_in (str): Path to the input PDF file.

    Returns:
        Tuple[str, str, str, Dict[str, Any]]: text, title, date and passages.
    """
    text = pdf_parser.pdf_to_text(pdf_in)  # load all text from pdf
    parse = pdf_parser.parse_text(text)  # parse pdf with specific document parser

    title = parse["title"]
    date = parse["date"]
    passages = parse["passages"]  # this document has natural passages
    return text, title, date, passages


def create_segments(passages: List[Dict[str, Any]], length: int) -> pd.DataFrame:
    """Create a table with a row for each sentence and further information about the passage and sentence ID.

    Args:
        passages (List[Dict[str, Any]]): Passages of the input text.
        length (int):  The number of sentences that should be combined into a statement.

    Returns:
        pd.DataFrame: Sentences and metadata as a table.
    """

    doc = []  # create empty doc
    passage_id = 0

    for passage in passages:
        if length > 1:
            segments = segmenter.split_segments(passage.get("text"), segment_len=length)
        else:
            sentences = segmenter.split_sentences(passage.get("text"))
            segments = [{"sentence": sentence} for sentence in sentences]

        for segment in segments:
            segment["passage_id"] = passage_id
            doc.append(segment)

        passage_id += 1
    doc_table = pd.DataFrame(doc)
    return doc_table


def get_similarity(
    main_concept: str,
    text: str,
    title: str,
    doc_table: pd.DataFrame,
    intro: str,
    main_concept_th: float,
) -> pd.DataFrame:
    """Calculate the similarity between the press briefing itself and each sentence based on the provided method.

    Args:
        main_concept (str): Name of the method, one of "embedding", "wikify_title" or "wikify_intro".
        text (str): Full text of the press briefing.
        title (str): Title of the press briefing.
        doc_table (pd.DataFrame): Tabulated press briefing.
        intro (str): Path to the intro text file.
        main_concept_th (float): Threshold the similarity score has to exceed.

    Returns:
        pd.DataFrame: Dataframe of only relevant rows.
    """
    if main_concept == "embedding":
        similarities = sentence_title_similarity(text, title, doc_table["sentence"].to_list())
        doc_table["similaritie"] = similarities

    elif main_concept == "wikify_title":
        title_wikifyed = wikify(text=title, service="tagme", token=TAGME_TOKEN)
        title_article_ids = {
            article["id"] for article in title_wikifyed
        }  # get all article ids from title
        doc_table["similaritie"] = doc_table.apply(
            lambda x: wikified_article_score(x["sentence"], title_article_ids), axis=1
        )

    elif main_concept == "wikify_intro":
        assert os.path.isfile(intro)
        with open(intro, "r") as inFile:
            intro_text = inFile.read()
        intor_wikifyed = wikify(text=intro_text, service="tagme", token=TAGME_TOKEN)
        intro_article_ids = {
            article["id"] for article in intor_wikifyed
        }  # get all article ids from introduction
        doc_table["similaritie"] = doc_table.apply(
            lambda x: wikified_article_score(x["sentence"], intro_article_ids), axis=1
        )
    doc_table = doc_table[doc_table["similaritie"] > main_concept_th]
    return doc_table


def detect_claim_sentences(doc_table: pd.DataFrame, claim_th: float) -> pd.DataFrame:
    """Apply the claim detection model to the tabulated press briefing.

    Args:
        doc_table (pd.DataFrame): Tabulated press briefing.
        claim_th (float): Threshold the confidence score of the model has to exceed.

    Returns:
        pd.DataFrame: Tabulated press briefing with only claim sentences.
    """
    detector = detect_claims.claim_detector(MODEL_NAME, MODEL_WEIGHTS_PATH)
    if claim_th:
        doc_table["claim"] = doc_table.apply(
            lambda x: detector.is_claim(x["sentence"], min_confidence=claim_th), axis=1
        )
    else:
        doc_table["claim"] = doc_table.apply(lambda x: detector.is_claim(x["sentence"]), axis=1)

    relevant_claims = doc_table[
        (doc_table["claim"] == True) & (doc_table["speaker"] != "Moderator")
    ]
    return relevant_claims


def highlight(
    doc_table: pd.DataFrame,
    relevant_claims: pd.DataFrame,
    length: int,
    pdf_in: str,
    pdf_out: str,
):
    """Highlight the press briefing.

    Args:
        doc_table (pd.DataFrame): Tabulated press briefing.
        relevant_claims (pd.DataFrame): Table of relevant claim sentences.
        length (int): Statment lenghth in sentences.
        pdf_in (str): Path to input pdf.
        pdf_out (str): Output path for the results pdf.
    """
    if length > 1:
        click.echo(str(datetime.datetime.now()) + ": Highlight statements.")
        c = "yellow"
        highlight_pdf.highlight_text([], pdf_in, pdf_out)

        passage_id_old = None
        segment_id_old = None

        for row in relevant_claims.iterrows():
            passage_id = row[1]["passage_id"]
            segment_id = row[1]["segment_id"]

            if passage_id_old == passage_id and segment_id_old == segment_id:
                continue
            else:
                sentences = doc_table[
                    (doc_table["passage_id"] == passage_id)
                    & (doc_table["segment_id"] == segment_id)
                ]["sentence"].to_list()
                highlight_pdf.highlight_text(sentences, pdf_out, pdf_out, color=c)
                passage_id_old = passage_id
                segment_id_old = segment_id

                if c == "yellow":
                    c = "red"
                else:
                    c = "yellow"
    else:
        click.echo(str(datetime.datetime.now()) + ": Highlight claims.")
        claim_sentences = relevant_claims["sentence"].to_list()
        highlight_pdf.highlight_text(claim_sentences, pdf_in, pdf_out, color=["green", "yellow"])


@click.command()
@click.option("--pdf_in", "-i", help="Path to the input pdf file.")
@click.option("--pdf_out", "-o", help="Output path.")
@click.option("--length", "-n", default=1, help="Number of sentences per statement.")
@click.option(
    "--main_concept",
    "-m",
    help='Method to detect the main concept for similarity score calculation. Options are "embedding", "wikify_title", "wikify_intro".',
)
@click.option("--intro", help="Path to the introduction text to be used for wikification.")
@click.option(
    "--main_concept_th", default=0.0, help="Threshold for a minimum main concept similarity"
)
@click.option(
    "--claim_th",
    default=0.0,
    help="Minimal claim confidence betweene 0.1 and 1.0, by default the max confidence class is choosen.",
)
def annotate(pdf_in, pdf_out, length, main_concept, intro, main_concept_th, claim_th):
    # Load data
    click.echo(str(datetime.datetime.now()) + ": Parse pdf file.")
    text, title, date, passages = load_data(pdf_in)

    # create segments
    click.echo(str(datetime.datetime.now()) + ": Create segments.")
    doc_table = create_segments(passages, length)

    # add metadata
    doc_table["title"] = title
    doc_table["date"] = date

    doc_table["speaker"] = None
    doc_table["timestamp"] = None

    for idx, passage in enumerate(passages):
        speaker = passage.get("speaker")
        timestamp = passage.get("timestamp")

        doc_table["speaker"].loc[doc_table["passage_id"] == idx] = speaker
        doc_table["timestamp"].loc[doc_table["passage_id"] == idx] = timestamp

    # detect main concept
    if main_concept:
        click.echo(str(datetime.datetime.now()) + ": Detect main concept.")
        if main_concept != "wikify_intro":
            intro = ""
        doc_table = get_similarity(main_concept, text, title, doc_table, intro, main_concept_th)

    # detect claim sentences
    click.echo(str(datetime.datetime.now()) + ": Detect claim sentences.")
    relevant_claims = detect_claim_sentences(doc_table, claim_th)

    # highlight
    highlight(doc_table, relevant_claims, length, pdf_in, pdf_out)


if __name__ == "__main__":
    annotate()
