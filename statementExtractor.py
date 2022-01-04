import click
import pandas as pd
import datetime
import transformers

transformers.logging.set_verbosity_error()

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.utils import pdf_parser
from src import segmenter, detect_claims, highlight_pdf
from config import MODEL_NAME, MODEL_WEIGHTS_PATH


@click.command()
@click.option("--pdf_in", help="Path to the pdf file to parse.")
@click.option("--pdf_out", help="Output path.")
@click.option("--segment_len", default=4, help="Number of sentences per segments.")
def annotate(pdf_in, pdf_out, segment_len):

    click.echo(str(datetime.datetime.now()) + ": Parse pdf file.")
    text = pdf_parser.pdf_to_text(pdf_in)  # load all text from pdf
    parse = pdf_parser.parse_text(text)  # parse pdf with specific document parser

    title = parse["title"]
    date = parse["date"]
    passages = parse["passages"]  # this document has natural passages

    doc = []  # create empty doc
    passage_id = 0

    # create segments
    click.echo(str(datetime.datetime.now()) + ": Create segments.")
    for passage in passages:
        segments = segmenter.split_segments(passage.get("text"), segment_len=segment_len)

        for segment in segments:
            segment["passage_id"] = passage_id
            doc.append(segment)

        passage_id += 1

    # load to table
    doc_table = pd.DataFrame(doc)

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

    # detect claim sentences
    click.echo(str(datetime.datetime.now()) + ": Detect claim sentences.")
    detector = detect_claims.claim_detector(MODEL_NAME, MODEL_WEIGHTS_PATH)
    doc_table["claim"] = doc_table.apply(lambda x: detector.is_claim(x["sentence"]), axis=1)

    relevant_claims = doc_table[
        (doc_table["claim"] == True) & (doc_table["speaker"] != "Moderator")
    ]
    claim_sentences = relevant_claims["sentence"].to_list()

    # highlight
    click.echo(str(datetime.datetime.now()) + ": Highlight results.")
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
                (doc_table["passage_id"] == passage_id) & (doc_table["segment_id"] == segment_id)
            ]["sentence"].to_list()
            highlight_pdf.highlight_text(sentences, pdf_out, pdf_out, color=c)
            passage_id_old = passage_id
            segment_id_old = segment_id

            if c == "yellow":
                c = "red"
            else:
                c = "yellow"

    # highlight_pdf.highlight_text(claim_sentences, pdf_out, pdf_out, color="green")


if __name__ == "__main__":
    # python statementExtractor.py --pdf_in ./data/pdf/Transkript_Quantencomputer_SMC-Press-Briefing_2021-04-12.pdf --pdf_out ./test.pdf --segment_len 4
    annotate()
