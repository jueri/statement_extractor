# -*- coding: utf-8 -*-
"""Parse pdfs to structured PressBriefing dicts. The pdf text is extracted and parsed 
using simple regex matching.

Todo:
    * Improve parser to be more robust
    * retrieve video url from website !not in pdf!
"""

import io
import re
from typing import Any, Callable, Dict, List, Match, Pattern, Optional, Union

from pdfminer.converter import TextConverter  # type: ignore
from pdfminer.layout import LAParams  # type: ignore
from pdfminer.pdfdocument import PDFDocument  # type: ignore
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter  # type: ignore
from pdfminer.pdfpage import PDFPage  # type: ignore
from pdfminer.pdfparser import PDFParser  # type: ignore


def pdf_to_text(file_path: str) -> str:
    """Extrect the text of an pdf from the given path. Hyphens are identifyed by `-\n` and deleted.

    Args:
        path (str): Path of the PDF to extract the text from.

    Returns:
        str: Textual contents of th PDF file.
    """
    output_string = io.StringIO()
    with open(file_path, "rb") as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue().replace("-\n", "")


def parse_text(text: str):
    """Specific parser for the SMC PressBriefing PDFs. Extrects the metadata as well as data
    from the given text file.

    Args:
        text (str): Text string on an parsed PDF file.

    Returns:
        Dict: Structured Press briefing.
    """

    def match_date(line: str):
        """Match dates in the format xx.xx.xx from a given string. The date is automatcly set
        as the PressBriefing date and true is returned if successfull.

        Args:
            line (str): Single line of the PressBriefing

        Returns:
            Bool: True if sucessfull
        """
        date: Pattern[str] = re.compile(r"\d+\.\d+\.\d{4}")
        result: Optional[Match[str]] = date.match(line)
        if bool(result):  # catch empty results
            pb["date"] = result.string.strip()  # type: ignore
            return True

    def match_url(line: str) -> Any:
        """Match URLs from a given string. The URL is automaticly added as PressBriefing URL.

        Args:
            line (str): Single line of the PressBriefing

        Returns:
            Bool: True if sucessfull
        """
        url: Pattern[str] = re.compile(
            r"(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))"
        )
        result: Optional[Match[str]] = url.search(line)
        if bool(result):  # catch empty results
            if "youtube" in result.group(1).strip():  # type: ignore
                pb["video_url"] = result.group(1).strip()  # type: ignore
                return True
            elif "sciencemediacenter" in result.group(1).strip():  # type: ignore
                pb["pdf_file"] = result.group(1).strip()  # type: ignore
                return True

    def match_name(line: str) -> Any:
        """Matches names from the transcripts in a given line. If the line starts with
        the Unicode Token \uf075 and all following tokens start with uppercase letters the
        name is metched.

        Args:
            line (str): Single line of the PressBriefing

        Returns:
            Str: The extracted name
        """

        def check_capital_letters(tokens: List[str]):
            first_letters = []
            for token in tokens:
                if token != "":
                    first_letters.append(token[0])
            if all(letter.isupper() for letter in first_letters[1:]):
                return True

        tokens = line.strip().split(" ")
        if len(tokens) >= 2:

            if tokens[0].startswith(u"\uf075"):
                if check_capital_letters(tokens[1:]):
                    return " ".join(tokens).strip()
            else:
                if check_capital_letters(tokens):
                    return line.strip()

            # first_letters = []
            # for token in line.split(" ")[1:]:
            #     if token != "":
            #         first_letters.append(token[0])
            # if all(letter.isupper() for letter in first_letters[1:]):
            #     return line.split("  ")[1].strip()

    def match_timecode(line: str) -> Any:
        """Match and extract the timecode and speaker of a new passage.

        Args:
            line (str): Single line of the PressBriefing

        Returns:
            Dict or False: If a new passage could be indecated, a dict is extracted, containing
            the speaker and timestamp.
        """
        timecode: Pattern[str] = re.compile(
            r"(\[|\()\d+:\d+(:\d+)?(\]|\))"
        )  # [00:00] [00:00:00] (00:00)
        results: Optional[Match[str]] = timecode.search(line)
        if bool(results):  # catch empty results
            try:
                passage = {"timestamp": "", "text": "", "speaker": ""}
                if ":" in results.string.split("[")[0]:  # type: ignore
                    passage["speaker"] = results.string.split(": ")[0].strip()  # type: ignore
                    passage["timestamp"] = results.string.split(": ")[1].strip()  # type: ignore
                else:
                    tokens = results.string.split(" ")  # type: ignore
                    passage["speaker"] = " ".join(tokens)[:-1].strip()
                    passage["timestamp"] = tokens[-1:][0].strip()
                return passage
            except:
                return False

    names = []
    current_passage: Dict[str, list] = {}
    pb: Dict[str, Any] = {
        "title": "",
        "date": "",
        "guests": "",
        "host": "",
        "video_url": "",
        "passages": [],
        "pdf_file": "",
    }

    lines = text.split("\n")
    for line in lines:
        if line.strip() != "":

            if not pb.get("date"):
                if match_date(line):
                    continue

            if not pb.get("title"):
                if line.strip().startswith("„"):
                    pb["title"] = line
                    continue

            if pb.get("title"):
                if not pb.get("title").strip().endswith("“"):  # type: ignore
                    pb["title"] += line.rstrip()  # type: ignore
                    continue

            # if not pb.get("guests"):
            #     name = match_name(line)
            #     if name:
            #         names.append(name)
            #         continue

            if not pb.get("video_url") or not pb.get("pdf_file"):
                if match_url(line):
                    # pb["host"] = names[-1:]
                    # pb["guests"] = names[:-1]
                    continue

            if match_timecode(line):
                if current_passage:
                    pb["passages"].append(current_passage)  # type: ignore
                current_passage = match_timecode(line)
                continue

            if current_passage:
                current_passage["text"] += line
                if line.strip() == "Ansprechpartner in der Redaktion":
                    current_passage = {}

    if not pb.get("passages"):  # catch empty passage error
        raise ValueError("No passages found")

    for passage in pb.get("passages"):  # type: ignore
        names.append(passage.get("speaker"))  # type: ignore
    pb["guests"] = list(set(names))

    return pb
