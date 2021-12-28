# -*- coding: utf-8 -*-
"""Module to highlight pdfs.

Example:
    $ from highlight_pdf import highlight_text
    $
    $ highlight_text(claim_sentences, SAMPLE_PDF_PATH, "claims_test.pdf")
"""

import fitz


def highlight_text(texts: list[str], input_pdf_path: str, output_pdf_path: str):
    """Highlight text in a pdf file. The texts are searched and higlighted. If the search returns no results an error is printed.

    Args:
        texts (list[str]): List of strings to be searched and highlighted.
        input_pdf_path (str): Path to the original pdf.
        output_pdf_path (str): Path the highlighted pdf will be saved to.
    """

    pdf = fitz.open(input_pdf_path)

    for text in texts:
        text_instances = []
        for page in pdf:
            instances = page.search_for(text)
            text_instances += instances

            for inst in instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()
        if not text_instances:
            print("Error, sentence not found")

    pdf.save(output_pdf_path, garbage=4, deflate=True, clean=True)
