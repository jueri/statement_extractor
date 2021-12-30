# -*- coding: utf-8 -*-
"""Module to highlight pdfs.

Example:
    $ from highlight_pdf import highlight_text
    $
    $ texts = ["Sentence number one", "sentence number two"]
    $
    $ highlight_text(texts, "my_pdf.pdf", "my_pdf.pdf", color="green")

"""

import fitz


def highlight_text(
    texts: list[str], input_pdf_path: str, output_pdf_path: str, color: str = "yellow"
):
    """Highlight text in a pdf file. The texts are searched and higlighted. If the search returns no results an error is printed.

    Args:
        texts (list[str]): List of strings to be searched and highlighted.
        input_pdf_path (str): Path to the original pdf.
        output_pdf_path (str): Path the highlighted pdf will be saved to.
    """
    colors = {
        "green": [0.8, 0.9, 0.7],
        "yellow": [1.0, 0.9, 0.5],
        "red": [0.9, 0.6, 0.5],
        "blue": [0.5, 0.5, 0.6],
        "grey": [0.6, 0.7, 0.7],
    }

    pdf = fitz.open(input_pdf_path)

    for text in texts:
        text_instances = []
        for page in pdf:
            instances = page.search_for(text)
            text_instances += instances

            for inst in instances:
                if isinstance(color, list):
                    color_item = color[len(text_instances) % len(color)]
                else:
                    color_item = color
                highlight = page.add_highlight_annot(
                    inst,
                )
                highlight.set_colors(stroke=colors.get(color_item, "yellow"))
                highlight.update()

        if not text_instances:
            print("Error, sentence not found")

    incremental = True if input_pdf_path == output_pdf_path else False
    pdf.save(output_pdf_path, encryption=False, incremental=incremental, deflate=True, clean=True)
