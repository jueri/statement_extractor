import os
import nltk
import subprocess
import spacy

DEBUG: bool = False


SAMPLE_PDF_PATH = os.path.join(
    "data", "pdf", "Transkript_Quantencomputer_SMC-Press-Briefing_2021-04-12.pdf"
)


# NLTK
NLTK_DATA_PATH = os.path.join("data", "nltk_data")

# setup NLTK
if not os.path.exists(NLTK_DATA_PATH):
    os.makedirs(NLTK_DATA_PATH)
    # nltk.download("vader_lexicon", NLTK_DATA_PATH)
    # nltk.download("wordnet", NLTK_DATA_PATH)
    # nltk.download("stopwords", NLTK_DATA_PATH)
    nltk.download("punkt", NLTK_DATA_PATH)
nltk.data.path.append(NLTK_DATA_PATH)


# Spacy
SPACY_MODEL_NAME = "de_core_news_lg"
SPACY_DATA_PATH = os.path.join("data", "spacy_data", SPACY_MODEL_NAME)

if not os.path.exists(SPACY_DATA_PATH):
    subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL_NAME])  # download model
    os.makedirs(SPACY_DATA_PATH)  # create foleder
    nlp = spacy.load(SPACY_MODEL_NAME)  # load model
    nlp.to_disk(SPACY_DATA_PATH)  # save model to created dir


# Model
MODEL_NAME = "deepset/gbert-base"
MODEL_WEIGHTS_PATH = os.path.join("data", "models", "SMC_Full_895", "tf_model.h5")
# MODEL_NAME = "deepset/gbert-base"
# MODEL_WEIGHTS_PATH = os.path.join("data", "models", "CM_gbert-base_full_g", "tf_model.h5")


# Wikify
DANDELION_TOKEN = os.getenv("DANDELION_TOKEN")
TAGME_TOKEN = os.getenv("TAGME_TOKEN")
