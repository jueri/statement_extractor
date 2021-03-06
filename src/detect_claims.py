# -*- coding: utf-8 -*-
"""Module to detect claims in sentences.

Example:
    $ from segmenter import segment_passages, segment_sentences
    $
    $ detector = detect_claims.claim_detector(MODEL_NAME, MODEL_WEIGHTS_PATH)
    $ 
    $ text = "This is not a claim sentence."
    $ detector.is_claim(text)

"""
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


class claim_detector:
    """Class to detect claims in sentences."""

    def __init__(self, model_name: str, model_weights_path: str):
        self.model_name = model_name
        self.model_weights_path = model_weights_path
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        """Load the tokenizer for the model.

        Returns:
            transformer tokenizer: Tokenizer object.
        """
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)  # load tokenizer
        return tokenizer

    def load_model(self):
        """Load the base model.

        Returns:
            object: Base model.
        """
        loaded_model = TFDistilBertForSequenceClassification.from_pretrained(self.model_name)
        loaded_model.load_weights(self.model_weights_path)
        return loaded_model

    def tokenize_text(self, input_text: str) -> np.ndarray:
        """Tokenize the input text with the apropreate tokenizer for the model and return the tokenized text.

        Args:
            input_text (str): Sentence to be classified.

        Returns:
            np.ndarray: Token sequence.
        """
        input_text_tokenized = self.tokenizer.encode(
            input_text, truncation=True, padding=True, return_tensors="np"
        )
        return input_text_tokenized

    def classify(self, input_text: str) -> np.ndarray:
        """Classify the input text as claim or non claim and return the result scores.

        Args:
            input_text (str): Sentence to be classified.

        Returns:
            np.ndarray: Class probabilitys of the sentence.
        """
        input_text_tokenized = self.tokenize_text(input_text)  # tokenize Text
        prediction = self.model(input_text_tokenized)
        prediction_logits = prediction[0]
        prediction_probs = tf.nn.softmax(prediction_logits, axis=1).numpy()
        return prediction_probs

    def is_claim(self, input_text: str, min_confidence: float = 0.0) -> bool:
        """Classify the input text as claim or non claim and return a boolean value.

        Args:
            input_text (str): Sentence to be classified.
            min_confidence (float): Minimal confidence to count as claim if 0.0 the highest available confidence is used independently of any threashold. Defaults to 0.0.

        Returns:
            bool: True if the sentence is a claim, else False.
        """
        input_text_tokenized = self.tokenize_text(input_text)
        prediction = self.model(input_text_tokenized)
        prediction_logits = prediction[0]
        if min_confidence == 0.0:
            result = np.argmax(prediction_logits, axis=1)[0]
        else:
            prediction_probs = tf.nn.softmax(prediction_logits, axis=1).numpy()
            if prediction_probs[0][1] >= min_confidence:
                result = True
            else:
                result = False

        if result == 1:
            return True
        else:
            return False

