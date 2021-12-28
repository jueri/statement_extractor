import os

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

    def classify(self, input_text: str) -> bool:
        """Classify the input text

        Args:
            input_text (str): Sentence to be classified.

        Returns:
            bool: True if sentence contains a claim.
        """
        input_text_tokenized = self.tokenize_text(input_text)  # tokenize Text
        prediction = self.model(input_text_tokenized)
        prediction_logits = prediction[0]
        prediction_probs = tf.nn.softmax(prediction_logits, axis=1).numpy()
        return prediction_probs

    def is_claim(self, input_text):
        input_text_tokenized = self.tokenize_text(input_text)
        prediction = self.model(input_text_tokenized)
        prediction_logits = prediction[0]
        result = np.argmax(prediction_logits, axis=1)[0]
        if result == 1:
            return True
        else:
            return False
