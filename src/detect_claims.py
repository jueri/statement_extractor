from transformers import TFDistilBertForSequenceClassification

loaded_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
loaded_model.load_weights("./distillbert_tf.h5")
input_text = "The text on which I test"
input_text_tokenized = tokenizer.encode(
    input_text, truncation=True, padding=True, return_tensors="tf"
)
prediction = loaded_model(input_text_tokenized)
prediction_logits = prediction[0]
prediction_probs = tf.nn.softmax(prediction_logits, axis=1).numpy()
print(f"The prediction probs are: {prediction_probs}")
