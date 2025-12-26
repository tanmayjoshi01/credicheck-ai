from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sample_text = "Breaking: Scientists discover that eating chocolate every day can cure all diseases. A groundbreaking study published today reveals that consuming dark chocolate has miraculous health benefits that were previously unknown to medical science."

result = classifier(sample_text, candidate_labels=["fake news", "real news"])

label = result["labels"][0]
confidence = result["scores"][0]


if "fake" in label.lower():
    prediction_label = "Fake"
else:
    prediction_label = "Real"


print(f"Prediction: {prediction_label}")
print(f"Confidence: {confidence:.2f}")
