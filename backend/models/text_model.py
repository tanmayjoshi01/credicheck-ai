from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_text(text):
    result = classifier(text, candidate_labels=["fake news", "real news"])
    
    label = result["labels"][0]
    confidence = result["scores"][0]
    
    if "fake" in label.lower():
        prediction_label = "Fake"
    else:
        prediction_label = "Real"
    
    return prediction_label, confidence

