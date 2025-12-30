from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def generate_text_explanation(label, confidence):
    if confidence >= 0.75:
        if label == "Fake":
            return "The analysis suggests this content may contain false or misleading information. Consider verifying claims with reputable sources before sharing."
        else:
            return "The analysis suggests this content appears credible. However, always verify important information through multiple reliable sources."
    elif confidence >= 0.55:
        if label == "Fake":
            return "The analysis indicates some characteristics of unreliable content. Exercise caution and verify information from trusted sources."
        else:
            return "The analysis suggests this content may be credible, but confidence is moderate. Additional verification is recommended."
    else:
        return "The analysis results are inconclusive. It is strongly recommended to verify this information through multiple reputable sources before making any decisions."

def analyze_text(text):
    result = classifier(text, candidate_labels=["fake news", "real news"])
    
    label = result["labels"][0]
    confidence = result["scores"][0]
    
    if "fake" in label.lower():
        prediction_label = "Fake"
    else:
        prediction_label = "Real"
    
    explanation = generate_text_explanation(prediction_label, confidence)
    
    return prediction_label, confidence, explanation

