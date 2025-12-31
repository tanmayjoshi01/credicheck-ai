from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
from duckduckgo_search import DDGS

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

tokenizer_style = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model_style = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

nlp = spacy.load("en_core_web_sm")

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

def analyze_text_style(text):
    inputs = tokenizer_style(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model_style(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence_scores = predictions[0].tolist()
    max_index = confidence_scores.index(max(confidence_scores))
    style_confidence = confidence_scores[max_index]
    
    if max_index == 1:
        style_label = "Fake"
    else:
        style_label = "Real"
    
    return {
        "style_label": style_label,
        "style_confidence": style_confidence
    }

def extract_entities(text):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entities.append(ent.text)
    
    return entities

def verify_headline(headline):
    trusted_domains = ["bbc", "reuters", "apnews", "npr", "gov", "edu"]
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(headline, max_results=3))
        
        for result in results:
            url = result.get('href', '').lower()
            for domain in trusted_domains:
                if domain in url:
                    source_name = domain.capitalize()
                    if domain == "gov":
                        source_name = "Government"
                    elif domain == "edu":
                        source_name = "Educational"
                    return f"Verified by {source_name}"
        
        return "Unverified claim"
    except Exception:
        return "Unverified claim"

def analyze_text(text):
    inputs = tokenizer_style(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model_style(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    confidence_scores = predictions[0].tolist()
    max_index = confidence_scores.index(max(confidence_scores))
    score = round(float(confidence_scores[max_index]), 2)
    
    if max_index == 1:
        label = "Fake"
    else:
        label = "Real"
    
    # Extract entities
    doc = nlp(text)
    entities = []
    seen = set()
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entity_text = ent.text
            if entity_text not in seen:
                entities.append(entity_text)
                seen.add(entity_text)
    
    # Verify headline
    verification_note = verify_headline(text)
    
    return {
        "label": label,
        "score": score,
        "entities": entities,
        "verification_note": verification_note
    }

