from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

clip_model = SentenceTransformer('clip-ViT-B-32')

def generate_image_explanation(label, confidence):
    if confidence >= 0.75:
        if "AI-generated" in label:
            return "The analysis suggests this image may be artificially generated. Consider verifying its authenticity through reverse image search or source verification."
        else:
            return "The analysis suggests this appears to be a genuine photograph. However, images can be manipulated, so verify the source when possible."
    elif confidence >= 0.55:
        if "AI-generated" in label:
            return "The analysis indicates some characteristics suggesting artificial generation. Exercise caution and verify the image source and authenticity."
        else:
            return "The analysis suggests this may be a real photograph, but confidence is moderate. Additional verification of the image source is recommended."
    else:
        return "The analysis results are inconclusive. It is strongly recommended to verify the image authenticity through reverse image search and source verification before making any decisions."

def analyze_image(image_path):
    image = Image.open(image_path)
    
    candidate_descriptions = ["AI-generated image", "Real photograph"]
    
    inputs = processor(text=candidate_descriptions, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    
    confidence_scores = probs[0].tolist()
    
    max_index = confidence_scores.index(max(confidence_scores))
    image_label = candidate_descriptions[max_index]
    image_confidence = confidence_scores[max_index]
    
    explanation = generate_image_explanation(image_label, image_confidence)
    
    return image_label, image_confidence, explanation

def verify_image_context(image_path, headline_text):
    image = Image.open(image_path)
    
    image_embedding = clip_model.encode(image)
    text_embedding = clip_model.encode(headline_text)
    
    image_embedding = image_embedding.reshape(1, -1)
    text_embedding = text_embedding.reshape(1, -1)
    
    similarity = cosine_similarity(image_embedding, text_embedding)[0][0]
    
    if similarity < 0.20:
        image_context = "Mismatch"
    elif similarity >= 0.25:
        image_context = "Consistent"
    else:
        image_context = "Mismatch"
    
    return {
        "image_context": image_context,
        "similarity_score": float(similarity)
    }

