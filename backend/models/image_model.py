from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

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
    
    return image_label, image_confidence

