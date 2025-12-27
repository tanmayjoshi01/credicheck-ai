from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sample_text = "Breaking: Scientists discover that eating chocolate every day can cure all diseases. A groundbreaking study published today reveals that consuming dark chocolate has miraculous health benefits that were previously unknown to medical science."

result = classifier(sample_text, candidate_labels=["fake news", "real news"])

label = result["labels"][0]
confidence = result["scores"][0]


if "fake" in label.lower():
    prediction_label = "Fake"
else:
    prediction_label = "Real"


print(f"Text Prediction: {prediction_label}")
print(f"Text Confidence: {confidence:.2f}")

image_path = "test_image.jpg"
image = Image.open(image_path)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

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

print(f"Image Prediction: {image_label}")
print(f"Image Confidence: {image_confidence:.2f}")
