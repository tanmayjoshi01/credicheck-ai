from flask import Flask, request, jsonify
from models.text_model import analyze_text
from models.image_model import verify_image_context
import hashlib
import time

app = Flask(__name__)

cache = {}

@app.route('/analyze-text', methods=['POST'])
def analyze_text_endpoint():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text field'}), 400
    
    text = data['text']
    
    try:
        label, confidence, explanation = analyze_text(text)
        return jsonify({
            'label': label,
            'confidence': round(confidence, 2),
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image_endpoint():
    data = request.json
    
    if not data or 'image_path' not in data:
        return jsonify({'error': 'Missing image_path field'}), 400
    
    if 'headline' not in data:
        return jsonify({'error': 'Missing headline field'}), 400
    
    image_path = data['image_path']
    headline = data['headline']
    
    try:
        result = verify_image_context(image_path, headline)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-full', methods=['POST'])
def analyze_full_endpoint():
    start_time = time.time()
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing required field: text'}), 400
    
    text = data['text']
    
    if 'headline' not in data:
        headline = text
    else:
        headline = data['headline']
    
    has_image = 'image_path' in data
    image_path = data.get('image_path', '')
    
    # Generate hash from inputs
    hash_input = f"{text}{image_path}{headline}"
    cache_key = hashlib.md5(hash_input.encode()).hexdigest()
    
    # Check cache
    if cache_key in cache:
        response_time_ms = (time.time() - start_time) * 1000
        cached_result = cache[cache_key].copy()
        result = {
            'final_label': cached_result.get('final_label', 'Unverified'),
            'final_trust_score': cached_result.get('final_trust_score', 50),
            'reason': cached_result.get('reason', 'Analysis result retrieved from cache.'),
            'cache_hit': True,
            'response_time_ms': round(response_time_ms, 2)
        }
        return jsonify(result)
    
    try:
        text_label, text_confidence, text_explanation = analyze_text(text)
        
        # Image analysis (skip if image_path not provided)
        image_context = None
        image_similarity = None
        if has_image:
            try:
                image_result = verify_image_context(image_path, headline)
                image_context = image_result['image_context']
                image_similarity = image_result['similarity_score']
            except Exception:
                has_image = False
        
        # Aggregation logic
        text_is_fake = text_label == "Fake"
        text_high_confidence = text_confidence >= 0.75
        image_mismatch = image_context == "Mismatch" if has_image else False
        
        # Determine final_label
        if has_image:
            if text_is_fake and text_high_confidence and image_mismatch:
                final_label = "High Risk Fake"
                base_score = 20
                reason = "The text analysis indicates highly unreliable content with strong confidence, and the image does not match the headline, suggesting potential manipulation or misinformation."
            elif (text_is_fake and text_high_confidence) or (text_is_fake and image_mismatch) or (image_mismatch and text_confidence >= 0.55):
                final_label = "Possibly Misleading"
                base_score = 45
                reason = "Multiple signals suggest this content may be unreliable - either the text shows characteristics of misinformation or the image does not align with the headline, but confidence levels vary."
            elif (text_is_fake and not text_high_confidence) or (not text_is_fake and image_mismatch):
                final_label = "Unverified"
                base_score = 60
                reason = "The analysis shows conflicting or uncertain signals. The text and image context do not fully align, and verification through additional sources is recommended before sharing or relying on this content."
            else:
                final_label = "Likely Credible"
                base_score = 80
                reason = "Both the text analysis and image-headline alignment suggest this content appears credible. However, always verify important claims through multiple reliable sources as automated analysis is not infallible."
        else:
            # Text-only analysis
            if text_is_fake and text_high_confidence:
                final_label = "Possibly Misleading"
                base_score = 45
                reason = "The text analysis suggests this content may be unreliable. Image context was not available for verification. Additional verification through reliable sources is recommended."
            elif text_is_fake:
                final_label = "Unverified"
                base_score = 60
                reason = "The text analysis shows some characteristics of unreliable content, but confidence is moderate. Image context was not available. Verification through additional sources is recommended."
            else:
                final_label = "Likely Credible"
                base_score = 75
                reason = "The text analysis suggests this content appears credible. However, image context was not available for verification. Always verify important claims through multiple reliable sources."
        
        # Calculate final_trust_score
        if has_image and image_mismatch:
            similarity_penalty = max(0, (0.20 - image_similarity) * 100)
            final_trust_score = max(0, base_score - similarity_penalty)
        elif has_image:
            confidence_bonus = (text_confidence - 0.5) * 40
            final_trust_score = min(100, base_score + confidence_bonus)
        else:
            confidence_bonus = (text_confidence - 0.5) * 30
            final_trust_score = min(100, base_score + confidence_bonus)
        
        final_trust_score = round(final_trust_score)
        
        response_time_ms = (time.time() - start_time) * 1000
        
        result = {
            'final_label': final_label,
            'final_trust_score': final_trust_score,
            'reason': reason
        }
        
        cache[cache_key] = result.copy()
        
        result['cache_hit'] = False
        result['response_time_ms'] = round(response_time_ms, 2)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()
