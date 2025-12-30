from flask import Flask, request, jsonify
from models.text_model import analyze_text
from models.image_model import analyze_image

app = Flask(__name__)

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
    
    image_path = data['image_path']
    
    try:
        label, confidence, explanation = analyze_image(image_path)
        return jsonify({
            'label': label,
            'confidence': round(confidence, 2),
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run()
