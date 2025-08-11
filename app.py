from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load model once when the server starts
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

@app.route('/embedding', methods=['POST'])
def get_embedding():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # Generate embedding locally
    embedding = model.encode(question).tolist()

    return jsonify({"embedding": embedding})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses this PORT
    app.run(host="0.0.0.0", port=port)
