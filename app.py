from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Initialize the sentiment analysis pipeline
sentiment_model = pipeline("sentiment-analysis")

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data['text']
    result = sentiment_model(text)[0]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
