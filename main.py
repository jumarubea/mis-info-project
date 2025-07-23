import google.genai as genai
from flask import Flask, jsonify
from flask_cors import CORS

import json
import os
import logging
from datetime import datetime
from utils import fetch_facebook_posts, run_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load environment variables
FB_ACCESS_TOKEN = os.getenv('FB_ACCESS_TOKEN')
FB_USER_ID = os.getenv('FB_USER_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')
MODEL_ENDPOINT = os.getenv('MODEL_ENDPOINT')


# Main function
@app.route('/analyze', methods=['GET'])
def analyze_posts():
    """
    Fetch Facebook posts and run inferences using the fine-tuned Gemini model.
    """
    logger.info("Starting analysis request...")
    
    # Initialize client
    try:
        client = genai.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION
        )
        logger.info("Vertex AI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI client: {e}")
        return jsonify({"error": f"Failed to initialize AI client: {str(e)}"}), 500
    
    # Fetch posts
    posts = fetch_facebook_posts(FB_USER_ID, FB_ACCESS_TOKEN, limit=10, logger=logger)
    
    # Debug: Always log the number of posts fetched
    logger.info(f"Posts fetched: {len(posts)}")
    
    if not posts:
        logger.warning("No posts retrieved. Using sample data for testing.")
        # Return sample data for testing when no real posts are available
        sample_results = [
            {
                'post': 'Uchaguzi mwezi wa kumi.',
                'prediction': 'Trustworthy',
                'timestamp': datetime.now().isoformat(),
                'detectionTime': 150.5
            }
        ]
        return jsonify({"results": sample_results}), 200

    logger.info(f"Retrieved {len(posts)} posts for analysis")
    
    # Run inferences
    results = []
    for i, post in enumerate(posts):
        logger.info(f"Processing post {i+1}/{len(posts)}")
        result = run_inference(post, client, MODEL_ENDPOINT, logger=logger)
        results.append(result)
    
    # Save results to JSONL
    output_file = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        logger.info(f"Inference results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to file: {e}")
    
    # Print summary
    logger.info("Analysis complete. Results summary:")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}: {result['post'][:50]}... -> {result['prediction']}")
    
    return jsonify({"results": results}), 200



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)