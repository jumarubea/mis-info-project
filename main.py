import requests
import google.genai as genai
from google.genai import types
from flask import Flask, jsonify, request

from flask_cors import CORS

import json
import os
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration (replace with your actual credentials)
FB_ACCESS_TOKEN = os.getenv('FB_ACCESS_TOKEN', '')
FB_USER_ID = os.getenv('FB_USER_ID', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
PROJECT_ID = "1066946535135"
LOCATION = "us-central1"
MODEL_ENDPOINT = "projects/1066946535135/locations/us-central1/endpoints/1053240879944302592"


# Function to fetch Facebook posts
def fetch_facebook_posts(user_id, access_token, limit=10):
    """
    Fetch recent posts from the user's Facebook account.
    
    Args:
        user_id (str): Facebook user ID
        access_token (str): Facebook access token
        limit (int): Number of posts to fetch
    
    Returns:
        list: List of post messages
    """
    url = f"https://graph.facebook.com/v20.0/{user_id}/posts"
    params = {
        'access_token': access_token,
        'fields': 'message,created_time',
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Facebook API response: {data}")  # Debug: Log the full response
        
        if 'data' not in data:
            logger.error("No posts found in response")
            return []
        
        posts = [post['message'] for post in data['data'] if 'message' in post]
        logger.info(f"Fetched {len(posts)} posts from Facebook")
        
        # Debug: Log the posts
        for i, post in enumerate(posts):
            logger.info(f"Post {i+1}: {post[:100]}...")
        
        return posts
    
    except requests.RequestException as e:
        logger.error(f"Error fetching Facebook posts: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response content: {e.response.text}")
        return []

# Function to run inference on a single post
def run_inference(post, client, model_endpoint):
    """
    Run inference on a single post using the fine-tuned Gemini model.
    
    Args:
        post (str): Text of the Facebook post
        client: Initialized Google AI Studio client
        model_endpoint (str): Endpoint of the fine-tuned model
    
    Returns:
        dict: Inference result with post and prediction
    """
    start_time = time.time()  # Track detection time
    
    try:
        # Define system instruction and content
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=post)]
            )
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.5,
            top_p=1,
            seed=0,
            max_output_tokens=65535,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ],
            system_instruction=[types.Part.from_text(text="Classify text as Trustworthy or Misinformation")],
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        
        # Run inference
        response = ""
        for chunk in client.models.generate_content_stream(
            model=model_endpoint,
            contents=contents,
            config=generate_content_config
        ):
            response += chunk.text
        
        detection_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"Processed post: {post[:50]}... | Prediction: {response.strip()} | Time: {detection_time:.2f}ms")
        
        return {
            'post': post,
            'prediction': response.strip(),
            'timestamp': datetime.now().isoformat(),
            'detectionTime': round(detection_time, 2)  # Add detection time
        }
    
    except Exception as e:
        detection_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.error(f"Error running inference on post: {e}")
        return {
            'post': post,
            'prediction': f"Error: {str(e)}",
            'timestamp': datetime.now().isoformat(),
            'detectionTime': round(detection_time, 2)
        }

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
    posts = fetch_facebook_posts(FB_USER_ID, FB_ACCESS_TOKEN, limit=10)
    
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
            },
            {
                'post': 'Mabadiliko ya ratiba ya uchaguzi.',
                'prediction': 'Misinformation',
                'timestamp': datetime.now().isoformat(),
                'detectionTime': 203.2
            }
        ]
        return jsonify({"results": sample_results}), 200

    logger.info(f"Retrieved {len(posts)} posts for analysis")
    
    # Run inferences
    results = []
    for i, post in enumerate(posts):
        logger.info(f"Processing post {i+1}/{len(posts)}")
        result = run_inference(post, client, MODEL_ENDPOINT)
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