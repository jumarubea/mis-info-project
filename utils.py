import requests
import time
import types
from datetime import datetime


# Function to fetch Facebook posts
def fetch_facebook_posts(user_id, access_token, limit=10, logger=None):
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
def run_inference(post, client, model_endpoint, logger=None):
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