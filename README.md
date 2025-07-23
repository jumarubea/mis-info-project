# Mis-Information Detector

An AI-powered system designed to scan and analyze social media posts from Tanzania, classifying each post as either **Trustworthy** or **Misinformation**, helping to prevent the spread of false information.

---

## Features

- **Automated Facebook Post Analysis**: Fetches recent posts from a Facebook user account.
- **AI-Powered Classification**: Uses a fine-tuned Gemini model on Google Vertex AI to classify posts.
- **Real-Time Dashboard**: Interactive web UI built with React and Tailwind CSS for visualizing results.
- **Metrics & Insights**: Displays total posts, trustworthy/misinformation counts, and detection times.
- **API-First**: RESTful Flask backend with CORS support.
- **Logging & Debugging**: Detailed logs for API calls, inference, and errors.
- **Sample Data Fallback**: Returns sample results if Facebook credentials are missing or no posts are found.

---

## Project Structure

```
.
├── main.py             # Flask backend API and AI inference logic
├── ui.html             # React-based dashboard UI
├── pyproject.toml      # Python project metadata and dependencies
├── uv.lock             # Python dependency lock file
├── .python-version     # Python version specifier
├── .gitignore          # Git ignore rules
└── README.md           # This documentation
```

---

## Quickstart

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-directory>
```

### 2. Set Up Python Environment

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv init
uv sync
```

Or manually with `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
# Or if using PEP 621/pyproject.toml:
pip install .
```

### 4. Set Environment Variables

```bash
export FB_ACCESS_TOKEN="your_facebook_access_token"
export FB_USER_ID="your_facebook_user_id"
export GOOGLE_API_KEY="your_google_api_key"
```

On Windows (CMD):

```cmd
set FB_ACCESS_TOKEN=your_facebook_access_token
set FB_USER_ID=your_facebook_user_id
set GOOGLE_API_KEY=your_google_api_key
```

### 5. Run the Backend API

```bash
uv run python main.py
```

The Flask server will start at: [http://localhost:5000](http://localhost:5000)

### 6. Launch the Dashboard

Open `ui.html` directly in your browser.
Click **"Analyze Facebook Posts"** to fetch and analyze posts.

---

## ⚙️ Configuration

### Facebook API

Requires a valid **Facebook Graph API** access token and user ID.
Set them as environment variables or edit directly in `main.py`.

### Google Vertex AI

Requires access to a fine-tuned **Gemini Flash** model endpoint and a Google API key.

### Model Description

The model is a **fine-tuned Gemini Flash** retrained on Swahili-language social media posts labeled as `Trusted` or `Misinformation`. It is deployed to **Google Vertex AI** for scalable inference.
