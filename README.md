# 🎬 CineMatch — AI Movie Recommendation System

An AI-powered movie recommendation web app using collaborative filtering, content-based filtering (TF-IDF + cosine similarity), and live data from the TMDb API.

## Features
- 🔍 Search movies by title in real-time
- 🎭 Filter by genre with smart ML ranking
- 🤖 Content-based similarity using TF-IDF
- 🌐 Live data from The Movie Database (TMDb)
- 🔒 Secure API key management via environment variables

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/PriyadharshiniC05/Movie-recommendation-.git
cd Movie-recommendation-
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your TMDb API key
Get a free API key at https://www.themoviedb.org/settings/api

```bash
# Mac/Linux
export TMDB_API_KEY=your_key_here

# Windows
set TMDB_API_KEY=your_key_here
```

### 5. Run the app
```bash
python app.py
```
Visit **http://localhost:5000**

## Project Structure
```
├── app.py                  # Flask backend + ML logic
├── templates/
│   ├── index.html          # Home page with genre selector
│   └── recommendations.html # Results page
├── requirements.txt
├── .env.example            # Template for API key (safe to commit)
├── .gitignore              # Keeps .env out of GitHub
└── README.md
```

## Tech Stack
- **Backend**: Python, Flask
- **ML**: scikit-learn (TF-IDF, cosine similarity, KNN)
- **Data**: TMDb REST API
- **Frontend**: Jinja2 templates, vanilla CSS
