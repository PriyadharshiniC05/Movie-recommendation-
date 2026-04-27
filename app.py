import os
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# ✅ Load API key from environment variable (never hardcode!)
API_KEY = os.getenv('TMDB_API_KEY', '')
BASE_URL = 'https://api.themoviedb.org/3'

GENRE_MAP = {
    28: 'Action', 12: 'Adventure', 16: 'Animation',
    35: 'Comedy', 80: 'Crime', 99: 'Documentary',
    18: 'Drama', 10751: 'Family', 14: 'Fantasy',
    36: 'History', 27: 'Horror', 10402: 'Music',
    9648: 'Mystery', 10749: 'Romance', 878: 'Sci-Fi',
    53: 'Thriller', 10752: 'War', 37: 'Western'
}

GENRE_NAME_TO_ID = {v: k for k, v in GENRE_MAP.items()}


def get_popular_movies():
    """Fetch popular movies from TMDb API with error handling."""
    try:
        url = f"{BASE_URL}/movie/popular?api_key={API_KEY}&language=en-US&page=1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json().get('results', [])
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return []


def search_movies(query):
    """Search movies by title."""
    try:
        url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={query}&language=en-US"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json().get('results', [])
    except requests.exceptions.RequestException as e:
        print(f"Search Error: {e}")
        return []


def filter_movies_by_genre(movies, preferred_genre_names):
    """Filter movies by genre names."""
    preferred_ids = [GENRE_NAME_TO_ID.get(g) for g in preferred_genre_names if g in GENRE_NAME_TO_ID]
    filtered = []
    for movie in movies:
        movie_genre_ids = movie.get('genre_ids', [])
        if any(gid in preferred_ids for gid in movie_genre_ids):
            movie['genre_names'] = [GENRE_MAP.get(gid, '') for gid in movie_genre_ids]
            filtered.append(movie)
    return filtered


def get_content_similarity(movies):
    """Use TF-IDF + cosine similarity on movie overviews."""
    if len(movies) < 2:
        return movies
    overviews = [m.get('overview', '') or '' for m in movies]
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(overviews)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # Sort movies by their average similarity score
        avg_sim = cosine_sim.mean(axis=1)
        sorted_indices = np.argsort(avg_sim)[::-1]
        return [movies[i] for i in sorted_indices]
    except Exception:
        return movies


def format_movie(movie):
    """Add poster URL and genre names to a movie dict."""
    poster = movie.get('poster_path')
    movie['poster_url'] = f"https://image.tmdb.org/t/p/w342{poster}" if poster else None
    if 'genre_names' not in movie:
        movie['genre_names'] = [GENRE_MAP.get(gid, '') for gid in movie.get('genre_ids', [])]
    return movie


@app.route('/')
def home():
    return render_template('index.html', genres=list(GENRE_NAME_TO_ID.keys()))


@app.route('/recommend', methods=['POST'])
def recommend():
    selected_genres = request.form.getlist('genres')
    search_query = request.form.get('search', '').strip()

    if search_query:
        movies = search_movies(search_query)
    else:
        movies = get_popular_movies()

    if selected_genres:
        movies = filter_movies_by_genre(movies, selected_genres)
    else:
        for m in movies:
            m['genre_names'] = [GENRE_MAP.get(gid, '') for gid in m.get('genre_ids', [])]

    # Apply content-based similarity ranking
    movies = get_content_similarity(movies)
    movies = [format_movie(m) for m in movies[:12]]  # Top 12 results

    return render_template('recommendations.html',
                           movies=movies,
                           selected_genres=selected_genres,
                           search_query=search_query)


if __name__ == '__main__':
    if not API_KEY:
        print("⚠️  Warning: TMDB_API_KEY not set. Set it with: export TMDB_API_KEY=your_key")
    app.run(debug=True)
