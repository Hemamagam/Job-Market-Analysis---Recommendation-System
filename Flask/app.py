from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Define global variables for model, vectorizer, and data
model = None
vectorizer = None
data = pd.DataFrame()

# Load the model and vectorizer for job recommendation
try:
    with open('D:/AI Course Digicrome/One Python/Nexthike-Project Work/Project 8- Job Analysis/Job-Market-Analysis---Recommendation-System/Flask/job_recommender_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    pass  # Handle this case as per your application's logic

try:
    with open('D:/AI Course Digicrome/One Python/Nexthike-Project Work/Project 8- Job Analysis/Job-Market-Analysis---Recommendation-System/Flask/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    pass  # Handle this case as per your application's logic

# Load job analysis data
try:
    data = pd.read_csv('D:/AI Course Digicrome/One Python/Nexthike-Project Work/Project 8- Job Analysis/Job-Market-Analysis---Recommendation-System/Flask/job_data.csv', parse_dates=['published_date'])
except FileNotFoundError:
    pass  # Handle this case as per your application's logic

@app.route('/')
def home():
    if data.empty:
        return render_template('index.html', error="Data file not found or empty")
    categories = data['category'].unique()
    return render_template('index.html', categories=categories)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('description')
    user_category = request.form.get('category')

    if not user_input:
        return render_template('index.html', error="Please provide a job description", categories=data['category'].unique())

    if model is None or vectorizer is None:
        return render_template('index.html', error="Model or vectorizer not found", categories=data['category'].unique())

    user_vec = vectorizer.transform([user_input])
    distances, indices = model.kneighbors(user_vec, n_neighbors=5)
    recommended_jobs = data.iloc[indices[0]]

    if user_category:
        recommended_jobs = recommended_jobs[recommended_jobs['category'] == user_category]

    if recommended_jobs.empty:
        return render_template('index.html', error="No recommendations found for the selected category", categories=data['category'].unique())

    return render_template('result.html', recommendations=recommended_jobs.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5002)