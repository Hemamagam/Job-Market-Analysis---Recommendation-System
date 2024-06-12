from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the model and vectorizer
with open('job_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load job data
csv_file_path = 'D:/AI Course Digicrome/One Python/Nexthike-Project Work/Project 8- Job Analysis/Job-Market-Analysis---Recommendation-System/python/job_data.csv'
data = pd.read_csv(csv_file_path)

# Generate mock historical data
dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
job_postings = [150 + i * 10 for i in range(12)]  # Simulated job postings data
historical_data = pd.DataFrame({'Month': dates, 'Job_Postings': job_postings})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('description')

    if not user_input:
        return render_template('index.html', error="Please provide a job description")

    # Transform user input
    user_vec = vectorizer.transform([user_input])

    # Find nearest neighbors
    distances, indices = model.kneighbors(user_vec, n_neighbors=5)

    # Get recommended jobs
    recommendations = data.iloc[indices[0]]

    return render_template('result.html', recommendations=recommendations)

@app.route('/market-trends')
def market_trends():
    # Plotting job market trends
    plt.figure(figsize=(10, 6))
    plt.plot(historical_data['Month'], historical_data['Job_Postings'], marker='o')
    plt.title('Monthly Job Postings')
    plt.xlabel('Month')
    plt.ylabel('Number of Job Postings')
    plt.grid(True)
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('market_trends.html', plot_url=plot_url)


# List of remote-friendly countries
remote_friendly_countries = ['United States', 'Canada', 'United Kingdom', 'Germany', 'Australia', 'India', 'Singapore', 'Netherlands', 'Israel', 'Ireland', 'Sweden', 'Switzerland', 'Denmark', 'Norway', 'Finland', 'New Zealand', 'Belgium', 'France', 'Spain', 'Italy', 'Japan', 'South Korea']

remote_friendly_countries = [
    'United States', 'Canada', 'United Kingdom', 'Germany', 'Australia', 
    'India', 'Singapore', 'Netherlands', 'Israel', 'Ireland', 'Sweden', 
    'Switzerland', 'Denmark', 'Norway', 'Finland', 'New Zealand', 'Belgium', 
    'France', 'Spain', 'Italy', 'Japan', 'South Korea'
]

@app.route('/remote_work_trends')
def remote_work_trends():
    # Check if 'location' column exists in the data
    if 'location' not in data.columns:
        return "Error: 'location' column is missing in the dataset", 400
    
    # Calculate remote job statistics
    remote_jobs = data[data['category'] == 'Remote']
    non_remote_jobs = data[data['category'] != 'Remote']
    
    total_jobs = len(data)
    remote_jobs_count = len(remote_jobs)
    non_remote_jobs_count = len(non_remote_jobs)
    remote_percentage = (remote_jobs_count / total_jobs * 100) if total_jobs > 0 else 0
    
    # Prepare a report
    report = {
        'total_jobs': total_jobs,
        'remote_jobs': remote_jobs_count,
        'non_remote_jobs': non_remote_jobs_count,
        'remote_percentage': remote_percentage
    }
    
    return render_template('remote_work_trends.html', report=report, remote_friendly_countries=remote_friendly_countries)

@app.route('/predict-trends')
def predict_trends():
    # Use KNN model for prediction
    model = NearestNeighbors(n_neighbors=1)
    model.fit(historical_data[['Job_Postings']])
    
    # Predict future job postings
    future_months = pd.date_range(start='2023-01-01', periods=24, freq='M')
    future_index = [[len(historical_data) + i] for i in range(24)]  # Predict for 2023 and 2024 (12 months each)
    predictions = model.kneighbors(future_index, return_distance=False)
    
    prediction_data = pd.DataFrame({'Month': future_months, 'Predicted_Job_Postings': [p[0] for p in predictions]})
    
    return render_template('predict_trends.html', prediction_data=prediction_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)