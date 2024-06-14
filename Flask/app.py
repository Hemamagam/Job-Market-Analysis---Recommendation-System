from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the model and vectorizer
with open('job_recommender_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load job data
csv_file_path = 'D:/AI Course Digicrome/One Python/Nexthike-Project Work/Project 8- Job Analysis/Job-Market-Analysis---Recommendation-System/job_data.csv'
data = pd.read_csv(csv_file_path)

# Generate mock historical data
dates = pd.date_range(start='2023-01-01', periods=12, freq='MS')  # Use 'MS' for month start frequency
job_postings = [150 + i*10 for i in range(12)]  # Simulated job postings data
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

    recommendations_list = recommendations.to_dict('records')
    
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

@app.route('/remote-work-trends')
def remote_work_trends():
    remote_jobs = data[data['Job Type'] == 'Remote']
    non_remote_jobs = data[data['Job Type'] != 'Remote']
    
    report = {
        'total_jobs': len(data),
        'remote_jobs': len(remote_jobs),
        'non_remote_jobs': len(non_remote_jobs),
        'remote_percentage': len(remote_jobs) / len(data) * 100
    }
    
    return render_template('remote_work_trends.html', report=report)

@app.route('/predict-trends')
def predict_trends():
    # Simple linear regression model for prediction
    model = LinearRegression()
    X = historical_data.index.values.reshape(-1, 1)
    y = historical_data['Job_Postings'].values

    model.fit(X, y)
    future_index = np.array(range(len(historical_data), len(historical_data) + 12)).reshape(-1, 1)
    predictions = model.predict(future_index)

    future_months = pd.date_range(start=historical_data['Month'].iloc[-1], periods=13, freq='MS')[1:]

    prediction_data = pd.DataFrame({'Month': future_months, 'Predicted_Job_Postings': predictions})

    return render_template('predict_trends.html', prediction_data=prediction_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)