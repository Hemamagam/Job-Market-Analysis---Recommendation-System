from flask import Flask, request, jsonify, render_template
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = Flask(__name__)

# Load the dataset
csv_file_path = r'D:\AI Course Digicrome\One Python\Nexthike-Project Work\Project 8- Job Analysis\Job-Market-Analysis---Recommendation-System\Flask\job_data.csv'
df = pd.read_csv(csv_file_path)

# Preprocess the data for the recommendation engine
df['text'] = df['title'] + " " + df['keywords']
df['text'] = df['text'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Task 5: Job Recommendation Engine
@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.json.get('title')
    idx = indices.get(title)
    if idx is None:
        return jsonify({'error': 'Job title not found'}), 404
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    job_indices = [i[0] for i in sim_scores]
    recommendations = df['title'].iloc[job_indices].tolist()
    
    return jsonify(recommendations)

# Task 6: Track Job Market Dynamics
@app.route('/job-market-trends', methods=['GET'])
def job_market_trends():
    monthly_trends = df.groupby('year_month').size().reset_index(name='count')
    fig = px.line(monthly_trends, x='year_month', y='count', title='Job Market Trends Over Time')
    graph_json = fig.to_json()
    return jsonify(graph_json)

# Task 7: Investigate Remote Work Trends
@app.route('/remote-work-trends', methods=['GET'])
def remote_work_trends():
    df['is_remote'] = df['keywords'].str.contains('remote', case=False, na=False)
    remote_trends = df[df['is_remote']].groupby('year_month').size().reset_index(name='count')
    fig = px.line(remote_trends, x='year_month', y='count', title='Remote Work Trends Over Time')
    graph_json = fig.to_json()
    return jsonify(graph_json)

# Task 8: Predict Future Job Market Trends
@app.route('/job-market-forecast', methods=['GET'])
def job_market_forecast():
    df['year_month'] = pd.to_datetime(df['year_month'], format='%Y-%m')
    time_series = df.groupby('year_month').size()
    model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    forecast = results.get_forecast(steps=12)
    pred_ci = forecast.conf_int()

    fig = px.line(time_series, title='Job Market Forecast')
    fig.add_scatter(x=forecast.predicted_mean.index, y=forecast.predicted_mean, mode='lines', name='Forecast')
    fig.add_scatter(x=pred_ci.index, y=pred_ci.iloc[:, 0], mode='lines', fill='tonexty', name='Lower CI')
    fig.add_scatter(x=pred_ci.index, y=pred_ci.iloc[:, 1], mode='lines', fill='tonexty', name='Upper CI')
    graph_json = fig.to_json()
    return jsonify(graph_json)

if __name__ == '__main__':
    app.run(debug=True)