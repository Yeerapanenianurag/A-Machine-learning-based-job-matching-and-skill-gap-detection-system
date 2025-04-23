import os
import csv
from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
import tensorflow as tf
import cv2
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For loading the saved model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from database import *
import random
import os
import tempfile
import MySQLdb
from datetime import datetime
# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
@app.route('/')
def m():
    return render_template('indexb.html')
@app.route('/c')
def m11111():
    return render_template('c.html')
@app.route('/al')
def m112():
    return render_template('alogin.html')
@app.route('/ar')
def m111():
    return render_template('areg.html')
@app.route('/sa')
def m1():
    return render_template('sa.html')
@app.route('/sga')
def m12():
    return render_template('sga.html')
@app.route('/cr')
def career_roadmaps():
    careers = [
    "frontend",
    "backend",
    "devops",
    "full-stack",
    "ai-engineer",
    "data-analyst",
    "ai-data-scientist",
    "android",
    "ios",
    "postgresql",
    "blockchain",
    "qa",
    "software-design-architecture",
    "cyber-security",
    "ux-design",
    "game-developer",
    "technical-writer",
    "mlops",
    "product-manager",
    "engineering-manager",
    "developer-relations",
    "computer-science",
    "react",
    "vue",
    "angular",
    "javascript",
    "node.js",
    "typescript",
    "python",
    "sql",
    "system-design",
    "api-design",
    "aspnet-core",
    "java",
    "cpp",
    "flutter",
    "spring-boot",
    "golang",
    "rust",
    "graphql",
    "software-design-architecture",
    "design-system",
    "react-native",
    "aws",
    "code-review",
    "docker",
    "kubernetes",
    "linux",
    "mongodb",
    "prompt-engineering",
    "terraform",
    "datastructures-and-algorithms",
    "git-github",
    "redis",
    "php"]
    return render_template('cr.html', careers=careers)
@app.route('/uh')
def m15():
    return render_template('ahome.html')
@app.route('/l')
def logout():
    return render_template('index.html')  # Replace with logic for logout if needed



@app.route("/Admin_register", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            return render_template("/index.html", m1="Please fill all fields")

        print(username, email, password)
        status = acc_reg(username, email, password)
        if status == 1:
            return render_template("/index.html")
        else:
            return render_template("/index.html", m1="failed")
    return render_template("/Admin_register.html")  # Render the form on GET
@app.route("/admin_login", methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            return render_template("/alogin.html", m1="Please enter email and password")

        status = acc_loginact(email, password)
        print(status)
        if status == 1:
            session['email'] = email
            return render_template("/ahome.html", m1="success")
        else:
            return render_template("/alogin.html", m1="Login Failed")
    return render_template("/alogin.html")  # Render the login form on GET

        


# Step 1: Load the dataset again (you need the dataset for predictions)
def load_data(file_path):
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']  # Try multiple common encodings
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Data loaded successfully using {encoding} encoding.")
            return df
        except UnicodeDecodeError:
            print(f"Failed to load data with {encoding} encoding.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    print("Failed to load data with all tried encodings.")
    return None

# Step 2: Load the dataset for prediction
df = load_data('new_data.csv')  # Replace with your file path
if df is None:
    exit()  # Exit if loading the data fails

# Step 3: Load the saved model (vectorizer and skills_matrix)
vectorizer = joblib.load('tfidf_vectorizer.joblib')
skills_matrix = joblib.load('skills_matrix.joblib')

# Load saved components
model = joblib.load('RandomForest_model.pkl')      # Load the best saved model
vectorizer = joblib.load('vectorizer.pkl')          # Load the saved vectorizer
encoder = joblib.load('label_encoder.pkl')          # Load the saved label encoder
df = pd.read_csv('jb.csv')                          # Load the dataset

# Function to predict job title
def predict_job_title(skills):
    skills_vectorized = vectorizer.transform([skills])
    predicted_label = model.predict(skills_vectorized)
    job_title = encoder.inverse_transform(predicted_label)
    return job_title[0]

# Function to predict job description and show matching job titles using ML model
def predict_job_description(input_skills, vectorizer, model, encoder, df):
    # Transform input skills to vector
    input_vector = vectorizer.transform([input_skills])
    
    # Predict the job title
    predicted_label = model.predict(input_vector)
    predicted_job_title = encoder.inverse_transform(predicted_label)[0]
    
    # Get prediction probabilities for all classes
    probabilities = model.predict_proba(input_vector)[0]
    
    # Get top 5 probable job titles
    top_indices = np.argsort(probabilities)[-5:][::-1]
    matching_job_titles = encoder.inverse_transform(top_indices)
    
    # Get job description for the predicted title
    job_description = df[df['Job Title'] == predicted_job_title]['Job Description'].values[0]
    
    # Eliminate duplicates from the matching titles
    unique_matching_job_titles = list(set(matching_job_titles))
    
    return predicted_job_title, job_description, unique_matching_job_titles

@app.route("/sa1", methods=['POST', 'GET'])
def sa1():
    if request.method == 'POST':
        input_skills = request.form['skills']
        predicted_job_title, predicted_job_description, matching_job_titles = predict_job_description(
            input_skills, vectorizer, model, encoder, df
        )
        return render_template("/sa.html", 
                               predicted_job_title=predicted_job_title,
                               predicted_job_description=predicted_job_description,
                               matching_job_titles=matching_job_titles)

# Load the CSV file with encoding specified
job_data_path = "new_data.csv"
try:
    df = pd.read_csv(job_data_path, encoding='ISO-8859-1')  # Update encoding if necessary
except UnicodeDecodeError as e:
    raise Exception(f"Error reading CSV file: {e}. Please check the file encoding.")

@app.route('/analyze', methods=['POST'])
def analyze_skills():
    try:
        # Parse the incoming JSON data
        user_data = request.json
        target_job = user_data.get('target_job')
        user_skills = set(user_data.get('skills', '').split(', '))
        
        # Validate inputs
        if not target_job or not user_skills:
            return jsonify({"message": "Please provide both 'target_job' and 'skills'"}), 400
        
        # Find the target job in the dataset
        job = df[df['Job Title'] == target_job]
        if job.empty:
            return jsonify({"message": "Target job not found"}), 404
        
        # Extract required skills
        required_skills = set(job.iloc[0]['Required Skills'].split(', '))
        missing_skills = required_skills - user_skills
        
        # Return the result
        if not missing_skills:
            return jsonify({"message": "You are ready for the job!"})
        else:
            return jsonify({"missing_skills": list(missing_skills)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Clean salary range and extract the upper limit
def clean_salary(salary_range):
    # Replace problematic characters like 'œ' and remove commas
    salary_range = salary_range.replace('œ', '').replace('£', '').replace(',', '').strip()
    salary_parts = salary_range.split('-')
    try:
        upper_limit = int(salary_parts[1].strip()) if len(salary_parts) == 2 else 0
    except ValueError:
        upper_limit = 0
    return upper_limit

# Read jobs and process the CSV data
def read_jobs():
    jobs = []
    with open('new_data.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Fix salary range if it contains problematic characters
            row['Salary Range'] = row['Salary Range'].replace('œ', '£')
            # Clean and process salary range
            row['Upper Limit'] = clean_salary(row['Salary Range'])
            jobs.append(row)
    return sorted(jobs, key=lambda x: x['Upper Limit'], reverse=True)

# Route for home with sorting and pagination
@app.route('/ej')
def home():
    jobs = read_jobs()

    # Pagination parameters
    page = int(request.args.get('page', 1))  # Current page
    per_page = 3  # Jobs per page
    total_jobs = len(jobs)
    start = (page - 1) * per_page
    end = start + per_page

    # Determine next and previous page links
    next_page = page + 1 if end < total_jobs else None
    prev_page = page - 1 if start > 0 else None

    return render_template(
        'ej.html',
        jobs=jobs[start:end],
        next_page=next_page,
        prev_page=prev_page
    )
if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)