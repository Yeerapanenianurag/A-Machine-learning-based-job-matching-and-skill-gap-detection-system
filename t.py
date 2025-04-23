import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For loading the saved model

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

# Step 4: Function to predict job description and show matching job titles based on input skills
def predict_job_description(input_skills, vectorizer, skills_matrix, df):
    input_vector = vectorizer.transform([input_skills])  # Transform input skills to vector
    similarity_scores = cosine_similarity(input_vector, skills_matrix)  # Calculate similarity
    best_match_index = similarity_scores.argmax()  # Find the best match (most similar)
    
    # Get the predicted job title and description
    predicted_job_title = df.iloc[best_match_index]["Job Title"]
    predicted_job_description = df.iloc[best_match_index]["Job Description"]
    
    # Get other matching job titles, sort by similarity score, and eliminate duplicates
    top_matches_indices = similarity_scores.argsort()[0][-5:][::-1]  # Get top 5 matching job titles
    matching_job_titles = df.iloc[top_matches_indices]["Job Title"].values
    
    # Remove duplicate job titles
    unique_matching_job_titles = list(set(matching_job_titles))  # Eliminate duplicates
    
    return predicted_job_title, predicted_job_description, unique_matching_job_titles

# Step 5: Example input skills for prediction (you can change this as per your requirement)
input_skills = "AWS, Azure"  # Example input skills for the prediction

# Step 6: Get the predicted job title, job description, and matching job titles
predicted_job_title, predicted_job_description, matching_job_titles = predict_job_description(input_skills, vectorizer, skills_matrix, df)

# Step 7: Print the results
print(f"Predicted Job Title: {predicted_job_title}")
print(f"Predicted Job Description: {predicted_job_description}")
print(f"Matching Job Titles: {', '.join(matching_job_titles)}")
