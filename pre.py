import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib  # For saving and loading the model

# Step 1: Load the dataset with encoding handling
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

# Step 2: Load the dataset
df = load_data('new_data.csv')  # Replace with your file path
if df is None:
    exit()  # Exit if loading the data fails

# Step 3: Vectorize the 'Required Skills' column using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
skills_matrix = vectorizer.fit_transform(df['Required Skills'])

# Step 4: Save the trained model (vectorizer and skills_matrix)
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')  # Save vectorizer
joblib.dump(skills_matrix, 'skills_matrix.joblib')  # Save skills matrix

print("Model and components saved successfully!")
