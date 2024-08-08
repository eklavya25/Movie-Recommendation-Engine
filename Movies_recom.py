from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load the dataset
df = pd.read_excel("movies_info.xlsx")

# Ensure all genres are strings and replace NaNs with empty strings
df['Genres'] = df['Genres'].astype(str).apply(lambda x: x if x != 'nan' else '')

# Replace ', ' with ' ' in the Genres column
df['Genres'] = df['Genres'].apply(lambda x: x.replace(', ', ' '))

# Create a CountVectorizer to convert the genres into a matrix of token counts
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(df['Genres'])

# Calculate the cosine similarity using linear_kernel
cosine_sim = linear_kernel(genre_matrix, genre_matrix)

# Create a mapping from movie title to index
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

def mov_rec(name, n=5):
    if name not in indices:
        return f"Movie '{name}' not found in the dataset."

    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    lst = []
    for i in range(1, n + 1):
        if i >= len(sim_scores):
            break
        movie_index = sim_scores[i][0]
        s = {
            "Movie Title": df['Title'][movie_index],
            "Rating": df['Rating'][movie_index],
            "Genre": df['Genres'][movie_index],
            "Cast": df['Cast'][movie_index]
        }
        lst.append(s)
        
    return lst

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']
    recommendations = mov_rec(movie_name)
    
    if isinstance(recommendations, str):
        return jsonify({'error': recommendations})
    else:
        return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
