from flask import Flask, render_template, request
from recommender import recommend, new_df

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    movie_list = sorted(new_df['title'].tolist())
    selected_movie = ""
    if request.method == 'POST':
        selected_movie = request.form['movie']
        recommendations = recommend(selected_movie)
    return render_template('index.html', recommendations=recommendations, movie_list=movie_list, selected_movie=selected_movie)

if __name__ == '__main__':
    app.run(debug=True)