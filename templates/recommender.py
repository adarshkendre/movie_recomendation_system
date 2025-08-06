import pandas as pd
import pickle

new_df = pd.read_pickle('new_df.pkl')
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]