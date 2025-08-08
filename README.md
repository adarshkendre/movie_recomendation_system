# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Python and Jupyter Notebook, leveraging the TMDB 5000 movie dataset.

📅 **Last Updated:** August 5, 2025  
👨‍💻 **Author:** Adarsh Kendre

---

## 🚀 Project Overview

This project aims to build a movie recommendation engine that suggests similar movies based on content features like genre, cast, crew, and more. The system processes data from TMDB (The Movie Database) and applies NLP and similarity techniques to recommend relevant movies.

---

## 📂 Repository Structure

```bash
Movie Recommendation Model\
  ├── app.py (main Flask file)
  ├── recommender.ipynb (similar to reccommender.ipynb file used for data cleaning and uderstanding every function)
  ├── recommender.py
  ├── templates/
  │   └── index.html
  ├── static/
  │   └── style.css
  ├── tmdb_5000_movies.csv
  └── tmdb_5000_credits.csv

🔄 Project Flow
1. Dataset & Jupyter Notebook Setup
Loaded the TMDB 5000 Movies and Credits datasets.

Set up the Jupyter Notebook (Recommendation_Sys_project.ipynb) for analysis and modeling.

2. Data Preprocessing
Merged the datasets on movie titles.

Cleaned and transformed key columns: genres, keywords, cast, and crew.

Extracted and structured relevant features for content-based filtering.

📌 Further steps like feature engineering, vectorization (TF-IDF or CountVectorizer), and cosine similarity-based recommendation logic will follow in the upcoming versions.

📊 Dataset Description
Datasets used in this project:

tmdb_5000_movies.csv

Contains details like movie titles, genres, keywords, popularity, and overviews.

tmdb_5000_credits.csv

Includes cast and crew information linked to each movie.

Source: Kaggle - TMDB 5000 Movie Dataset

📌 Future Improvements
Improve recommendations using hybrid filtering (content + collaborative).

Deploy the model via Flask or Streamlit.

Add user interface for real-time movie suggestions.

🛠️ Tech Stack
Python

Jupyter Notebook

Pandas, NumPy

Scikit-learn

NLTK

Matplotlib / Seaborn (for visuals)

🤝 Contribution
Feel free to fork the repo, create issues, or contribute improvements through pull requests. Collaboration is welcome!

📜 License
This project is open source and available under the MIT License.

🌟 Acknowledgments
TMDB for the open-source dataset

Kaggle for hosting the dataset

Python community for excellent open-source libraries
