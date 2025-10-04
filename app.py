import gradio as gr
import pandas as pd
import pickle

# Load precomputed SVD predictions and movies
R_pred_df = pickle.load(open("R_pred_df.pkl", "rb"))
user_item_matrix = pickle.load(open("user_item_matrix.pkl", "rb"))
movies = pd.read_csv("movie.csv")

def recommend_movies(user_id, N=5):
    if user_id not in R_pred_df.index:
        return [f"User {user_id} not found in dataset."]
    user_ratings = R_pred_df.loc[user_id]
    already_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = user_ratings.drop(already_rated)
    top_movies = recommendations.sort_values(ascending=False).head(N).index
    recommended_titles = movies[movies['movieId'].isin(top_movies)]['title'].tolist()
    return recommended_titles

demo = gr.Interface(
    fn=recommend_movies,
    inputs=[gr.Number(label="User ID"), gr.Number(label="Top-N")],
    outputs="text",
    title="Movie Recommender (SVD)",
    description="Enter a user ID and number of recommendations (N)."
)

if __name__ == "__main__":
    demo.launch()
