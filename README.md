# Movie_Recommendation_System
Using the MovieLens dataset, which contains user ratings for various  movies. The objective is to develop a Recommendation System that can suggest the  Top-N movies for a given user. This is then uploaded to Hugging face.
Link to huggingface: https://huggingface.co/spaces/SumaiyaTasnim-000/DataSynthis_ML_JobTask


. Introduction
•	Goal: Build a recommendation system using the MovieLens dataset.
•	Implemented methods: Collaborative Filtering (CF) and Matrix Factorization (SVD).
•	Evaluation metrics: Precision@K, Recall@K, and NDCG@K.
________________________________________
2. Data Preprocessing
•	Loaded MovieLens 20M dataset.
•	Checked for missing values and duplicates → nonsignificant ; seems like the dataset is already preprocessed
•	Encoded userId and movieId as categorical indices for sparse matrix creation.
•	Used a filtered subset (10,000 users × 1,000–2,000 movies) for computational efficiency.
________________________________________
3. Implemented Models
a) Collaborative Filtering (CF)
•	Item-based CF using cosine similarity on sparse user–item matrix.
•	Recommendation based on most similar movies to those the user rated.
b) Matrix Factorization (SVD)
•	Decomposed the user–item matrix into latent factors (UΣVᵀ).
•	Retained top 50 factors for prediction.
•	Generated recommendations by reconstructing missing ratings.
________________________________________
4. Evaluation Metrics
We evaluated Precision@10, Recall@10, and NDCG@10 for a sample user (User 4) and averaged results across 100 users.
Sample Results (User 4):
•	Collaborative Filtering
o	Precision@10: 0.10
o	Recall@10: 0.0357
o	NDCG@10: 0.139
•	SVD
o	Precision@10: 0.70
o	Recall@10: 0.25
o	NDCG@10: 0.753
Interpretation:
•	CF achieved low precision and recall, showing that nearest-neighbor similarity alone isn’t sufficient.
•	SVD achieved much higher precision (0.70) and better ranking (NDCG = 0.753), confirming that latent factor models capture hidden user–movie relationships better.
________________________________________
5. Comparison
•	CF: Simple and interpretable, but struggled with sparse data → poor precision/recall.
•	SVD: Significantly better in precision and ranking quality → recommended as the baseline model.
•	Neural CF: Attempted but excluded from final comparison due to evaluation misalignment. (Optional mention: highlight as a potential extension with more training time and alignment work.)
________________________________________
6. Conclusion
•	SVD is the best-performing method in this assignment.
•	It balances accuracy and computational efficiency, with strong results across metrics.
•	CF is useful for intuition but insufficient for large sparse datasets like MovieLens.
•	Neural embeddings can be explored further for improving recall and personalization in future work.

