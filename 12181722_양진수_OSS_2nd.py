import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


file_path = 'ratings.dat'
ratings = pd.read_csv(file_path, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')


movie_id_map = {movie_id: idx for idx, movie_id in enumerate(ratings['movie_id'].unique())}
ratings['movie_idx'] = ratings['movie_id'].map(movie_id_map)


num_users = ratings['user_id'].nunique()
num_movies = len(movie_id_map)
user_item_matrix = np.zeros((num_users, num_movies))


for row in ratings.itertuples():
    user_item_matrix[row.user_id - 1, row.movie_idx] = row.rating


kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(user_item_matrix)

clustered_users = [[] for _ in range(3)]
for user_id, cluster_id in enumerate(clusters):
    clustered_users[cluster_id].append(user_id)


    
def average_rating(matrix):
    return np.nanmean(matrix, axis=0)

def additive_utilitarian(matrix):
    return np.nansum(matrix, axis=0)

def simple_count(matrix):
    return np.sum(~np.isnan(matrix), axis=0)

def approval_voting(matrix, threshold=4):
    return np.sum(matrix >= threshold, axis=0)

def borda_count(matrix):
    ranks = matrix.argsort().argsort()
    return np.sum(ranks, axis=0)


recommendations = {}

for cluster_id, user_ids in enumerate(clustered_users):
    cluster_matrix = user_item_matrix[user_ids, :]
    
    recommendations[cluster_id] = {
        'average_rating': average_rating(cluster_matrix).argsort()[::-1][:10],
        'additive_utilitarian': additive_utilitarian(cluster_matrix).argsort()[::-1][:10],
        'simple_count': simple_count(cluster_matrix).argsort()[::-1][:10],
        'approval_voting': approval_voting(cluster_matrix).argsort()[::-1][:10],
        'borda_count': borda_count(cluster_matrix).argsort()[::-1][:10],
        
    }
    
for cluster_id, recs in recommendations.items():
    print(f"\n클러스터 {cluster_id}:")
    for algo, movies in recs.items():
        print(f"  {algo}: {movies}")
