# Recommendation System - Collaborative Filtering
# Curtis Raympi

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample user-item rating matrix
data = {
    'item1': [5, 4, 1, 0],
    'item2': [3, 0, 0, 5],
    'item3': [4, 0, 0, 4],
    'item4': [0, 2, 4, 5]
}

user_ids = ['user1', 'user2', 'user3', 'user4']
ratings = pd.DataFrame(data, index=user_ids)

# Compute similarity matrix
similarity = cosine_similarity(ratings.fillna(0))
similarity_df = pd.DataFrame(similarity, index=user_ids, columns=user_ids)

def recommend(user, ratings, similarity_df, k=2):
    user_similarities = similarity_df[user].drop(user)
    top_users = user_similarities.sort_values(ascending=False).index[:k]

    recommended_items = pd.Series(dtype=float)

    for other_user in top_users:
        other_ratings = ratings.loc[other_user]
        recommended_items = recommended_items.add(other_ratings, fill_value=0)

    recommended_items = recommended_items / k
    user_ratings = ratings.loc[user]
    recommendations = recommended_items[user_ratings == 0].sort_values(ascending=False)

    return recommendations.index.tolist()

if __name__ == "__main__":
    user_to_recommend = 'user1'
    recs = recommend(user_to_recommend, ratings, similarity_df)
    print(f"Recommendations for {user_to_recommend}: {recs}")
