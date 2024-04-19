import numpy as np


def baseline(train_data, test_data):
    
    # Calculate average rating for each user and each item
    user_means = train_data.groupby('userID')['rating'].mean()
    item_means = train_data.groupby('songID')['rating'].mean()

    # Predict ratings for validation data using average ratings
    test_data['user_mean_rating'] = test_data['userID'].map(user_means)
    test_data['item_mean_rating'] = test_data['songID'].map(item_means)

    # Fill missing user or item means with global mean
    global_mean = train_data['rating'].mean()
    test_data['user_mean_rating'].fillna(global_mean, inplace=True)
    test_data['item_mean_rating'].fillna(global_mean, inplace=True)

    # Calculate predicted ratings using average of user and item means
    test_data['predicted_rating'] = (test_data['user_mean_rating'] + test_data['item_mean_rating']) / 2

    return np.array(test_data['predicted_rating'].tolist()), np.array(test_data['rating'].tolist())


