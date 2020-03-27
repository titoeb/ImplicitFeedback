#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scipy.sparse
import os


# This is where the indivual files are
# The original netlix dataset comes in one file per movie, containing the indexes,
# ratings and timestamps of the users
rel_directory = 'training_set/'


# Count the total number of interactions
n_interactions = 0
for file_name in os.listdir(rel_directory):
    with open(os.path.join(rel_directory[:-1], file_name)) as cur_file:
        cur_movie = int(cur_file.readline().strip()[:-1])
        for line in cur_file:
            user, rating = line.split(',')[:2]
            if int(rating) >= 4:
                n_interactions += 1

# pre-allocate the vectors.
users = np.array([0] * n_interactions)
movies = np.array([0] * n_interactions)

# Read in the acutal data.
n_interactions = 0
for file_name in os.listdir(rel_directory):
    with open(os.path.join(rel_directory[:-1], file_name)) as cur_file:
        cur_movie = int(cur_file.readline().strip()[:-1])
        for line in cur_file:
            user, rating = line.split(',')[:2]
            if int(rating) >= 4:
                users[n_interactions] = int(user)
                movies[n_interactions] = cur_movie
                n_interactions += 1

# Movies have ids from 1, ..., n
# Make is 0, ..., n-1 to use for indexing.
movies = movies - 1

# Create mappings to map movie IDs to unique columns
unique_movies = np.unique(movies)
movies_mapping = {movie:mapping for (movie, mapping) in zip(unique_movies, np.arange(len(unique_movies)))}

# Do the acutal mapping.
movies_transformed = np.array([movies_mapping[movie] for movie in movies])

# Map unique users to rows.
unique_users = np.unique(users)
users_mapping = {user:mapping for (user, mapping) in zip(unique_users, np.arange(len(unique_users)))}
users_transformed = np.array([users_mapping[user] for user in users])

# Create the acutal matrix.
netflix_matrix = scipy.sparse.csr_matrix((np.full(n_interactions, 1), (users_transformed, movies_transformed)), shape=(len(unique_users), len(unique_movies)))

# Save the matrix.
scipy.sparse.save_npz('Netflix.npz', netflix_matrix)
