#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse


# Count number of lines
MSD = open('train_triplets.txt', 'r')
n_lines = 0
for line in MSD:
    n_lines += 1
MSD.close()


# Allocate list for the elements
users = np.array([0] * n_lines)
songs = np.array([0] * n_lines)
counts = np.array([0] * n_lines)


# Load data
MSD = open('train_triplets.txt', 'r')
cur_line = 0
for line in MSD:
    user, song, count = line.split()
    users[cur_line] = hash(user.strip())
    songs[cur_line] = hash(song.strip())
    counts[cur_line] = int(count)
    
    cur_line += 1
MSD.close()

# Compute Unique users
unique_users = np.unique(users)
unique_songs = np.unique(songs)

# Mapping song to column in matrix
song_mapping = {song: ID for (song, ID) in zip(unique_songs, np.arange(len(unique_songs)))}

# Mapping user to row in matrix
user_mapping = {user: ID for (user, ID) in zip(unique_users, np.arange(len(unique_users)))}

# Mapped rows, columns for indexing the matrix later on.
user_row = np.array([user_mapping[user] for user in users])
songs_col = np.array([song_mapping[song] for song in songs])

# Create empty sparse matrix
MSD_csr = scipy.sparse.csr_matrix((counts, (user_row, songs_col)), shape=(len(unique_users), len(unique_songs)))
MSD_bin = MSD_csr.copy()
MSD_bin.data = np.full(len(MSD_csr.data) ,1.0)

# Filter user with at least 20 songs, songs that were listened to at least 200 times
user_counts = MSD_bin.sum(axis=1)
item_counts =  MSD_bin.sum(axis=0)

MSD_csr = MSD_csr[(user_counts >= 20).A1, :]
MSD_csr = MSD_csr[:, (item_counts >= 200).A1]

# Save the matrix.
scipy.sparse.save_npz('MSD.npz', MSD_csr)

