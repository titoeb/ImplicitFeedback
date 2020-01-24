#!/bin/bash

# Static Links to the Data source. Change if not available any longer.
NETFLIX_LINK=https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz
ML20_LINK=http://files.grouplens.org/datasets/movielens/ml-20m.zip
MSD_LINK=http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip


# Create Netflix dataset

# Download data
wget $NETFLIX_LINK

# Unpack it
tar -zxvf nf_prize_dataset.tar.gz
tar -xvf download/training_set.tar

# Make data to sparse matrix
python Netflix.py

# Clean-up
rm -r download
rm -r training_set
rm nf_prize_dataset.tar.gz

# Create ML20 dataset
# Download data
wget $ML20_LINK

# Unpack it
unzip ml-20m.zip

# Make data to sparse matrix
python Movielens.py

# Clean-up
rm ml-20m.zip 
rm -r ml-20m

# Create Million Song Dataset
# Download data
wget $MSD_LINK

# Unpack it
unzip train_triplets.txt.zip

# Make data to sparse matrix
python MillionSongDataset.py 

# Clean-up
rm train_triplets.txt
rm train_triplets.txt.zip

exit 0
