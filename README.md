# ImplicitFeedback

This small repository downloads the three famous Recommendation dataset, Netlix, ML 20M and MillionSongDataset and makes them to sparse matrices in the csr format using scipy.sparse. As typical in Recommendation, the rows denote the users and the columns the products. The Script Create_Data has to be started to download and create the data. It was developed on Ubuntu and needs the software unzip to extract the zip files. Moreover, python needs to be available as 'python' and the packages numpy, pandas and scipy sparse need to be available.
