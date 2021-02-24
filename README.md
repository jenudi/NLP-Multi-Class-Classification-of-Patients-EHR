The aim of the project is to analyze the difference between 2 known Natural Language Processing (NLP) models in regards to analyzing medical textual data. The 2 NLP models are: Word2vec and TF-IDF. The data for the project are over 3,000 SOAP notes (format of medical visit documentation) and from visits to medical professionals. For each one of the 2 NLP models there are 2 tasks, in the fields of unsupervised learning and supervised learning:
1. Unsupervised learning: Cluster the SOAP notes to clusters using K-means.
2. Supervised learning: Classify each SOAP note to the correct diagnosis. In order to classify to the Word2vec model a RNN (Recurrent Neural Network) model is added and to the TF-IDF model a Random forest model is added.

A database in MongoDB stores the SOAP notes and final K-means clusters for each model. For each of the final 4 models (Word2vec & K-means, TF-IDF & K-means, Word2vec & RNN, TF-IDF & Random forest there is an REST API developed with Flask. The API allows to send a request with an unknown SOAP note (or a similar medical textual description) and to get one of the following:
1.	Most common labels and closest sentences in the cluster assigned to the SOAP note according to the Word2vec & K-means model or the TF-IDF & K-means model.
2.	Predicted diagnosis for the condition described in the SOAP note according to the Word2vec & RNN model, TF-IDF & Random forest model.

Technologies used in the project: Python, PyTorch, Flask, MongoDB.
