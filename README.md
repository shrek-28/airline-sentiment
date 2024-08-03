# Airline Sentiment
This project focuses on analyzing the sentiment of tweets related to airlines using natural language processing (NLP) and machine learning. The goal is to classify tweets into positive, negative, or neutral sentiments based on their text content.

# Project Overview
The project involves loading a dataset of airline tweets, preprocessing the text data, and applying various machine learning models to classify the sentiment of the tweets. The models used include Naive Bayes, Logistic Regression, and Support Vector Machines (SVM). The results are evaluated using classification metrics and confusion matrices.

# Dataset
The dataset used in this project, ```airline_tweets.csv```, contains tweets about airlines along with their associated sentiment (positive, negative, or neutral). It also includes features such as ```text```(the content of the tweet) and ```negativereason``` (if applicable).

# Key Steps in the Project
* Data Loading and Exploration:
    * Load the dataset using Pandas.
    * Explore the dataset with methods like head() and value_counts() to understand the distribution of sentiments.
* Data Visualization:
    * Visualize the sentiment distribution using count plots.
    * Analyze the distribution of negative reasons and sentiment across different airlines.
* Data Preprocessing:
    * Extract the relevant features (text and airline_sentiment) for modeling.
    * Split the data into training and testing sets using ```train_test_split```.
* Text Vectorization: Convert text data into numerical format using TF-IDF vectorization.
* Modeling:
    * Train Naive Bayes, Logistic Regression, and SVM models on the TF-IDF features.
    * Evaluate model performance using classification reports and confusion matrices.
* Model Evaluation: Use a custom function to generate classification reports and display confusion matrices for each model.
* Pipeline Implementation: Implement a pipeline combining TF-IDF vectorization and the LinearSVC model for streamlined predictions on new tweets.
* Prediction Examples: Test the pipeline with new tweet examples to predict their sentiment.
