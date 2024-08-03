# Airline Sentiment Analysis 
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

# Dependencies
To run this project, you will need the following Python libraries:
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
Install the required libraries using pip:
```
pip install numpy pandas matplotlib seaborn scikit-learn
```

# Running the Project
* Clone the repository:
```
git clone https://github.com/yourusername/airline-tweet-sentiment-analysis.git
cd airline-tweet-sentiment-analysis
```
* Launch Jupyter Notebook:
```jupyter notebook```
* Open the [Airline_Tweet_Sentiment_Analysis.ipynb](https://github.com/shrek-28/airline-sentiment/blob/main/Airline%20Sentiment%20Prediction%20from%20Classification%20-%20Supervised%20NLP.py) notebook and run the cells sequentially to execute the analysis.

# Results and Insights
The project successfully classifies airline-related tweets into positive, negative, or neutral sentiments using various machine learning models. The LinearSVC model, when used in a pipeline with TF-IDF vectorization, provides accurate predictions for new tweets.

# Future Work
* Experiment with more advanced NLP techniques like word embeddings (e.g., Word2Vec, GloVe) for improved sentiment analysis.
* Explore deep learning models like LSTMs or BERT for better text classification performance.
* Deploy the sentiment analysis model in a web application for real-time tweet classification.
