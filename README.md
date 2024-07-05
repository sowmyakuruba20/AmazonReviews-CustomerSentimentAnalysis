# Amazon Reviews - Customer Sentiment Analysis

This repository contains a project that performs sentiment analysis on Amazon customer reviews. The analysis aims to determine the overall sentiment of the reviews and extract meaningful insights from the data.

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Introduction

Customer sentiment analysis is a powerful tool for understanding the opinions and feedback of customers. This project focuses on analyzing Amazon product reviews to classify the sentiment as positive, negative, or neutral.

## Data Collection

The data for this project is collected from the [Amazon Customer Reviews dataset](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews). The dataset includes the following columns:
- `rating`: The rating given by the customer.
- `date`: The date of the review.
- `variation`: The product variation.
- `verified_reviews`: The review text.
- `feedback`: Feedback score.

## Data Preprocessing
Data preprocessing steps include cleaning the review text, handling missing values, and encoding categorical variables.

## Exploratory Data Analysis
In this section, we perform exploratory data analysis (EDA) to understand the distribution of the data and extract key insights.

## Modeling
We use various machine learning models to classify the sentiment of the reviews. The models include:

* Logistic Regression
* Random Forest
* Support Vector Machine
* Neural Networks
* Decision Tree
* Naive Bayes
* XGBoost

## Results
The performance of the models is evaluated using metrics such as accuracy, precision, recall, and F1-score.
Since the dataset is unbalanced with negative sentiment as the minority class the model is biased toward the positve predictions. 
I have used balanced class method to mitigate this issue.
Accuracy is not a good metrics when dealing with the imbalance class, AUC-ROC curve is a good metric to look at.
Higher the AUC-ROC score better the model is to distingush between the classes correctly

## Conclusion
This project demonstrates the process of performing sentiment analysis on Amazon customer reviews. The insights obtained can help businesses improve their products and services based on customer feedback.

## Requirements
The project requires the following Python packages:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
  
You can install the required packages using:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
## Usage
To run the project, execute the Jupyter Notebook Amazon_Review_Sentiment_Analysis.ipynb.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

