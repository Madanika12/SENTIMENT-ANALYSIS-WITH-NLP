# SENTIMENT-ANALYSIS-WITH-NLP
COMPANY: CODTECH IT SOLUTIONS
*NAME*: MADANIKA PITTAM
*INTERN ID*: CT04DG1969
*DOMAIN*: MACHINE LEARNING
*DURATION*: 4 WEEKS
*MENTOR*: NEELA SANTOSH
 Overview
This project aims to perform sentiment analysis on customer reviews using Natural Language Processing (NLP) techniques and machine learning. Specifically, it focuses on classifying Amazon product reviews as either positive or negative based on the textual content of each review. This is achieved using a Logistic Regression classifier, trained on TF-IDF-transformed review texts.

 Objective
The goal is to build a model that understands textual sentiment and predicts whether a given customer review expresses satisfaction (positive sentiment) or dissatisfaction (negative sentiment). This is useful in real-world scenarios where companies want to automatically process customer feedback at scale.

Dataset Description
The dataset used for this task is the Amazon Cells Labelled Sentences dataset, which is publicly available on Kaggle. It contains 1,000 labeled sentences extracted from Amazon product reviews. Each review is labeled with:

1 for positive sentiment

0 for negative sentiment

There are no missing values or duplicates in the dataset, which makes it ideal for beginner-level sentiment analysis tasks.

 Methodology
1. Loading and Inspecting Data
The raw dataset is a .txt file with two columns: one for the review sentence and one for the sentiment label. It was loaded into a pandas DataFrame and column names were assigned: review and sentiment.

2. Text Preprocessing
To prepare the text data for machine learning, a cleaning function was applied to:

Convert all text to lowercase

Remove punctuation and special characters using regular expressions

Remove extra whitespace

This step ensures the text is standardized and ready for feature extraction.

3. TF-IDF Vectorization
The clean reviews were transformed into numeric feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency). This helps highlight important words in the reviews while reducing the influence of commonly used words like “the” or “and”. The TfidfVectorizer from Scikit-learn was used to perform this transformation.

4. Train-Test Split
The data was split into a training set (70%) and a test set (30%) to evaluate the model’s performance on unseen data. This ensures the model is not overfitting and generalizes well to new inputs.

5. Model Training: Logistic Regression
A Logistic Regression model was chosen for this classification task. It is a simple yet effective algorithm for binary classification problems like sentiment analysis. The model was trained on the TF-IDF-transformed training data.

6. Evaluation
The model's performance was evaluated on the test set using:

Accuracy Score: Percentage of correctly classified reviews

Confusion Matrix: Shows counts of true positive, true negative, false positive, and false negative predictions

Classification Report: Includes precision, recall, and F1-score for both classes

The final model achieved high accuracy, demonstrating that even with a simple pipeline, sentiment classification can be effectively done with TF-IDF and logistic regression.

7. Sample Prediction
To demonstrate the model’s real-world application, a sample review text ("The product was amazing and I loved it!") was passed through the same preprocessing and vectorization pipeline. The model correctly predicted this as a positive review.
 Key Takeaways
TF-IDF is a powerful technique to convert text into a usable format for machine learning.

Logistic Regression, despite its simplicity, can be highly effective for binary text classification.

Sentiment analysis can help businesses automate the process of understanding customer feedback.
Output:
<img width="696" height="285" alt="Image" src="https://github.com/user-attachments/assets/3ce72bf1-bb22-4889-b683-59a00c9d9577" />




