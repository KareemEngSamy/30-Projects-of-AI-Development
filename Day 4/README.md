# 30 Projects in 30 Days of AI Development

This repository documents a challenge of building **30 AI/ML projects in 30 days**.  
Each project is small, focused, an# Spam Email Detector 📧  
***A machine learning project that detects whether an email is **Spam** or **Not Spam**.  
It uses **Logistic Regression** from Scikit-learn with feature scaling, cross-validation, and performance visualization.***  



## Features  
- Classifies emails into **Spam** and **Not Spam**.  
- Uses **Logistic Regression** with `StandardScaler` for stable training.  
- Evaluates model using **accuracy, precision, recall, f1-score**.  
- Provides **classification report** for per-class metrics.  
- Includes **cross-validation** to check robustness.  
- Visualizes performance using a **confusion matrix heatmap**.  



## Dataset  
- Dataset used: **Spambase.csv** .  
- Columns: 57 features representing email content and a target column.


## Model Evaluation Metrics  
- **Accuracy** → ratio of correct predictions to total predictions.  
- **Precision** → proportion of correctly predicted spam emails out of all predicted spam.  
- **Recall** → proportion of correctly predicted spam emails out of all actual spam.  
- **F1 Score** → harmonic mean of precision and recall.  
- **Confusion Matrix** → shows classification performance across both classes.  



## Results  
```
Accuracy: 0.9294
Precision: 0.9209
Recall: 0.8981
F1 Score: 0.9093

Classification Report:

             precision  recall   f1-score   support

Not Spam       0.93      0.95      0.94       558
Spam           0.92      0.90      0.91       363

accuracy                           0.93       921
macro avg      0.93      0.92      0.93       921
weighted avg   0.93      0.93      0.93       921

Cross-validation Accuracy: 0.9107 (+/- 0.0370)
```

## Confusion Matrix  
![Confusion Matrix](images/confusion_matrix.png)  


## How It Works  
1. Load and preprocess the dataset.  
2. Scale features using **StandardScaler**.  
3. Split into training and testing sets.  
4. Train a **Logistic Regression** model.  
5. Predict on test set.  
6. Evaluate with multiple metrics and cross-validation.  
7. Visualize confusion matrix using Seaborn heatmap.  


## Outcome
- A working spam detection system with >92% accuracy.

- Cross-validated and robust results.

- Clear visualization of classification performance.

- Reusable framework for trying other ML models.

## Future Improvements

- Add other classifiers (Naive Bayes, Random Forest, SVM, Neural Networks) for comparison.

- Apply NLP preprocessing (stopword removal, stemming, lemmatization).

- Use TF-IDF vectorization instead of raw numerical features.

- Deploy the model as a Flask/Django web app.

- Integrate into an email client for real-time spam detection.
d designed to strengthen practical machine learning and AI skills.  

---

## Projects Overview

1. Basic Calculator using Python  
2. Image Classifier using Keras and TensorFlow  
3. Simple Chatbot using predefined responses  
4. Spam Email Detector using Scikit-learn  
5. Handwritten Digit Recognition with MNIST dataset  
6. Sentiment Analysis on text data using NLTK  
7. Movie Recommendation System using cosine similarity  
8. Predict House Prices with Linear Regression  
9. Weather Forecasting using historical data  
10. Basic Neural Network from scratch  
11. Stock Price Prediction using Linear Regression  
12. Predict Diabetes using logistic regression  
13. Dog vs. Cat Classifier with CNN  
14. Tic-Tac-Toe AI using Minimax Algorithm  
15. Credit Card Fraud Detection using Scikit-learn  
16. Iris Flower Classification using Decision Trees  
17. Simple Personal Assistant using Python speech libraries  
18. Text Summarizer using NLTK  
19. Fake Product Review Detection using NLP techniques  
20. Detect Emotion in Text using NLTK  
21. Book Recommendation System using collaborative filtering  
22. Predict Car Prices using Random Forest  
23. Identify Fake News using Naive Bayes  
24. Resume Scanner using keyword extraction  
25. Customer Churn Prediction using classification algorithms  
26. Named Entity Recognition (NER) using spaCy  
27. Predict Employee Attrition using XGBoost  
28. Disease Prediction (e.g., Heart Disease) using ML algorithms  
29. Movie Rating Prediction using Collaborative Filtering  
30. Automatic Essay Grading using BERT  

---

## Goal

- Strengthen AI/ML fundamentals  
- Build practical, working projects quickly  
- Gain hands-on experience with libraries like TensorFlow, Scikit-learn, NLTK, spaCy, and more  
- Share progress daily for consistency  

---

## How to Use

- Each project is in its own folder  
- Open the folder to view the Python scripts and notebooks  
- Run the code in your local environment  

---

## Tech Stack

- **Python**  
- **TensorFlow / Keras**  
- **Scikit-learn**  
- **NLTK / spaCy**  
- **Pandas / NumPy / Matplotlib**  

---

## License

This project is for learning and personal development.  
Feel free to fork and experiment.  
