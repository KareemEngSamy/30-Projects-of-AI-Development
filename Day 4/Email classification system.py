import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score   #cross_val_score for robustness check
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report  #classification_report for more detailed metrics
from sklearn.preprocessing import StandardScaler   #scaling for logistic regression stability
import seaborn as sns
import matplotlib.pyplot as plt

#Model Evaluation Metrics:
#accuracy score is the ratio of correctly predicted instances to the total instances
#confusion matrix is used to evaluate the performance of a classification model
#precision score is the ratio of correctly predicted positive observations to the total predicted positives
#recall score is the ratio of correctly predicted positive observations to the all observations in actual class
#f1 score is the weighted average of precision and recall

#load the dataset
data = pd.read_csv('spambase.csv')
x = data.drop('spam', axis=1)
y = data['spam']

#scale the features for better performance of logistic regression
scaler = StandardScaler()   
x_scaled = scaler.fit_transform(x)

#split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42, stratify=y   #stratify for balanced class distribution
)

#train the logistic regression model to classify emails spam or not
model = LogisticRegression(max_iter=1000, solver='liblinear')  #max_iter and solver are for stability
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

#cross-validation to test model robustness
cv_scores = cross_val_score(model, x_scaled, y, cv=5, scoring='accuracy')
print(f"\nCross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

#visualize the confusion matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()   
plt.show()
