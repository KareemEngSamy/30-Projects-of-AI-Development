import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report  
import seaborn as sns
import matplotlib.pyplot as plt

#load the dataset
data = pd.read_csv('HAR.csv')

#preprocess the dataset
x = data.drop('Activity', axis=1)
y = data['Activity']

#split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#test_size is the proportion of the dataset to include in the test split
#0.2 means 20% test and 80% train

#train the Random Forest Classifier model to classify human activities
#random forest uses multiple decision trees to improve accuracy and control over-fitting
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1) 
#n_estimators is the number of trees in the forest
#random_state is for reproducibility    
#n_jobs=-1 to use all CPU cores for faster training 
model.fit(x_train, y_train)

#make predictions
y_pred = model.predict(x_test)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)   
precision = precision_score(y_test, y_pred, average='weighted')  #average='weighted' for multiclass
recall = recall_score(y_test, y_pred, average='weighted')       
f1 = f1_score(y_test, y_pred, average='weighted')                
confusion = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')    
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1 * 100:.2f}%')

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#visualize the confusion matrix
plt.figure(figsize=(8,6)) 
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())  
plt.xlabel('Predicted') #lablels from dataset
plt.ylabel('Actual')    
plt.title('Confusion Matrix')
plt.tight_layout()   
plt.show()

#feature importance plot to understand what features matter most
importances = model.feature_importances_
indices = importances.argsort()[::-1][:15]   #show top 15 features
plt.figure(figsize=(10,6))
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [x.columns[i] for i in indices], rotation=90)
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()
