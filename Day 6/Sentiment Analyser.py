import nltk
import random
from nltk.corpus import movie_reviews, stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

#(uncomment if running for the first time)               <<<<<<<<------------
#download necessary NLTK resources 
# nltk.download('movie_reviews')        
# nltk.download('punkt')
# nltk.download('stopwords')

# Preprocessing: feature extractor
# Load English stopwords once to avoid repeated calls
stop_words = set(stopwords.words('english'))

#Convert a list of words into a feature dictionary -> Lowercases words / Removes stopwords / Marks presence of words with True
def extract_features(words):
    return {word.lower(): True for word in words if word.lower() not in stop_words}

# Load and prepare the dataset
# Each document = (list of words, category)
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Shuffle to remove any order bias
random.shuffle(documents)

# Apply feature extraction
featuresets = [(extract_features(doc), category) for (doc, category) in documents]

# Split into training (80%) and testing (20%)
split = int(0.8 * len(featuresets))
train_set, test_set = featuresets[:split], featuresets[split:]

# Train Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate accuracy
accuracy = nltk_accuracy(classifier, test_set)
print(f'\nAccuracy: {accuracy * 100:.2f}%')

# Show most informative features
classifier.show_most_informative_features(10)
print("\n")

# Function for analyzing custom sentences
def analyze_sentiment(sentence):
    #Takes a sentence as input, tokenizes it, extracts features and returns the predicted sentiment (pos/neg).
    words = nltk.word_tokenize(sentence)
    features = extract_features(words)
    return classifier.classify(features)

# Test the sentiment analyzer
test_sentences = [ 
    "I absolutely loved this movie! It was fantastic and thrilling.",
    "This was the worst film I have ever seen. Totally boring and dull.",
    "What a masterpiece! The storyline and characters were incredible.",
    "I didn't like the movie. It was too long and uninteresting.",
    "A delightful experience, I would watch it again!" ,
    "Terrible plot and poor acting. Not worth my time." ,
    "I fell asleep halfway through, it was that boring." ,
    "A beautiful film with stunning visuals and a touching story." ,
    "I regret watching this movie, it was a complete waste of time." 
]

for sentence in test_sentences:
    sentiment = analyze_sentiment(sentence)
    print(f'Sentence: \"{sentence}\"')
    print(f'Predicted Sentiment: {sentiment}\n')
