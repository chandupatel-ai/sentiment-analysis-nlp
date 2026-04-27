import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_csv('IMDB Dataset.csv')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

print("Loading and training... please wait...")
df['clean'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

preds = model.predict(X_test_vec)
print("Model ready! Accuracy:", round(accuracy_score(y_test, preds) * 100, 2), "%")
print("-" * 40)

# Predict any sentence
def predict(sentence):
    cleaned = clean_text(sentence)
    vec = tfidf.transform([cleaned])
    result = model.predict(vec)[0]
    if result == 1:
        return "POSITIVE"
    else:
        return "NEGATIVE"

# Test sentences
print(predict("This movie was absolutely amazing!"))
print(predict("Worst film I have ever seen."))
print(predict("The acting was brilliant and story was touching."))
print(predict("Total waste of time and money."))
print(predict("It was okay, nothing special."))
# Interactive mode - type your own sentences!
print("-" * 40)
print("Type any sentence and press Enter.")
print("Type 'quit' to stop.")
print("-" * 40)

while True:
    user_input = input("Your sentence: ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    result = predict(user_input)
    print("Sentiment:", result)
    print()