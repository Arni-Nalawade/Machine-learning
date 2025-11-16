# Import necessary libraries
import sklearn.datasets as datasets
import string
import re
from stop_words import get_stop_words

# Load in the training and test datasets
d_train = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
d_test = datasets.fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

# Load dictionary from the text file
def load_dictionary(file_path):
    with open(file_path, 'r') as file:
        dictionary = [line.strip() for line in file.readlines()]
    return dictionary

dictionary = load_dictionary('keywords769.txt')

# Getting libraries for sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Processs data from txt 
def preprocess_texts(texts, stop_words_set):
    #Array to store processed text
    cleaned = []
    trans_table = str.maketrans('', '', string.punctuation) # Remove punctuation
    for doc in texts:
        if not isinstance(doc, str):
            doc = str(doc)
        s = doc.replace('\n',' ') #remove new lines
        s = s.translate(trans_table) # Remove punctuation
        s = s.lower() # Lowercase

        tokens = s.split() # Tokenize
        tokens = [t for t in tokens if not re.search(r'\d',t)] # Remove tokens with digits
        tokens = [t for t in tokens if t not in stop_words_set] # Remove stop words
        cleaned.append(' '.join(tokens)) # Rejoin tokens into a single string
    return cleaned

def run_experiments(dictionary, dict_name):
    print(f"\n--- Experiments with dictionary: {dict_name} (length={len(dictionary)}) ---")
    stop_words_set = set(get_stop_words('en'))
    X_train_text = preprocess_texts(d_train.data, stop_words_set)
    X_test_text = preprocess_texts(d_test.data, stop_words_set)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    y_train = d_train.target
    y_test = d_test.target

    # Logistic Regression One-vs-Rest
    clf_ovr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
    clf_ovr.fit(X_train, y_train)
    pred_ovr = clf_ovr.predict(X_test)
    acc_ovr = accuracy_score(y_test, pred_ovr)
    print(f"LogisticRegression One-vs-Rest accuracy: {acc_ovr:.4f}")

    # Logistic Regression One-vs-One
    clf_ovo = OneVsOneClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))
    clf_ovo.fit(X_train, y_train)
    pred_ovo = clf_ovo.predict(X_test)
    acc_ovo = accuracy_score(y_test, pred_ovo)
    print(f"LogisticRegression One-vs-One accuracy: {acc_ovo:.4f}")

    # SVM with linear kernel for different C values
    for C in [1000, 100, 1]:
        clf_svm = SVC(kernel='linear', C=C)
        clf_svm.fit(X_train, y_train)
        pred_svm = clf_svm.predict(X_test)
        acc_svm = accuracy_score(y_test, pred_svm)
        print(f"SVM (linear) C={C} accuracy: {acc_svm:.4f}")

if __name__ == "__main__":
    # Run experiments 
    run_experiments(load_dictionary('keywords769.txt'), 'keywords769.txt')
    run_experiments(load_dictionary('keywords35.txt'), 'keywords35.txt')
