import nltk
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Function to extract common words from text
def extract_common_words(text, num_words=10):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in tokens if word.lower() not in stop_words and word.lower() not in string.punctuation]

    # filter out punctuations that arent standard
    filtered_words = [word for word in filtered_words if word.strip() != '’' and word.strip() != '“' and word.strip() != '”']

    filtered_words = [word for word in filtered_words if word.strip() != '' and not word.isdigit()]

    # Count the occurrences of each word
    word_counts = Counter(filtered_words)

    # Get the most common words
    common_words = word_counts.most_common(num_words)
    
    return common_words

# Function to perform POS tagging and sentiment analysis for a single document
def analyze_document(document_text):
    # Tokenize the text
    tokens = word_tokenize(document_text)

    # Perform POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in tokens if word.lower() not in stop_words and word.lower() not in string.punctuation]

    # Filter out empty strings and digits
    filtered_words = [word for word in filtered_words if word.strip() != '' and not word.isdigit()]

    # Join filtered words into a single string for sentiment analysis
    filtered_text = ' '.join(filtered_words)

    # Perform sentiment analysis using VADER
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(filtered_text)

    return pos_tags, sentiment_scores

# Path to the folder containing the documents
documents_folder = 'lay'

count = 0

# Iterate over each document in the folder
for filename in os.listdir(documents_folder):
    # Read the content of the document
    with open(os.path.join(documents_folder, filename), 'r', encoding='utf-8', errors='replace') as file:
        document_text = file.read()

    # Perform analysis for the document
    pos_tags, sentiment_scores = analyze_document(document_text)
    
    # Print results
    if(sentiment_scores['compound'] < -0.05):
        count+=1
        common_words = extract_common_words(document_text, 10)
        print(f"Document with negative sentiment: {filename}")
        print(sentiment_scores)
        print(common_words)

print(count)