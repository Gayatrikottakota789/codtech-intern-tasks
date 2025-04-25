import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
from nltk.corpus import stopwords

# Download NLTK resources (only need to do this once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')


def clean_text(text):
    """Cleans the text by removing noise."""
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove @mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'https?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase

    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])

    return text


def analyze_sentiment(text):
    """Analyzes sentiment and returns label and score."""
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']

    if compound_score >= 0.05:
        return "positive", compound_score
    elif compound_score <= -0.05:
        return "negative", compound_score
    else:
        return "neutral", compound_score


# Main execution
if __name__ == "__main__":
    # Load data from CSV
    try:
        df = pd.read_csv("customer_reviews(task-4).csv")  # Ensure CSV is in the same directory
    except FileNotFoundError:
        print(
            "Error: 'customer_reviews.csv' not found. "
            "Please create the file or adjust the filename."
        )
        exit()

    # Clean the text data
    df["cleaned_text"] = df["text"].apply(clean_text)

    # Analyze sentiment
    df[["sentiment", "sentiment_score"]] = df["cleaned_text"].apply(
        lambda x: pd.Series(analyze_sentiment(x))
    )

    # Display results
    print(df.head())

    # Sentiment summary
    print("\nSentiment Analysis Summary:")
    print(df["sentiment"].value_counts())