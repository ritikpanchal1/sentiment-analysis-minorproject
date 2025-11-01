import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Loading (Cached for Performance) ---

# Load VADER (NLTK)
@st.cache_resource
def load_vader():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_vader()

# Load RoBERTa (Hugging Face Transformers)
@st.cache_resource
def load_roberta_model():
    """Loads the RoBERTa model and tokenizer."""
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load_roberta_model()

# --- Helper Functions (from your notebook) ---

def polarity_scores_roberta(example):
    """
    Analyzes text using the RoBERTa model.
    """
    encoded_text = tokenizer(example, return_tensors='pt', max_length=512, truncation=True)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

@st.cache_data # Caches the data loading and expensive processing
def load_and_process_data(file_path, num_rows):
    """
    Loads the CSV, samples it, and runs both VADER and RoBERTa analysis,
    mimicking the logic from the Jupyter Notebook.
    """
    try:
        df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        df = df.head(num_rows)  # Sample the dataframe
        
        res = {}
        for i, row in df.iterrows():
            text = row['Text']
            myid = row['Id']
            
            # VADER Analysis
            vader_result = sia.polarity_scores(text)
            vader_result_rename = {f"vader_{k}": v for k, v in vader_result.items()}
            
            # RoBERTa Analysis
            try:
                roberta_result = polarity_scores_roberta(text)
            except RuntimeError:
                # Handle texts that are too long or other errors
                roberta_result = {'roberta_neg': np.nan, 'roberta_neu': np.nan, 'roberta_pos': np.nan}
            
            both = {**vader_result_rename, **roberta_result}
            res[myid] = both
        
        # Merge results with original dataframe (as in notebook)
        results_df = pd.DataFrame(res).T
        results_df = results_df.reset_index().rename(columns={'index': 'Id'})
        results_df = results_df.merge(df, how='left')
        return results_df
    
    except FileNotFoundError:
        return None

# --- UI Layout ---

st.title("Sentiment Analysis AI Dashboard")

# --- Section 1: Live Sentiment Analyzer ---
st.header("Analyze Your Text")
st.write("Enter text below to see the sentiment analysis from both models.")

user_text = st.text_area("Enter a review, comment, or sentence:", "This is a fantastic product! I am very happy with my purchase.")

if st.button("Analyze Sentiment"):
    if user_text:
        st.subheader("Analysis Results")
        
        col1, col2 = st.columns(2)
        
        # --- VADER Analysis ---
        with col1:
            st.info("VADER Model (Rule-based)")
            vader_scores = sia.polarity_scores(user_text)
            
            if vader_scores['compound'] >= 0.05:
                vader_sentiment = f"Positive (Score: {vader_scores['compound']})"
                st.markdown(f"**Sentiment: :green[{vader_sentiment}]**")
            elif vader_scores['compound'] <= -0.05:
                vader_sentiment = f"Negative (Score: {vader_scores['compound']})"
                st.markdown(f"**Sentiment: :red[{vader_sentiment}]**")
            else:
                vader_sentiment = f"Neutral (Score: {vader_scores['compound']})"
                st.markdown(f"**Sentiment: :blue[{vader_sentiment}]**")
            
            st.json({
                'Positive': vader_scores['pos'],
                'Neutral': vader_scores['neu'],
                'Negative': vader_scores['neg'],
                'Compound': vader_scores['compound']
            })

        # --- RoBERTa Analysis ---
        with col2:
            st.info("RoBERTa Model (Transformer-based)")
            roberta_scores = polarity_scores_roberta(user_text)
            
            # Get the highest score's label
            roberta_sentiment = max(roberta_scores, key=roberta_scores.get)
            
            if roberta_sentiment == 'roberta_pos':
                sentiment_label = f"Positive (Score: {roberta_scores['roberta_pos']:.4f})"
                st.markdown(f"**Sentiment: :green[{sentiment_label}]**")
            elif roberta_sentiment == 'roberta_neg':
                sentiment_label = f"Negative (Score: {roberta_scores['roberta_neg']:.4f})"
                st.markdown(f"**Sentiment: :red[{sentiment_label}]**")
            else:
                sentiment_label = f"Neutral (Score: {roberta_scores['roberta_neu']:.4f})"
                st.markdown(f"**Sentiment: :blue[{sentiment_label}]**")
            
            # Format for clean display
            formatted_roberta = {
                "Positive": f"{roberta_scores['roberta_pos']:.4f}",
                "Neutral": f"{roberta_scores['roberta_neu']:.4f}",
                "Negative": f"{roberta_scores['roberta_neg']:.4f}"
            }
            st.json(formatted_roberta)
    else:
        st.warning("Please enter some text to analyze.")

# --- Section 2: Dataset Exploration (from Notebook) ---
st.header("Explore the 'Reviews.csv' Dataset")
st.write("This section loads the first 500 rows of your dataset and runs the full analysis, just like in your notebook.")

DATA_FILE = 'Reviews.csv'  # Make sure this file is in the same folder
results_df = load_and_process_data(DATA_FILE, 500)

if results_df is None:
    st.error(f"Error: `{DATA_FILE}` not found. Please place your dataset in the same directory as this `app.py` file.")
else:
    st.success("Successfully loaded and processed `Reviews.csv`.")
    
    # --- Show Plots from Notebook ---
    st.subheader("Visual Analysis of Model Scores")
    
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        # Plot 1: VADER Compound Score
        st.write("VADER Compound Score Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(results_df['vader_compound'], kde=True, ax=ax1, bins=30)
        ax1.set_title("VADER Compound Score")
        st.pyplot(fig1)

    with plot_col2:
        # Plot 2: Original Score Distribution
        st.write("Original User Score Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(results_df['Score'], kde=False, ax=ax2, bins=5, discrete=True)
        ax2.set_title("Original Review 'Score'")
        st.pyplot(fig2)

    # Plot 3: RoBERTa Scores
    st.write("RoBERTa Score Distributions")
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(results_df['roberta_neg'], kde=True, ax=axes[0], color='red', bins=30)
    axes[0].set_title("RoBERTa Negative")
    sns.histplot(results_df['roberta_neu'], kde=True, ax=axes[1], color='gray', bins=30)
    axes[1].set_title("RoBERTa Neutral")
    sns.histplot(results_df['roberta_pos'], kde=True, ax=axes[2], color='green', bins=30)
    axes[2].set_title("RoBERTa Positive")
    st.pyplot(fig3)
    
    # Plot 4: VADER vs. RoBERTa
    st.write("Comparison: VADER Compound vs. RoBERTa Positive")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=results_df, x='vader_compound', y='roberta_pos', hue='Score', palette='coolwarm', ax=ax4)
    ax4.set_title("VADER Compound vs. RoBERTa Positive (Colored by Original Score)")
    st.pyplot(fig4)
    
    # --- Show Dataframe ---
    st.subheader("Analyzed Data Sample")
    st.write("This is the final dataframe combining the original data with the scores from both models.")
    st.dataframe(results_df.head())
