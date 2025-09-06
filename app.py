import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Set page configuration for a professional look
st.set_page_config(
    page_title="Emotion-Based Music Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use Streamlit's caching to load and clean data only once
@st.cache_data
def load_and_clean_data(file_path):
    """
    Loads the messy dataset, cleans it, and returns the cleaned DataFrame.
    
    Args:
        file_path (str): The path to the CSV dataset.
        
    Returns:
        pd.DataFrame: The cleaned pandas DataFrame.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at {file_path}. Please ensure the file is in the same directory as the app.")
        return None

    st.sidebar.header("Data Cleaning Report")
    st.sidebar.markdown(f"**Original shape:** {df.shape}")
    
    # --- Step 1: Handling Missing Values ---
    df.dropna(subset=['User_ID', 'User_Text', 'Sentiment_Label'], inplace=True)
    for col in ['Artist', 'Genre', 'Song_Name', 'Mood']:
        df[col] = df[col].fillna(f'Unknown {col.split("_")[0]}')
    
    # Fill remaining numerical NaNs with the mean
    for col in ['Tempo (BPM)', 'Energy', 'Danceability']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].mean())
    
    st.sidebar.markdown("**Missing values handled.**")

    # --- Step 2: Handling Inconsistent Data & Data Types ---
    # Convert 'Energy' and 'Danceability' to numerical, handling errors
    df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')
    df['Danceability'] = pd.to_numeric(df['Danceability'], errors='coerce')
    
    # Replace outliers like '999' with the mean for numerical columns
    if 'Energy' in df.columns:
        df['Energy'] = df['Energy'].replace(999, df['Energy'].mean())
    
    st.sidebar.markdown("**Data types and outliers handled.**")

    # --- Step 3: Removing Duplicates ---
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    st.sidebar.markdown(f"**Duplicates removed:** {duplicates_removed}")
    st.sidebar.markdown(f"**Final shape:** {df.shape}")
    
    # Final check on sentiment labels
    df['Sentiment_Label'] = df['Sentiment_Label'].str.strip().str.title()
    
    return df

# Define a text preprocessor function for Feature Engineering
def preprocess_text(text):
    """Simple text preprocessing function."""
    text = str(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower()  # Convert to lowercase
    return text

# Use Streamlit's caching for the model training process
@st.cache_resource
def train_and_tune_model(df):
    """
    Performs feature engineering, model building, and hyperparameter tuning.
    
    Returns:
        Pipeline: The trained and tuned scikit-learn pipeline.
    """
    st.sidebar.header("Model Training Report")
    
    # Apply text preprocessing
    df['User_Text_Cleaned'] = df['User_Text'].apply(preprocess_text)
    
    # Split the data
    X = df['User_Text_Cleaned']
    y = df['Sentiment_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Feature Selection/Importance ---
    st.sidebar.subheader("Top Words per Sentiment")
    sentiment_words = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        sentiment = row['Sentiment_Label']
        text = row['User_Text_Cleaned']
        words = text.split()
        for word in words:
            sentiment_words[sentiment][word] += 1
            
    for sentiment, words_count in sentiment_words.items():
        top_words = sorted(words_count.items(), key=lambda item: item[1], reverse=True)[:5]
        words_str = ", ".join([f"'{word}'" for word, count in top_words])
        st.sidebar.markdown(f"**{sentiment}**: {words_str}")

    # --- Model Building and Hyperparameter Tuning ---
    st.sidebar.subheader("Model Building")
    st.sidebar.write("Training with GridSearchCV for optimal performance...")
    
    # Create a pipeline with TfidfVectorizer and a RandomForestClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Define the parameter grid for GridSearchCV
    param_grid = {
    'tfidf__ngram_range': [(1,1), (1,2)],
    'tfidf__max_df': [0.75, 0.85, 1.0],
    'tfidf__min_df': [1, 2, 5],
    'clf__n_estimators': [200, 300, 400, 500],
    'clf__max_depth': [10, 20, 30, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}


    # Perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    st.sidebar.write(f"Best Parameters Found: {grid_search.best_params_}")
    
    best_pipeline = grid_search.best_estimator_
    
    # Evaluate the best model
    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.metric("Final Model Accuracy", f"{accuracy:.2f}")

    return best_pipeline

def recommend_song(sentiment, df):
    """
    Recommends a random song from the dataset based on the predicted sentiment.
    """
    matching_songs = df[df['Sentiment_Label'] == sentiment]
    if not matching_songs.empty:
        return matching_songs.sample(1).iloc[0]
    return None

# --- Main Streamlit Application ---

st.title("🎶 Emotion-Based Music Recommender")
st.markdown("### A project for the Machine Learning and Data Science marking rubric.")

# Dataset Info Section
st.subheader("1. Dataset Information")
st.markdown("This project utilizes a music sentiment dataset to build a recommendation system. The dataset includes user-provided text, which is classified by a machine learning model to predict their emotional state.")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Key Variables")
    st.write("""
    - `User_Text`: The input text from the user (the feature).
    - `Sentiment_Label`: The target sentiment (e.g., Happy, Sad, Relaxed).
    - `Song_Name`, `Artist`, `Genre`: Metadata for song recommendations.
    """)
with col2:
    st.markdown("#### Sample Data")
    # Load data for displaying a sample
    sample_df = pd.read_csv('music_sentiment_dataset.csv').head(5)
    st.dataframe(sample_df)

st.markdown("---")

# --- Exploratory Data Analysis (EDA) ---
st.subheader("2. Data Analysis and Visualization")
st.markdown("Visualizations provide insight into the dataset's structure and characteristics.")

eda_col1, eda_col2 = st.columns(2)

df = load_and_clean_data('music_sentiment_dataset.csv')

if df is not None:
    with eda_col1:
        st.write("### Sentiment Distribution")
        sentiment_counts = df['Sentiment_Label'].value_counts()
        fig_sentiment, ax_sentiment = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax_sentiment, palette="viridis")
        ax_sentiment.set_title('Distribution of Sentiments')
        ax_sentiment.set_xlabel('Sentiment')
        ax_sentiment.set_ylabel('Number of Entries')
        st.pyplot(fig_sentiment)
    
    with eda_col2:
        st.write("### Top 10 Genres")
        genre_counts = df['Genre'].value_counts().head(10)
        fig_genre, ax_genre = plt.subplots()
        sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax_genre, palette="magma")
        ax_genre.set_title('Top 10 Genres')
        ax_genre.set_xlabel('Number of Entries')
        ax_genre.set_ylabel('Genre')
        st.pyplot(fig_genre)

    st.markdown("---")

    # --- Model Training and Prediction ---
    st.subheader("3. Sentiment-Based Music Recommendation")
    
    pipeline = train_and_tune_model(df)
    
    if pipeline:
        st.markdown("#### Try It Out!")
        user_input = st.text_area("Tell me how you are feeling today...", "I'm feeling really sad and need a song to cheer me up.")
        
        if st.button("Get My Song Recommendation"):
            if user_input:
                predicted_sentiment = pipeline.predict([user_input])[0]
                song_rec = recommend_song(predicted_sentiment, df)

                if song_rec is not None:
                    st.success(f"I predict you are feeling: **{predicted_sentiment}**! Here's a song for you:")
                    
                    st.markdown(f"**Song:** {song_rec['Song_Name']}")
                    st.markdown(f"**Artist:** {song_rec['Artist']}")
                    st.markdown(f"**Genre:** {song_rec['Genre']}")
                    st.markdown(f"**Mood:** {song_rec['Mood']}")
                    st.markdown(f"**Energy:** {song_rec['Energy']:.2f}")
                    st.markdown(f"**Danceability:** {song_rec['Danceability']:.2f}")
                else:
                    st.warning("Sorry, no song found for that sentiment. Try a different text!")
            else:
                st.warning("Please enter some text to get a recommendation.")