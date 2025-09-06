# Music_Recommendation_System
**Aim:** To develop a music recommendation system that leverages Natural Language Processing to analyze and classify a user's emotional state from text input. The core approach involves training a machine learning model to predict sentiment and subsequently recommend a song from the dataset that aligns with the detected emotion. 

### Project Description
This project is a Music Recommendation System built as a web application using Streamlit. It leverages machine learning to analyze user-provided text, predict their emotional state, and recommend a song from a pre-curated dataset that matches the detected sentiment. The application demonstrates a full end-to-end machine learning workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model building, and deployment.

### Features
* **Sentiment Analysis:** Uses a powerful RandomForestClassifier with TfidfVectorizer to accurately predict sentiment from text input.

* **Data Cleaning:** Handles missing values, inconsistent data types, and duplicate entries in a messy dataset.

* **EDA:** Provides visual insights into the dataset through interactive charts for sentiment and genre distribution.

* **Hyperparameter Tuning:** Employs GridSearchCV to automatically find the best model parameters for optimal accuracy.

* **Interactive UI:** A user-friendly web interface built with Streamlit allows users to input their feelings and get instant song recommendations.

### Repository Structure
The project is structured with a single main file for simplicity and ease of use.

* app.py: The main Python script containing all the code for the Streamlit application, including data preparation, model training, and the user interface.

* music_sentiment_dataset.csv: The primary dataset used for training the model.

* requirements.txt: A file listing all the necessary Python libraries for the project.

* README.md: This file, which provides a comprehensive overview of the project.

### Requirements
To run this project, you need to have Python installed on your system. It is highly recommended to use a virtual environment to manage dependencies.

The required libraries are listed in requirements.txt. You can install them all at once using pip

       pip install -r requirements.txt

### How to Run the Project
Follow these steps to get the application running on your local machine.

### Clone the Repository:

         git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
         cd your-repository-name

### Create a Virtual Environment (recommended):

# On Windows
           python -m venv venv
           venv\Scripts\activate

# On macOS/Linux
           python3 -m venv venv
           source venv/bin/activate

### Install the Requirements:

           pip install -r requirements.txt

### Run the Streamlit Application:
Ensure you have the music_sentiment_dataset_messy_realistic.csv file in the same directory as app.py.

          streamlit run app.py

After running the command, your web browser will automatically open a new tab with the "Music_Recommendation_System" application.
