# IMDB Sentiment Classification (TF-IDF + LogisticRegression)

**Project:** Binary sentiment classification on the IMDB movie-review dataset (positive vs negative).  
**Language:** Python 3.9+  
**Libraries:** datasets, scikit-learn, pandas, numpy, joblib

## Project structure

imdb-sentiment-ml/
├── main_sentiment.py
├── requirements.txt
└── README.md

bash
Copy code

## Setup

1. (Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
requirements.txt:

shell
Copy code
datasets
scikit-learn>=1.2
pandas
numpy
joblib
tqdm
The script downloads the IMDB dataset (≈80 MB) the first time it runs via Hugging Face datasets. Internet required for the first run.

Run
bash
Copy code
python main_sentiment.py
The script will:

Download/load the IMDB dataset via datasets.load_dataset("imdb")

Subsample (optional) for quick runs (configurable)

Build a Pipeline with TfidfVectorizer + LogisticRegression

Run RandomizedSearchCV to tune C and ngram_range

Evaluate on held-out test set (accuracy, precision, recall, F1)

Save the best pipeline to sentiment_pipeline.joblib

Files produced
sentiment_pipeline.joblib — saved pipeline including TF-IDF vectorizer and classifier.

Next steps / ideas
Replace LogisticRegression with a small Transformer (e.g., distilbert) for better accuracy (requires transformers & GPU).

Add model explainability (LIME or SHAP).

Serve predictions with a small FastAPI app.