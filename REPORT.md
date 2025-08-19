# Employee Sentiment Analysis – Report

**Author:** Mohammed Ommar  
**Date:** 2025-08-19

## Objective
Analyze unlabeled employee messages to label sentiment, perform EDA, compute monthly scores, rank employees, detect flight risks, and build a baseline predictive model.

## Data & Preprocessing
- Dataset: `data/test.csv` (provided).  
- Columns auto-detected for **text**, **employee**, and **date** using heuristics.  
- Missing text filled with empty strings; dates coerced to datetimes when available.

## Methods
### Sentiment Labeling
- Deterministic lexicon approach (compact positive/negative wordlists).  
- Score = (#positives) - (#negatives).  
- Labels: Positive (>=1), Neutral (0), Negative (<= -1).

### EDA
- Distribution of labels across all messages.  
- Monthly trends (if dates available).  
- Basic text features: char length, word count.

### Monthly Scores
- Per message: Positive=+1, Neutral=0, Negative=-1.  
- Aggregated per **employee × month** to get `monthly_sentiment_score`.

### Ranking
- **Top-3 Positive** and **Top-3 Negative** employees per month.  
- Sorting: by score (desc/asc) then employee name alphabetically.

### Flight Risk
- Employee is flagged if they have **≥4 negative messages in any rolling 30-day window**.

### Predictive Model
- **Linear Regression** to predict monthly sentiment score.  
- Features: `message_count`, `avg_char_len`, `avg_word_count`, `neg_share`, `pos_share`.  
- Metrics: R², MAE.

## Key Results
- Label Distribution and Trend plots in `visualizations/`.  
- Rankings in CSVs: `top3_positive_by_month.csv`, `top3_negative_by_month.csv`.  
- Flight risk list in `flight_risk_employees.csv`.  
- Model summary in `ml_model_summary.json`: {
  "n_samples": 240,
  "r2": 0.9194763477531064,
  "mae": 0.7959380095864946,
  "coef": {
    "message_count": 0.5814427141214957,
    "avg_char_len": 0.003588888194961089,
    "avg_word_count": -0.01612573729629396,
    "neg_share": -13.62305370314275,
    "pos_share": 3.902527160980478
  },
  "intercept": -2.4266247701784023
}

## Recommendations
- Replace baseline lexicon with a stronger sentiment model (e.g., VADER/transformers).  
- Add topic modeling for qualitative insights.  
- Consider regularized or non-linear models for prediction.

## Reproducibility
- Run `main.ipynb` top-to-bottom. Outputs are written to CSV/PNG artifacts in repo.
