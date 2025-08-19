# Employee Sentiment Analysis

This repository contains a complete solution for the **Employee Sentiment Analysis** project. It labels messages, performs EDA, computes monthly sentiment scores, ranks employees, identifies flight risks, and builds a simple predictive linear regression model.

## Project Structure
```
.
├── main.ipynb
├── labeled_messages.csv
├── employee_monthly_scores.csv
├── top3_positive_by_month.csv
├── top3_negative_by_month.csv
├── flight_risk_employees.csv
├── ml_model_summary.json
├── visualizations/
│   ├── sentiment_distribution.png
│   ├── monthly_sentiment_trend.png
│   └── linear_regression_coefficients.png
├── data/
│   └── test.csv
├── docs/
│   ├── Problem_Statement.docx
│   └── Problem_Statement.pdf  (*placeholder note about conversion*)
└── README.md
```

## Quick Start
1. Place the dataset at `data/test.csv` (already added by this bundle).
2. Open `main.ipynb` and run all cells to reproduce everything from raw data to outputs.
3. Results are saved as CSVs and PNGs in the repo folders.

## Methods Overview
- **Sentiment Labeling**: Compact lexicon-based scoring (+1 for positive words, -1 for negative words), labeled as Positive / Neutral / Negative with simple thresholds.
- **EDA**: Distribution of labels and monthly trends; basic stats on message lengths and word counts.
- **Monthly Scores**: Positive = +1, Neutral = 0, Negative = -1; aggregated per **employee × month**.
- **Ranking**: Top-3 positive and bottom-3 negative employees per month (ties broken alphabetically).
- **Flight Risk**: Flag employees who send **≥4 negative messages in any rolling 30-day window**.
- **Predictive Model**: Linear Regression to predict monthly sentiment score using features: message_count, avg_char_len, avg_word_count, neg_share, pos_share.

## Key Outputs
- `labeled_messages.csv`: original data + `sentiment_label`, `sentiment_num`.
- `employee_monthly_scores.csv`: the monthly score per employee.
- `top3_positive_by_month.csv`, `top3_negative_by_month.csv`: rankings.
- `flight_risk_employees.csv`: flagged employees.
- `ml_model_summary.json`: R², MAE, and coefficients for the regression model.

## Notes
- This baseline is **deterministic** and does not require internet access or heavy models.
- You can upgrade the sentiment component to `VADER`, `AFINN`, or `transformers` if allowed internet/packages.
- Date handling is automatic if a date column is present; otherwise, a single dummy month is used.
- The code is robust to common schema names (e.g., `message`, `employee_id`, `date`). See the first cell in `main.ipynb`.

## How to Improve
- Swap the lexicon for a better sentiment model (e.g., VADER from NLTK).
- Add topic modeling (LDA) to cluster concerns.
- Build a classification model with weak labels (pseudo-labeling) if gold labels become available.
- Use regularization (Ridge/Lasso) or tree models for the predictive task.
