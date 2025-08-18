
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Tuple
from sklearn.linear_model import LinearRegression
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure lexicon is available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

def load_config() -> Dict[str, str]:
    load_dotenv(override=True)
    cfg = {
        "DATA_PATH": os.getenv("DATA_PATH", "data/employee_feedback.csv"),
        "DATE_COL": (os.getenv("DATE_COL") or "").strip(),
        "TEXT_COL": (os.getenv("TEXT_COL") or "").strip(),
        "EMP_ID_COL": (os.getenv("EMP_ID_COL") or "").strip(),
        "DEPT_COL": (os.getenv("DEPT_COL") or "").strip(),
        "OUT_DIR": "outputs"
    }
    Path(cfg["OUT_DIR"]).mkdir(parents=True, exist_ok=True)
    return cfg

def _find_col(cols, candidates):
    cset = [c.lower().strip() for c in cols]
    for cand in candidates:
        if cand in cset:
            return cols[cset.index(cand)]
    return None

def load_data(cfg: Dict[str,str]) -> Tuple[pd.DataFrame, Dict[str,str]]:
    path = cfg["DATA_PATH"]
    df = pd.read_csv(path)
    original_cols = df.columns.tolist()

    # Normalize headers for detection
    lower_map = {c: c.lower().strip() for c in original_cols}
    inv_map = {v:k for k,v in lower_map.items()}

    # Try auto-detect
    date_col  = cfg["DATE_COL"] or _find_col(original_cols, ["date","created_at","timestamp","created","time"])
    text_col  = cfg["TEXT_COL"] or _find_col(original_cols, ["text","feedback","comment","review","message","body"])
    emp_col   = cfg["EMP_ID_COL"] or _find_col(original_cols, ["employee_id","emp_id","id","employee","user_id"])
    dept_col  = cfg["DEPT_COL"] or _find_col(original_cols, ["department","dept","team"])

    needed = {"DATE_COL":date_col, "TEXT_COL":text_col, "EMP_ID_COL":emp_col}
    for k,v in needed.items():
        if not v:
            raise ValueError(f"Could not detect required column for {k}. Please set it in .env")

    # Parse date
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, text_col, emp_col]).copy()

    return df, {"date":date_col, "text":text_col, "emp":emp_col, "dept":dept_col}

def add_sentiment(df: pd.DataFrame, cols: Dict[str,str]) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    txt = cols["text"]
    scores = df[txt].astype(str).apply(lambda x: sia.polarity_scores(str(x))["compound"])
    df = df.copy()
    df["compound"] = scores

    def label(v):
        if v >= 0.05:
            return "Positive"
        elif v <= -0.05:
            return "Negative"
        return "Neutral"

    df["sentiment_label"] = df["compound"].apply(label)
    return df

def monthly_sentiment(df: pd.DataFrame, cols: Dict[str,str]) -> pd.DataFrame:
    date_col = cols["date"]
    mdf = df.copy()
    mdf["month"] = mdf[date_col].dt.to_period("M").dt.to_timestamp()
    out = mdf.groupby("month")["compound"].mean().reset_index()
    return out

def employee_ranking(df: pd.DataFrame, cols: Dict[str,str]) -> pd.DataFrame:
    emp = cols["emp"]
    g = df.groupby(emp).agg(
        avg_compound=("compound","mean"),
        n=("compound","size"),
        neg_share=("sentiment_label", lambda x: (x=="Negative").mean() if len(x)>0 else 0.0),
        pos_share=("sentiment_label", lambda x: (x=="Positive").mean() if len(x)>0 else 0.0),
    ).reset_index().sort_values(["avg_compound","n"], ascending=[False, False])
    return g

def linear_trend(monthly_df: pd.DataFrame) -> Tuple[float, float]:
    \"\"\"Fit LinearRegression on monthly mean sentiment vs month index.
    Returns (slope, intercept).\"\"\"
    if monthly_df.empty:
        return 0.0, 0.0
    x = (monthly_df["month"].view("int64") // 10**9).values.reshape(-1,1)  # seconds since epoch
    x = (x - x.min())  # normalize to start at 0
    y = monthly_df["compound"].values
    model = LinearRegression()
    model.fit(x, y)
    return float(model.coef_[0]), float(model.intercept_)

def identify_flight_risk(df: pd.DataFrame, cols: Dict[str,str]) -> pd.DataFrame:
    emp = cols["emp"]
    date_col = cols["date"]
    risks = []
    for eid, sub in df.sort_values(date_col).groupby(emp):
        avg = sub["compound"].mean()
        neg_share = (sub["sentiment_label"]=="Negative").mean() if len(sub)>0 else 0.0

        # compute short trend
        tmp = sub.copy()
        tmp["t"] = tmp[date_col].view("int64") // 10**9
        if len(tmp) >= 2:
            X = (tmp["t"].values.reshape(-1,1) - tmp["t"].values.min())
            y = tmp["compound"].values
            reg = LinearRegression().fit(X,y)
            slope = float(reg.coef_[0])
        else:
            slope = 0.0

        last3 = sub["compound"].tail(3).mean() if len(sub)>=3 else sub["compound"].mean()

        flag = (avg < -0.20) or (neg_share > 0.60 and len(sub)>=5) or (slope < -0.05 and last3 < 0)
        risks.append({
            emp: eid,
            "avg_compound": avg,
            "neg_share": neg_share,
            "entries": len(sub),
            "trend_slope": slope,
            "last3_avg": last3,
            "flight_risk": bool(flag)
        })
    return pd.DataFrame(risks).sort_values(["flight_risk","avg_compound"], ascending=[False, True])

def save_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
