from flask import Flask, request, jsonify
import yfinance as yf
from finvizfinance.quote import finvizfinance
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import requests
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Load FinBERT model and tokenizer, downloading if necessary
model_name = "ProsusAI/finbert"
model_dir = "./finbert"
if not os.path.exists(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Alpha Vantage API key (replace with your own)
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "YOUR_API_KEY")

def get_historical_data(ticker):
    """Fetch 1-year historical stock data using yfinance."""
    stock = yf.Ticker(ticker)
    try:
        df = stock.history(period="1y")
    except:
        return None
    if df.empty:
        return None
    return df

def get_technical_indicators(df):
    """Calculate technical indicators using ta."""
    # Calculate 50-day and 200-day SMA
    sma_50 = SMAIndicator(df['Close'], window=50).sma_indicator()
    sma_200 = SMAIndicator(df['Close'], window=200).sma_indicator()
    # Calculate RSI
    rsi = RSIIndicator(df['Close'], window=14).rsi()
    # Calculate MACD
    macd = MACD(df['Close']).macd()
    
    latest_sma_50 = sma_50.iloc[-1] if not sma_50.empty else None
    latest_sma_200 = sma_200.iloc[-1] if not sma_200.empty else None
    latest_rsi = rsi.iloc[-1] if not rsi.empty else None
    latest_macd = macd.iloc[-1] if not macd.empty else None
    
    return {
        "sma_50": latest_sma_50,
        "sma_200": latest_sma_200,
        "rsi": latest_rsi,
        "macd": latest_macd
    }

TICKER_MAPPING = {
    "tesla": "TSLA",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "intel": "INTC"
}

def get_sentiment_score(ticker):
    """Analyze news sentiment using FinBERT."""
    try:
        stock = finvizfinance(ticker)
        news = stock.ticker_news()
        if not news:
            return 0.0
        headlines = [item['Title'] for item in news][:10]  # Limit to 10 headlines
        sentiments = []
        for headline in headlines:
            inputs = tokenizer(headline, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            print(outputs)
            probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            # FinBERT: [negative, positive, neutral]
            sentiment_score = probs[1] - probs[0]  # Positive - Negative
            sentiments.append(sentiment_score)
        return np.mean(sentiments) if sentiments else 0.0
    except:
        return 0.0

def get_fundamentals(ticker):
    """Fetch fundamental data using Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if not data or "Symbol" not in data:
            return {}
        return {
            "pe_ratio": float(data.get("PERatio", 0)) or None,
            "debt_to_equity": float(data.get("DebtToEquityRatio", 0)) or None,
            "operating_margin": float(data.get("OperatingMarginTTM", 0)) or None
        }
    except:
        return {}

def analyze_stock(ticker):
    print("starting analysis")
    """Analyze stock and provide summary and recommendation."""
    # Fetch historical data
    df = get_historical_data(ticker)
    if df is None:
        return {"error": "Invalid ticker or no data available"}
    print( "Data fetched successfully")
    # Technical indicators
    indicators = get_technical_indicators(df)
    
    # Sentiment analysis
    sentiment_score = get_sentiment_score(ticker)
    
    # Fundamentals
    fundamentals = get_fundamentals(ticker)
    
    # Trend analysis
    latest_price = df['Close'][-1]
    price_1y_ago = df['Close'][0]
    trend = ((latest_price - price_1y_ago) / price_1y_ago) * 100
    
    # Pattern analysis (simplified: check for golden/death cross)
    sma_50 = indicators["sma_50"]
    sma_200 = indicators["sma_200"]
    pattern = "Neutral"
    if sma_50 and sma_200:
        pattern = "Bullish (Golden Cross)" if sma_50 > sma_200 else "Bearish (Death Cross)"
    
    # Recommendation logic (heuristic scoring)
    score = 0
    # Sentiment: +1 for positive, -1 for negative
    if sentiment_score > 0.1:
        score += 1
    elif sentiment_score < -0.1:
        score -= 1
    # Trend: +1 for upward trend > 10%, -1 for downward < -10%
    if trend > 10:
        score += 1
    elif trend < -10:
        score -= 1
    # RSI: +1 if 30 < RSI < 70 (neutral), -1 if overbought (>70) or oversold (<30)
    if indicators["rsi"] and 30 < indicators["rsi"] < 70:
        score += 1
    elif indicators["rsi"] and (indicators["rsi"] > 70 or indicators["rsi"] < 30):
        score -= 1
    # MACD: +1 if MACD > 0 (bullish), -1 if < 0
    if indicators["macd"] and indicators["macd"] > 0:
        score += 1
    elif indicators["macd"] and indicators["macd"] < 0:
        score -= 1
    # Fundamentals: +1 for low P/E (<15), low D/E (<1), high margin (>0.1)
    if fundamentals.get("pe_ratio") and fundamentals["pe_ratio"] < 15:
        score += 1
    if fundamentals.get("debt_to_equity") and fundamentals["debt_to_equity"] < 1:
        score += 1
    if fundamentals.get("operating_margin") and fundamentals["operating_margin"] > 0.1:
        score += 1
    
    recommendation = "Buy" if score > 0 else "Sell" if score < 0 else "Hold"
    
    # Summary
    summary = (
        f"Analysis for {ticker}:\n"
        f"- Current Price: ${latest_price:.2f}\n"
        f"- 1-Year Price Change: {trend:.2f}%\n"
        f"- Sentiment: {'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'} (Score: {sentiment_score:.2f})\n"
        f"- Technical Pattern: {pattern}\n"
        f"- RSI: {indicators['rsi']:.2f} ({'Overbought' if indicators['rsi'] > 70 else 'Oversold' if indicators['rsi'] < 30 else 'Neutral'})\n"
        f"- MACD: {indicators['macd']:.2f} ({'Bullish' if indicators['macd'] > 0 else 'Bearish'})\n"
        f"- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}\n"
        f"- Debt-to-Equity: {fundamentals.get('debt_to_equity', 'N/A')}\n"
        f"- Operating Margin: {fundamentals.get('operating_margin', 'N/A')}\n"
        f"Recommendation: {recommendation}"
    )
    
    # Factors considered
    factors = [
        "1-year historical price data (trend and patterns)",
        "Technical indicators (50-day SMA, 200-day SMA, RSI, MACD)",
        "Market sentiment (news headlines analyzed with FinBERT)",
        "Fundamental metrics (P/E ratio, debt-to-equity ratio, operating margin)"
    ]
    
    return {
        "ticker": ticker,
        "summary": summary,
        "recommendation": recommendation,
        "factors_considered": factors
    }

@app.route("/api/analyze", methods=["GET"])
def analyze():
    """API endpoint to analyze stock based on ticker."""
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    
    # Map common names to tickers
    ticker = TICKER_MAPPING.get(ticker.lower(), ticker)
    
    try:
        result = analyze_stock(ticker.upper())
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)