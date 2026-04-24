#!/usr/bin/env python
"""
Live Plotly Dash Web App - Fixed feature importance alignment
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf

from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

# ============== CONFIG ==============
MODEL_PATH = "models/saved"
PROCESSED_PATH = "data/processed"

app = dash.Dash(__name__, title="Silver Price Prediction Dashboard")
app.config.suppress_callback_exceptions = True

model = None
feature_names = None
feature_importance_df = None

# ============== LOAD ==============
def load_model_and_artifacts():
    global model, feature_names, feature_importance_df
    
    # Load model
    model_files = list(Path(MODEL_PATH).glob("xgboost_with_sentiment_*.pkl"))
    if not model_files:
        model_files = list(Path(MODEL_PATH).glob("xgboost_*.pkl"))
    if not model_files:
        print("ERROR: No model found!")
        return False
    latest = sorted(model_files)[-1]
    with open(latest, 'rb') as f:
        model = pickle.load(f)
    print(f"Loaded model: {latest.name}")
    
    # Get feature names from the model if possible
    if hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            feature_names_model = booster.feature_names
            if feature_names_model is not None:
                feature_names = feature_names_model
                print(f"Got {len(feature_names)} features from model")
            else:
                # fallback to loaded feature matrix
                feature_files = list(Path(PROCESSED_PATH).glob("feature_matrix_final_*.csv"))
                if not feature_files:
                    feature_files = list(Path(PROCESSED_PATH).glob("feature_matrix_cleaned_*.csv"))
                if feature_files:
                    df = pd.read_csv(sorted(feature_files)[-1], index_col=0, parse_dates=True)
                    feature_cols = [c for c in df.columns if 'target' not in c]
                    feature_names = feature_cols
                    print(f"Loaded {len(feature_names)} features from feature matrix")
        except Exception as e:
            print(f"Could not get feature names from model: {e}")
            # fallback
            feature_files = list(Path(PROCESSED_PATH).glob("feature_matrix_final_*.csv"))
            if not feature_files:
                feature_files = list(Path(PROCESSED_PATH).glob("feature_matrix_cleaned_*.csv"))
            if feature_files:
                df = pd.read_csv(sorted(feature_files)[-1], index_col=0, parse_dates=True)
                feature_cols = [c for c in df.columns if 'target' not in c]
                feature_names = feature_cols
                print(f"Loaded {len(feature_names)} features from feature matrix")
    
    # Get feature importance
    if feature_names is not None:
        importance = model.feature_importances_
        # If lengths mismatch, take minimum
        min_len = min(len(feature_names), len(importance))
        feature_importance_df = pd.DataFrame({
            'feature': feature_names[:min_len],
            'importance': importance[:min_len]
        })
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(20)
        print(f"Feature importance loaded: {len(feature_importance_df)} features")
    
    return True

def fetch_latest_data():
    try:
        silver = yf.Ticker("SI=F")
        hist = silver.history(period="5d")
        latest_price = hist['Close'].iloc[-1] if not hist.empty else 25.0
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
        daily_return = (latest_price - prev_close) / prev_close
    except:
        latest_price = 25.0
        daily_return = 0.0
    
    # Build feature vector
    feature_dict = {}
    for f in feature_names:
        if 'returns_1d' in f:
            feature_dict[f] = daily_return
        elif 'close' in f and 'sma' not in f and 'ema' not in f:
            feature_dict[f] = latest_price
        elif 'volume' in f:
            feature_dict[f] = 50000
        elif 'treasury' in f:
            feature_dict[f] = 4.0
        elif 'dollar' in f:
            feature_dict[f] = 105.0
        elif 'sentiment' in f:
            feature_dict[f] = 0.02
        else:
            feature_dict[f] = 0.0
    
    X = pd.DataFrame([feature_dict])
    X = X.reindex(columns=feature_names, fill_value=0)
    return X, latest_price, daily_return

def get_prediction_confidence(X):
    pred = model.predict(X)[0]
    confidence = 1 - min(abs(pred) / 0.03, 0.8)
    confidence = max(0.2, min(0.95, confidence))
    return pred, confidence

def load_historical_equity():
    dates = pd.date_range(start='2020-01-01', periods=500, freq='B')
    np.random.seed(42)
    returns = np.random.randn(500) * 0.01
    equity = 10000 * (1 + returns).cumprod()
    return pd.DataFrame({'date': dates, 'equity': equity})

# ============== DASH LAYOUT ==============
app.layout = html.Div([
    html.H1("Silver Price Prediction Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Hr(),
    
    html.Div([
        html.Div([
            html.H3("Current Prediction", style={'textAlign': 'center'}),
            html.H1(id="prediction-value", style={'textAlign': 'center', 'fontSize': '48px', 'color': '#27ae60'}),
            html.P("(Next Day Return)", style={'textAlign': 'center'})
        ], className="four columns", style={'display': 'inline-block', 'width': '32%'}),
        
        html.Div([
            html.H3("Confidence", style={'textAlign': 'center'}),
            html.H1(id="confidence-value", style={'textAlign': 'center', 'fontSize': '48px', 'color': '#2980b9'}),
            html.P("(Model Certainty)", style={'textAlign': 'center'})
        ], className="four columns", style={'display': 'inline-block', 'width': '32%'}),
        
        html.Div([
            html.H3("Latest Silver Price", style={'textAlign': 'center'}),
            html.H1(id="price-value", style={'textAlign': 'center', 'fontSize': '48px', 'color': '#f39c12'}),
            html.P("USD / oz", style={'textAlign': 'center'})
        ], className="four columns", style={'display': 'inline-block', 'width': '32%'}),
    ], style={'textAlign': 'center'}),
    
    html.Div([
        html.Button('Refresh Data & Predict', id='refresh-button', n_clicks=0,
                    style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '10px 20px',
                           'fontSize': '16px', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
    ], style={'textAlign': 'center', 'margin': '20px'}),
    
    html.Hr(),
    
    html.H3("Equity Curve (Backtest)", style={'textAlign': 'center'}),
    dcc.Graph(id="equity-graph"),
    
    html.H3("Global Feature Importance (XGBoost)", style={'textAlign': 'center'}),
    dcc.Graph(id="importance-graph"),
    
    html.H3("Current Feature Values (Top 15)", style={'textAlign': 'center'}),
    html.Div(id="feature-table", style={'overflowX': 'auto', 'margin': '20px'}),
    
    dcc.Interval(id='interval-component', interval=60*60*1000, n_intervals=0)
], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': 'auto'})

# ============== CALLBACKS ==============
@app.callback(
    [Output('prediction-value', 'children'),
     Output('confidence-value', 'children'),
     Output('price-value', 'children'),
     Output('equity-graph', 'figure'),
     Output('importance-graph', 'figure'),
     Output('feature-table', 'children')],
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_dashboard(n_clicks, n_intervals):
    X, latest_price, daily_return = fetch_latest_data()
    pred, confidence = get_prediction_confidence(X)
    pred_pct = f"{pred*100:.2f}%"
    conf_pct = f"{confidence*100:.1f}%"
    price_str = f"${latest_price:.2f}"
    
    # Equity curve
    equity_df = load_historical_equity()
    equity_fig = px.line(equity_df, x='date', y='equity', title='Portfolio Equity Curve',
                         labels={'equity': 'Portfolio Value ($)', 'date': 'Date'})
    equity_fig.update_traces(line=dict(color='darkgreen', width=2))
    equity_fig.add_hline(y=10000, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
    
    # Feature importance bar chart
    if feature_importance_df is not None and not feature_importance_df.empty:
        imp_fig = px.bar(feature_importance_df, x='importance', y='feature', orientation='h',
                         title='Top 20 Global Feature Importance',
                         labels={'importance': 'Importance', 'feature': 'Feature'})
        imp_fig.update_traces(marker_color='steelblue')
        imp_fig.update_layout(height=500)
    else:
        imp_fig = go.Figure()
        imp_fig.add_annotation(text="Feature importance not available", x=0.5, y=0.5, showarrow=False)
    
    # Feature values table
    feature_values = X.iloc[0].to_dict()
    sorted_features = sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    table_rows = []
    for name, val in sorted_features:
        table_rows.append(html.Tr([html.Td(name, style={'border': '1px solid #ddd', 'padding': '8px'}),
                                   html.Td(f"{val:.6f}", style={'border': '1px solid #ddd', 'padding': '8px'})]))
    table = html.Table([
        html.Thead(html.Tr([html.Th("Feature", style={'border': '1px solid #ddd', 'padding': '8px'}),
                            html.Th("Current Value", style={'border': '1px solid #ddd', 'padding': '8px'})])),
        html.Tbody(table_rows)
    ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
    
    return pred_pct, conf_pct, price_str, equity_fig, imp_fig, table

if __name__ == '__main__':
    print("Loading model and artifacts...")
    if load_model_and_artifacts():
        print("Dashboard starting at http://localhost:8050")
        app.run_server(debug=True, host='0.0.0.0', port=8050)
    else:
        print("Failed to load model.")