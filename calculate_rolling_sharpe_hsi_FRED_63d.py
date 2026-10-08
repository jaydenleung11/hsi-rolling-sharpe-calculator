import os

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta

TRADING_DAYS = 252


def fetch_t_bill_series(api_key, start_date):
    """
    Fetch the full history of the US 3-month T-bill yield (FRED series TB3MS)
    :param api_key: Your FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
    :param start_date: First observation date to pull
    :return: pandas Series of annualized yields (decimal) indexed by date, or None if unsuccessful
    """
    try:
        url = ("https://api.stlouisfed.org/fred/series/observations"
               f"?series_id=TB3MS&api_key={api_key}&file_type=json"
               f"&observation_start={start_date:%Y-%m-%d}")
        response = requests.get(url, timeout=20)

        if response.status_code == 200:
            observations = response.json().get('observations', [])
            series = pd.Series({pd.Timestamp(o['date']): float(o['value']) / 100  # percent -> decimal
                                for o in observations if o['value'] != '.'})
            if not series.empty:
                return series.sort_index()
        print(f"Warning: FRED API returned status {response.status_code}. Using fallback value.")
        return None
    except Exception as e:
        print(f"Error fetching from FRED: {str(e)}")
        return None


def align_risk_free(rf_series, index):
    """
    Spread FRED's monthly observations across the trading days in `index`.
    Each day uses the most recent monthly value on or before it, so the rate
    changes through time and never looks ahead.
    """
    combined = rf_series.reindex(rf_series.index.union(index)).ffill()
    return combined.reindex(index).bfill()  # bfill covers days before the first observation


def calculate_rolling_sharpe(ticker='^HSI', window_days=63, lookback_years=3, api_key=None):
    """
    Calculate a full rolling Sharpe ratio series using FRED for the risk-free rate
    :param ticker: Stock/index ticker (default: ^HSI)
    :param window_days: Rolling window in trading days (default: 63, about 3 months)
    :param lookback_years: Years of price history to pull (default: 3)
    :param api_key: FRED API key (if None, uses a flat 2% risk-free rate)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(lookback_years * 365.25))

    # Get asset data (auto_adjust=True folds dividends/splits into 'Close')
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if data.empty:
        print("Error: No price data fetched")
        return None

    if isinstance(data.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker) columns
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index).tz_localize(None)

    # Risk-free rate: one value per trading day from the FRED history, or flat 2%
    rf_series = fetch_t_bill_series(api_key, start_date - timedelta(days=62)) if api_key else None
    if rf_series is not None:
        data['Annual Risk Free'] = align_risk_free(rf_series, data.index)
    else:
        print("Using fallback risk-free rate: 2%")
        data['Annual Risk Free'] = 0.02

    if len(data) <= window_days:
        print(f"Error: only {len(data)} days of data, need more than the {window_days}-day window")
        return None

    # Prepare calculations
    data['Daily Risk Free'] = (1 + data['Annual Risk Free']) ** (1 / TRADING_DAYS) - 1  # annual -> daily
    data['Daily Return'] = data['Close'].pct_change()
    data['Excess Return'] = data['Daily Return'] - data['Daily Risk Free']

    # Rolling statistics
    data['Rolling Mean'] = data['Daily Return'].rolling(window_days).mean()
    data['Rolling Std'] = data['Daily Return'].rolling(window_days).std()

    # Annualized Sharpe ratio: rolling mean excess return / rolling std of excess return
    data['Rolling Sharpe'] = (data['Excess Return'].rolling(window_days).mean() /
                              data['Excess Return'].rolling(window_days).std() * np.sqrt(TRADING_DAYS))

    return data[data['Rolling Sharpe'].notna()].copy()


def display_results(data, ticker, window_days):
    """Format and print results"""
    if data is None:
        return

    results = data[['Close', 'Daily Return', 'Daily Risk Free',
                    'Rolling Mean', 'Rolling Std', 'Rolling Sharpe']].tail(10)

    print(f"\n{ticker} Rolling Sharpe Components (Latest 10 Days):")
    print("=" * 70)
    with pd.option_context('display.float_format', '{:.6f}'.format):
        print(results)

    sharpe = data['Rolling Sharpe']
    print(f"\nSummary Statistics ({len(sharpe)} rolling {window_days}-day readings, "
          f"{data.index[0]:%Y-%m-%d} to {data.index[-1]:%Y-%m-%d}):")
    print(f"- Latest Annual Risk-Free Rate: {100 * data['Annual Risk Free'].iloc[-1]:.2f}%")
    print(f"- Latest Sharpe Ratio: {sharpe.iloc[-1]:.4f}")
    print(f"- Average Sharpe Ratio: {sharpe.mean():.4f}")
    print(f"- Min / Max Sharpe: {sharpe.min():.4f} ({sharpe.idxmin():%Y-%m-%d}) / "
          f"{sharpe.max():.4f} ({sharpe.idxmax():%Y-%m-%d})")
    print(f"- Share of days with Sharpe > 1: {100 * (sharpe > 1).mean():.1f}%")
    print(f"- Latest Volatility (Std Dev, daily): {data['Rolling Std'].iloc[-1] * 100:.2f}%")


def plot_results(data, ticker, window_days):
    """Visualize results"""
    fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Sharpe Ratio
    ax[0].plot(data.index, data['Rolling Sharpe'], label='Sharpe Ratio', color='navy')
    ax[0].axhline(0, color='grey', linewidth=0.8)
    ax[0].set_ylabel('Sharpe Ratio')
    ax[0].legend()
    ax[0].grid(True)

    # Returns
    ax[1].plot(data.index, data['Rolling Mean'] * 100,
               label='Avg Daily Return (%)', color='green')
    ax[1].plot(data.index, data['Daily Risk Free'] * 100,
               label='Risk-Free Rate (%)', color='red', linestyle='--')
    ax[1].set_ylabel('Returns (%)')
    ax[1].legend()
    ax[1].grid(True)

    # Volatility
    ax[2].plot(data.index, data['Rolling Std'] * 100,
               label='Volatility (%)', color='purple')
    ax[2].set_ylabel('Std Dev (%)')
    ax[2].legend()
    ax[2].grid(True)

    plt.suptitle(f"{ticker} {window_days}-Day Rolling Analysis ({len(data)} readings)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration
    TICKER = '^HSI'          # Hang Seng Index
    WINDOW = 63              # 3 months of trading days (longer window = smoother, less noisy Sharpe)
    LOOKBACK_YEARS = 3       # years of history -> roughly 690 rolling readings
    FRED_API_KEY = os.environ.get('FRED_API_KEY')  # set it in your environment; never hard-code it

    # Run analysis
    print(f"Calculating {WINDOW}-day rolling Sharpe for {TICKER} over {LOOKBACK_YEARS} years...")
    df = calculate_rolling_sharpe(TICKER, WINDOW, LOOKBACK_YEARS, FRED_API_KEY)

    if df is not None:
        display_results(df, TICKER, WINDOW)
        df.to_csv('HSI_rolling_sharpe.csv')
        plot_results(df, TICKER, WINDOW)
