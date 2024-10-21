import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
import matplotlib.pyplot as plt

















# data.py
stock_symbol = 'AAPL'




























API_KEY = 'XKV2M77TLLDJHXQK'

# Function to fetch and save historical stock prices using yfinance
def fetch_historical_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        today = datetime.today().strftime('%Y-%m-%d')
        df = stock.history(start="2015-01-01", end=today)
        if df.empty:
            print(f"No historical data found for {symbol}")
            return
        print(f"Columns in historical data: {df.columns}")
        df.reset_index(inplace=True)
        if 'Adj Close' not in df.columns:
            if 'Close' in df.columns:
                df.rename(columns={'Close': 'Adj Close'}, inplace=True)
            else:
                print(f"'Adj Close' or 'Close' column not found in the historical data for {symbol}")
                return
        df = df[['Date', 'Adj Close', 'Volume']]
        df.to_csv(f'{symbol}_historical_data.csv', index=False)
        print(f"Historical stock data for {symbol} saved.")
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")

# Function to fetch and save balance sheet data
def fetch_balance_sheet_data(symbol, api_key):
    try:
        url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        if 'quarterlyReports' in data:
            df = pd.DataFrame(data['quarterlyReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df[df['fiscalDateEnding'] >= '2015-01-01']
            df.to_csv(f'{symbol}_balance_sheet_quarterly.csv', index=False)
            print(f"Balance sheet data for {symbol} saved.")
        else:
            print(f"Error fetching balance sheet data for {symbol}: {data}")
    except Exception as e:
        print(f"Exception occurred while fetching balance sheet data for {symbol}: {e}")

# Function to fetch and save cash flow data
def fetch_cash_flow_data(symbol, api_key):
    try:
        url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        if 'quarterlyReports' in data:
            df = pd.DataFrame(data['quarterlyReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df[df['fiscalDateEnding'] >= '2015-01-01']
            df.to_csv(f'{symbol}_cash_flow_quarterly.csv', index=False)
            print(f"Cash flow data for {symbol} saved.")
        else:
            print(f"Error fetching cash flow data for {symbol}: {data}")
    except Exception as e:
        print(f"Exception occurred while fetching cash flow data for {symbol}: {e}")

# Function to fetch and save income statement data
def fetch_income_statement_data(symbol, api_key):
    try:
        url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        if 'quarterlyReports' in data:
            df = pd.DataFrame(data['quarterlyReports'])
            df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df = df[df['fiscalDateEnding'] >= '2015-01-01']
            df.to_csv(f'{symbol}_income_statement_quarterly.csv', index=False)
            print(f"Income statement data for {symbol} saved.")
        else:
            print(f"Error fetching income statement data for {symbol}: {data}")
    except Exception as e:
        print(f"Exception occurred while fetching income statement data for {symbol}: {e}")

# Main function to fetch and save all data for a given stock symbol
def fetch_all_data(symbol, api_key):
    fetch_historical_stock_data(symbol)
    fetch_balance_sheet_data(symbol, api_key)
    fetch_cash_flow_data(symbol, api_key)
    fetch_income_statement_data(symbol, api_key)

# Function to plot historical stock data
def plot_historical_stock_data(symbol):
    # Read the historical data CSV file
    file_name = f'{symbol}_historical_data.csv'
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Plot the adjusted close price against time
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Adj Close'], label='Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.title(f'Historical Adjusted Close Price for {symbol}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to process and merge data
def process_and_merge_data(symbol):
    # Load the historical data
    historical_data_path = f'{symbol}_historical_data.csv'
    historical_data = pd.read_csv(historical_data_path)
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], utc=True)
    final_data = historical_data[['Date', 'Adj Close', 'Volume']]
    final_data.to_csv(f'formatted_{symbol}_data.csv', index=False)
    print(f"Formatted data saved to 'formatted_{symbol}_data.csv'")

    # Load the existing CSV with historical data
    historical_data = pd.read_csv(f'formatted_{symbol}_data.csv')
    historical_data['Date'] = pd.to_datetime(historical_data['Date'], utc=True)
    balance_sheet_data = pd.read_csv(f'{symbol}_balance_sheet_quarterly.csv')
    balance_sheet_data['fiscalDateEnding'] = pd.to_datetime(balance_sheet_data['fiscalDateEnding'], utc=True)

    # Calculate DE Ratio
    balance_sheet_data['DE Ratio'] = balance_sheet_data['totalLiabilities'] / balance_sheet_data['totalShareholderEquity']
    balance_sheet_data.set_index('fiscalDateEnding', inplace=True)
    historical_data.set_index('Date', inplace=True)
    balance_sheet_data = balance_sheet_data.reindex(historical_data.index, method='ffill')
    historical_data['DE Ratio'] = balance_sheet_data['DE Ratio']
    historical_data.reset_index(inplace=True)
    historical_data.to_csv(f'formatted_{symbol}_data.csv', index=False)
    print(f"DE Ratio added to 'formatted_{symbol}_data.csv'")

    # Load the CSV files
    balance_sheet = pd.read_csv(f'{symbol}_balance_sheet_quarterly.csv')
    income_statement = pd.read_csv(f'{symbol}_income_statement_quarterly.csv')
    existing_data = pd.read_csv(f'formatted_{symbol}_data.csv')

    # Convert fiscalDateEnding to datetime for merging
    balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
    income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])

    # Merge balance sheet and income statement on fiscalDateEnding
    financial_data = pd.merge(balance_sheet[['fiscalDateEnding', 'totalShareholderEquity']],
                              income_statement[['fiscalDateEnding', 'netIncome']],
                              on='fiscalDateEnding', how='inner')
    financial_data['ROE'] = financial_data['netIncome'] / financial_data['totalShareholderEquity']
    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.quarter
    existing_data['Year'] = pd.to_datetime(existing_data['Date']).dt.year
    existing_data['Quarter'] = pd.to_datetime(existing_data['Date']).dt.quarter
    combined_data = pd.merge(existing_data, 
                             financial_data[['Year', 'Quarter', 'ROE']],
                             on=['Year', 'Quarter'], 
                             how='left')
    combined_data['ROE'] = combined_data['ROE'].ffill()
    combined_data.drop(columns=['Year', 'Quarter'], inplace=True)
    combined_data.to_csv(f'formatted_{symbol}_data.csv', index=False)
    print(f"ROE added to 'formatted_{symbol}_data.csv'")

    # Load the CSV files
    balance_sheet = pd.read_csv(f'{symbol}_balance_sheet_quarterly.csv')
    existing_data = pd.read_csv(f'formatted_{symbol}_data.csv')

    # Convert fiscalDateEnding to datetime for merging
    balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
    financial_data = balance_sheet[['fiscalDateEnding', 'totalShareholderEquity', 'commonStockSharesOutstanding']]
    existing_data['Date'] = pd.to_datetime(existing_data['Date'])
    existing_data['Year'] = existing_data['Date'].dt.year
    existing_data['Quarter'] = existing_data['Date'].dt.quarter
    financial_data['Price/Book'] = existing_data['Adj Close'] / (financial_data['totalShareholderEquity'] / financial_data['commonStockSharesOutstanding'])
    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.quarter
    combined_data = pd.merge(existing_data, 
                             financial_data[['Year', 'Quarter', 'Price/Book']],
                             on=['Year', 'Quarter'], 
                             how='left')
    combined_data['Price/Book'] = combined_data['Price/Book'].ffill()
    combined_data.drop(columns=['Year', 'Quarter'], inplace=True)
    combined_data.to_csv(f'formatted_{symbol}_data.csv', index=False)
    print(f"Price/Book added to 'formatted_{symbol}_data.csv'")

    # Load the existing combined data
    combined_data = pd.read_csv(f'formatted_{symbol}_data.csv', parse_dates=['Date'])
    income_statement = pd.read_csv(f'{symbol}_income_statement_quarterly.csv')
    income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
    financial_data = income_statement[['fiscalDateEnding', 'totalRevenue', 'netIncome']]
    financial_data['Profit Margin'] = financial_data['netIncome'] / financial_data['totalRevenue']
    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.quarter
    combined_data['Year'] = combined_data['Date'].dt.year
    combined_data['Quarter'] = combined_data['Date'].dt.quarter
    combined_data = pd.merge(combined_data, 
                             financial_data[['Year', 'Quarter', 'Profit Margin']],
                             on=['Year', 'Quarter'], 
                             how='left')
    combined_data['Profit Margin'] = combined_data['Profit Margin'].ffill()
    combined_data.drop(columns=['Year', 'Quarter'], inplace=True)
    combined_data.to_csv(f'formatted_{symbol}_data.csv', index=False)
    print(f"Profit Margin added to 'formatted_{symbol}_data.csv'")

    # Load the balance sheet and income statement data
    balance_sheet = pd.read_csv(f'{symbol}_balance_sheet_quarterly.csv')
    income_statement = pd.read_csv(f'{symbol}_income_statement_quarterly.csv')
    balance_sheet['fiscalDateEnding'] = pd.to_datetime(balance_sheet['fiscalDateEnding'])
    income_statement['fiscalDateEnding'] = pd.to_datetime(income_statement['fiscalDateEnding'])
    financial_data = pd.merge(balance_sheet, income_statement, on='fiscalDateEnding', how='left')
    financial_data['dilutedEPS'] = financial_data['netIncome'] / financial_data['commonStockSharesOutstanding']
    financial_data = financial_data[['fiscalDateEnding', 'dilutedEPS']]
    combined_data = pd.read_csv(f'formatted_{symbol}_data.csv')
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    combined_data['Year'] = combined_data['Date'].dt.year
    combined_data['Quarter'] = combined_data['Date'].dt.to_period('Q')
    financial_data['Year'] = financial_data['fiscalDateEnding'].dt.year
    financial_data['Quarter'] = financial_data['fiscalDateEnding'].dt.to_period('Q')
    combined_data = pd.merge(combined_data, financial_data, on=['Year', 'Quarter'], how='left')
    combined_data['dilutedEPS'] = combined_data['dilutedEPS'].ffill()
    combined_data.to_csv(f'formatted_{symbol}_data.csv', index=False)
    print(f"Diluted EPS added to 'formatted_{symbol}_data.csv'")

    # Load the stock and SP500 historical data
    stock_data = pd.read_csv(f'formatted_{symbol}_data.csv', parse_dates=['Date'])
    sp500_data = pd.read_csv('SP500_data.csv', parse_dates=['Date'])
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True).dt.tz_localize(None)
    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], utc=True).dt.tz_localize(None)
    merged_data = pd.merge(stock_data, sp500_data, on='Date', suffixes=('_stock', '_sp500'))

    # Calculate daily returns for stock and SP500
    merged_data['Returns_stock'] = merged_data['Adj Close_stock'].pct_change()
    merged_data['Returns_sp500'] = merged_data['Adj Close_sp500'].pct_change()
    merged_data.dropna(subset=['Returns_stock', 'Returns_sp500'], inplace=True)

    # Define a rolling window size, e.g., 252 trading days (approximately one year)
    window_size = 252

    # Calculate rolling covariance and variance
    rolling_cov = merged_data['Returns_stock'].rolling(window=window_size).cov(merged_data['Returns_sp500'])
    rolling_var_sp500 = merged_data['Returns_sp500'].rolling(window=window_size).var()

    # Calculate rolling beta
    merged_data['Beta'] = rolling_cov / rolling_var_sp500

    # Forward fill the Beta values to align with daily data
    merged_data['Beta'] = merged_data['Beta'].ffill()

    # Save the updated dataset with Beta
    merged_data.to_csv(f'formatted_{symbol}_data_with_rolling_beta.csv', index=False)
    print(f"Updated dataset with rolling Beta saved to 'formatted_{symbol}_data_with_rolling_beta.csv'")

fetch_all_data(stock_symbol, API_KEY)
process_and_merge_data(stock_symbol)










#########################################











def clean_stock_data(stock_symbol):
    """
    Cleans the stock data by renaming columns and removing unnecessary columns.

    Parameters:
    stock_symbol (str): The stock symbol/name used to construct input and output file paths.
    """
    input_file = f'formatted_{stock_symbol}_data_with_rolling_beta.csv'
    output_file = f'cleaned_{stock_symbol}_data.csv'
    
    # Load the data
    data = pd.read_csv(input_file)

    # Rename the columns
    column_mapping = {
        'Date': 'Date',
        'Adj Close_stock': 'adjusted_close_price',
        'Volume_stock': 'trading_volume',
        'DE Ratio': 'debt_to_equity_ratio',
        'ROE': 'return_on_equity',
        'Price/Book': 'price_to_book_ratio',
        'Profit Margin': 'profit_margin',
        'dilutedEPS': 'diluted_earnings_per_share',
        'Beta': 'company_beta'
    }

    # Rename the columns
    data = data.rename(columns=column_mapping)

    # Define the columns to keep
    columns_to_keep = [
        'Date', 'adjusted_close_price', 'trading_volume', 'debt_to_equity_ratio',
        'return_on_equity', 'price_to_book_ratio', 'profit_margin',
        'diluted_earnings_per_share', 'company_beta'
    ]
    
    # Keep only the necessary columns
    data = data[columns_to_keep]

    # Save the cleaned data
    data.to_csv(output_file, index=False)

    print(f"Column names and unnecessary columns removed. Data saved to '{output_file}'")

def replace_empty_values_with_mean(stock_symbol):
    """
    Replaces empty values in the stock data with the average value of the column.

    Parameters:
    stock_symbol (str): The stock symbol/name used to construct the input file path.
    """
    input_file = f'cleaned_{stock_symbol}_data.csv'
    
    # Load the data
    data = pd.read_csv(input_file)

    # Replace empty values with the average value of the column
    for column in data.columns:
        if data[column].isnull().any() or (data[column] == '').any():
            column_mean = data[column].replace('', pd.NA).astype(float).mean()
            data[column].replace('', column_mean, inplace=True)
            data[column].fillna(column_mean, inplace=True)

    # Save the updated data to the same CSV file
    data.to_csv(input_file, index=False)

    print(f"Empty values replaced with column averages. Data updated in '{input_file}'")

clean_stock_data(stock_symbol)
replace_empty_values_with_mean(stock_symbol)

