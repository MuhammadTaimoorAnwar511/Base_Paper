import yfinance as yf
import pandas as pd

# Define the ticker symbol for Bitcoin
ticker = "BTC-USD"  # Bitcoin in USD on Yahoo Finance

# Define the date range
start_date = "2014-09-17"  # September 17, 2014
end_date = "2024-07-15"    # July 15, 2024

# Fetch the data
bitcoin_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

# Save the data to a CSV file
output_file = "Data/bitcoin_data.csv"
bitcoin_data.to_csv(output_file)
print(f"Data saved to {output_file}")

# Display the first few rows of the data
print(bitcoin_data.head())
