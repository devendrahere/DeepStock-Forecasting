from backend.data import download_stock_data

df = download_stock_data("AAPL", "5y")
print("COLUMNS:", df.columns)
print(df.head())
