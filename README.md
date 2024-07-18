# stock-price-prediction
@CodeAlpha
# @CodeAlpha Stock Price Prediction using Linear Regression

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 2: Download Stock Data
# Using Yahoo Finance to download stock data
stock_symbol = 'AAPL'  # You can change this to any stock symbol
stock_data = yf.download(stock_symbol, start='2020-01-01', end='2023-12-31')

# Display the first few rows of the dataset
stock_data.head()
# Step 3: Data Preparation
# Convert the Date index to a column
stock_data['Date'] = stock_data.index
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data['Day'] = stock_data['Date'].apply(lambda x: x.toordinal())

# Define features (X) and target (y)
X = stock_data[['Day']]
y = stock_data['Close']

# Display the prepared data
stock_data[['Date', 'Close', 'Day']].head()
# Step 4: Split the Data
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print(f'Training set: X_train={X_train.shape}, y_train={y_train.shape}')
print(f'Testing set: X_test={X_test.shape}, y_test={y_test.shape}')


# Step 5: Train the Model
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Display the model coefficients
print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')



# Step 6: Make Predictions
# Predict the stock prices using the testing data
y_pred = model.predict(X_test)

# Evaluate the model performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')


# Step 7: Visualize the Results
# Plot the actual vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Date'], stock_data['Close'], label='Actual Price')
plt.scatter(pd.to_datetime(X_test['Day'], origin='unix', unit='D'), y_pred, color='red', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()



# Step 8: Predict Future Stock Price (Example)
# Predict the stock price for a future date
future_date = pd.to_datetime('2024-01-01').toordinal()
future_price = model.predict([[future_date]])
print(f'Predicted stock price for 2024-01-01: ${future_price[0]:.2f}')



