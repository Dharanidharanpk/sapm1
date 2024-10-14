import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import date
from PIL import Image
import requests
from io import BytesIO


# Function to display image from Google Drive
def display_image_from_google_drive(image_url, width=800):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    st.image(img, use_column_width=False, width=width)


# Set up session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Define the correct credentials
correct_username = "sharun875421"
correct_password = "1234"

# Main page for login
if not st.session_state.logged_in:
    # Title and Image for the Republic of Kailasa
    st.markdown("<h1 style='text-align: center;'>REPUBLIC OF KAILASAA</h1>", unsafe_allow_html=True)

    # Display the image centered
    image_url = "https://drive.google.com/uc?export=view&id=1otFI6-mvLF0hs4wRBbj548l4K6ZA-0C9"
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    display_image_from_google_drive(image_url,width=750)
    st.markdown("</div>", unsafe_allow_html=True)

    # Show login form only if not logged in
    with st.form("login_form"):
        st.markdown("<h3 style='text-align: center;'>RESERVE BANK OF KAILASAA Employee Login</h3>", unsafe_allow_html=True)

        # Create login form fields
        username = st.text_input("RB of KAILASAA Employee ID")
        password = st.text_input("Password", type="password")

        # Login button
        login_button = st.form_submit_button("Login")

        # Login validation
        if login_button:
            if username == correct_username and password == correct_password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Incorrect username or password. Please try again.")

# If the user is logged in, show the app
if st.session_state.logged_in:
    # Clear the login form by showing the app interface
    st.markdown("<h2 style='text-align: center;'>RESERVE BANK OF KAILASAA</h2>", unsafe_allow_html=True)

    # Input number of stocks
    num_stocks = st.number_input("Enter the number of stocks you want to analyze:", min_value=1, value=1)

    # Dynamically create fields based on the number of stocks
    stock_names = []
    for i in range(num_stocks):
        stock_symbol = st.text_input(f"Enter Stock Symbol {i + 1} (e.g., AAPL for Apple):", "")
        stock_names.append(stock_symbol)

    # Input date range
    start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    end_date = st.date_input("End Date", value=date.today())


    # Function to fetch stock data from Yahoo Finance
    def get_stock_data(ticker):
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        return data


    # Exponential Moving Average (EMA)
    def ema(data, window):
        return data.ewm(span=window, adjust=False).mean()


    # Relative Strength Index (RSI)
    def rsi(data, window=14):
        delta = data.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


    # Moving Average Convergence Divergence (MACD)
    def macd(data, fast=12, slow=26, signal=9):
        ema_fast = ema(data['Close'], fast)
        ema_slow = ema(data['Close'], slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        return macd_line, signal_line


    # Rate of Change (ROC)
    def roc(data, window=10):
        return data.pct_change(periods=window) * 100


    # Button to analyze stocks
    if st.button("Analyze Stock"):
        for stock_name in stock_names:
            if stock_name:
                # Fetch stock data
                stock_data = get_stock_data(stock_name)
                st.write(f"Showing data for {stock_name}")
                st.write(stock_data.head())  # Display first few rows of the data

                # -- Fundamental Analysis --
                st.header(f"Fundamental Analysis for {stock_name}")
                stock = yf.Ticker(stock_name)
                stock_info = stock.info
                st.write("Company Info:")
                st.write(f"Sector: {stock_info['sector']}")
                st.write(f"Industry: {stock_info['industry']}")
                st.write(f"Market Cap: {stock_info['marketCap'] / 1_000_000_000:.2f} billion")
                st.write(f"PE Ratio (TTM): {stock_info['trailingPE']}")
                st.write(f"EPS (TTM): {stock_info['trailingEps']}")
                st.write(f"Dividend Yield: {stock_info.get('dividendYield', 'N/A')}")

                # -- Technical Analysis --
                st.header(f"Technical Analysis for {stock_name}")

                # Calculate and plot EMA
                stock_data['EMA_20'] = ema(stock_data['Close'], 20)
                stock_data['EMA_50'] = ema(stock_data['Close'], 50)

                st.subheader("Exponential Moving Average (EMA)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Close Price"))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_20'], mode='lines', name="EMA 20"))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA_50'], mode='lines', name="EMA 50"))
                fig.update_layout(title=f"EMA - 20 and 50 for {stock_name}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

                # Calculate and display RSI
                stock_data['RSI'] = rsi(stock_data['Close'], 14)
                st.subheader("Relative Strength Index (RSI)")

                # Create the figure
                fig = go.Figure()

                # Add the RSI line
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name="RSI"))

                # Add horizontal lines at 30 and 70
                fig.add_shape(
                    type="line", x0=stock_data.index[0], x1=stock_data.index[-1], y0=30, y1=30,
                    line=dict(color="Red", dash="dash"),
                    name="Lower Limit"
                )
                fig.add_shape(
                    type="line", x0=stock_data.index[0], x1=stock_data.index[-1], y0=70, y1=70,
                    line=dict(color="Green", dash="dash"),
                    name="Upper Limit"
                )

                # Update layout
                fig.update_layout(
                    title=f"RSI for {stock_name}",
                    xaxis_title="Date",
                    yaxis_title="RSI",
                    shapes=[
                        dict(type='line', yref='y', y0=30, y1=30, xref='x', x0=stock_data.index[0],
                             x1=stock_data.index[-1], line=dict(color='red', dash='dash')),
                        dict(type='line', yref='y', y0=70, y1=70, xref='x', x0=stock_data.index[0],
                             x1=stock_data.index[-1], line=dict(color='green', dash='dash'))
                    ]
                )

                # Display the chart
                st.plotly_chart(fig)

                # Calculate and display MACD
                stock_data['MACD'], stock_data['MACD_signal'] = macd(stock_data)
                st.subheader("Moving Average Convergence Divergence (MACD)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name="MACD"))
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['MACD_signal'], mode='lines', name="MACD Signal"))
                fig.update_layout(title=f"MACD for {stock_name}", xaxis_title="Date", yaxis_title="MACD")
                st.plotly_chart(fig)

                # Calculate and display ROC
                stock_data['ROC'] = roc(stock_data['Close'])
                st.subheader("Rate of Change (ROC)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['ROC'], mode='lines', name="ROC"))
                fig.update_layout(title=f"ROC for {stock_name}", xaxis_title="Date", yaxis_title="ROC")
                st.plotly_chart(fig)

                # -- Future Prediction (Basic Linear Regression) --
                st.header(f"Future Prediction for {stock_name}")

                # Use last 60 days' closing prices for prediction
                prediction_days = 60
                stock_data['Prediction'] = stock_data['Close'].shift(-prediction_days)

                # Create the independent data set (X)
                X = np.array(stock_data[['Close']])
                X = X[:-prediction_days]

                # Create the dependent data set (y)
                y = np.array(stock_data['Prediction'])
                y = y[:-prediction_days]

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):], y[
                                                                                                 :int(0.8 * len(y))], y[
                                                                                                                      int(0.8 * len(
                                                                                                                          y)):]

                # Create and train the Linear Regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Testing the model's predictions
                predictions = model.predict(X_test)

                # Plot the predictions
                st.subheader("Future Stock Price Prediction")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=stock_data.index[-len(X_test):], y=stock_data['Close'][-len(X_test):], mode='lines',
                               name="Real Prices"))
                fig.add_trace(
                    go.Scatter(x=stock_data.index[-len(X_test):], y=predictions, mode='lines', name="Predicted Prices"))
                fig.update_layout(title=f"Stock Price Prediction for {stock_name}", xaxis_title="Date",
                                  yaxis_title="Price")
                st.plotly_chart(fig)

                st.success(f"Analysis Complete for {stock_name}")
