import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit App Title
st.title("💰 Personal Finance Expense Forecasting App")

# File path for dataset
file_path = r'C:\Users\Thota Ravindranath\Downloads\personal_finance_tracker_dataset.csv'

# Load dataset
df = pd.read_csv(file_path)
st.write("✅ Dataset loaded successfully!")
st.write("Shape:", df.shape)
st.write("Columns:", df.columns.tolist())

# --- Data Preprocessing ---
date_col = 'date'
expense_col = 'monthly_expense_total'

df_subset = df[[date_col, expense_col, 'user_id']].copy()
df_subset[date_col] = pd.to_datetime(df_subset[date_col])

top_user = df_subset['user_id'].value_counts().idxmax()
df_user = df_subset[df_subset['user_id'] == top_user].copy()
df_user.set_index(date_col, inplace=True)

df_monthly = df_user[[expense_col]].resample('MS').sum().sort_index()
df_monthly = df_monthly.replace(0, np.nan).ffill()

st.subheader(f"Analyzing User ID: {top_user}")
st.line_chart(df_monthly)

# --- Train-Test Split ---
split_point = int(len(df_monthly) * 0.8)
train_data = df_monthly.iloc[:split_point]
test_data = df_monthly.iloc[split_point:]

# --- Train ARIMA Model ---
model = ARIMA(train_data[expense_col], order=(5, 1, 0))
model_fit = model.fit()

# --- Forecast for Test Data ---
forecast_res = model_fit.get_forecast(steps=len(test_data))
predictions = forecast_res.predicted_mean

results = pd.concat([
    test_data[expense_col].rename('Actual'),
    predictions.rename('Predicted')
], axis=1).dropna()

# --- Model Evaluation ---
mae = mean_absolute_error(results['Actual'], results['Predicted'])
mse = mean_squared_error(results['Actual'], results['Predicted'])
rmse = np.sqrt(mse)

st.subheader("📊 Model Evaluation Metrics")
st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")

# --- Visualization ---
st.subheader("📈 Expense Forecast vs Actual Data")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train_data[expense_col], label='Training Data', color='skyblue')
ax.plot(test_data[expense_col], label='Actual Data', color='green')
ax.plot(results['Predicted'], label='Predicted', color='red', linestyle='--')
ax.set_title(f'Expense Forecast vs Actual Data (User {top_user})')
ax.set_xlabel('Date')
ax.set_ylabel('Monthly Expense')
ax.legend()
st.pyplot(fig)

# --- Future Forecast ---
st.subheader("🔮 Future Expense Forecast")

months_to_predict = st.slider("Select number of months to forecast:", 1, 12, 6)
future_forecast = model_fit.forecast(steps=months_to_predict)
future_dates = pd.date_range(df_monthly.index[-1] + pd.offsets.MonthBegin(1),
                             periods=months_to_predict, freq='MS')
future_df = pd.DataFrame({'Predicted_Expense': future_forecast}, index=future_dates)

# --- Add user adjustment ---
extra_expense = st.number_input("Enter any expected additional expense (₹):", value=0.0, step=500.0)
future_df['Adjusted_Expense'] = future_df['Predicted_Expense'] + extra_expense

st.write("### Future Expense Forecast")
st.dataframe(future_df)

st.subheader("📊 Future Forecast Chart")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(future_df.index, future_df['Predicted_Expense'], label='Predicted Expense', color='blue')
ax2.plot(future_df.index, future_df['Adjusted_Expense'], label='Adjusted (with your input)', color='orange', linestyle='--')
ax2.set_xlabel('Date')
ax2.set_ylabel('Expense Amount (₹)')
ax2.legend()
st.pyplot(fig2)

