import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Load data from Excel file
@st.cache_data
def load_data():
    df = pd.read_excel("FinData.xlsx")
    return df


df = load_data()

# Polynomial regression model for Contributions Collected
X = df[["Year"]]
y = df["Contributions Collected"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Streamlit UI
st.title("Predicting Contributions Collected for Financial Sustainability")

st.markdown(
    """
This app uses historical financial data to **predict Contributions Collected** from 2025 onwards using a polynomial regression model.  
The goal is to ensure the *Sustainability Ratio* improves steadily by growing contributions at an appropriate rate.
"""
)

# For display only, convert Year to string in a copy
display_hist_df = df.copy()
display_hist_df["Year"] = display_hist_df["Year"].astype(str)

st.subheader("Historical Data Snapshot")
st.write(display_hist_df)

# User input for prediction year range
start_year = st.number_input(
    "Start Year for Prediction", min_value=2025, max_value=2050, value=2025
)
end_year = st.number_input(
    "End Year for Prediction", min_value=2026, max_value=2050, value=2030
)

# Validate end year > start year
if end_year <= start_year:
    st.error("End Year must be greater than Start Year.")
else:
    future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
    future_years_poly = poly.transform(future_years)
    predicted_contributions = model.predict(future_years_poly)

    # Extrapolate Total Outflows growth rate from historical data
    historical_outflows = df["Total Outflows"].values
    outflow_growth_rates = (historical_outflows[1:] / historical_outflows[:-1]) - 1
    avg_outflow_growth = np.mean(outflow_growth_rates)

    last_outflow = historical_outflows[-1]
    predicted_outflows = []
    outflow = last_outflow
    for _ in future_years.flatten():
        outflow = outflow * (1 + avg_outflow_growth)
        predicted_outflows.append(outflow)

    # Estimate Total Inflows = Predicted Contributions + avg other inflows
    avg_other_inflows = np.mean(df["Net Investment Income"] + df["Other Income"])
    predicted_total_inflows = predicted_contributions + avg_other_inflows

    # Calculate predicted Sustainability Ratio
    predicted_sustainability_ratio = predicted_total_inflows / np.array(
        predicted_outflows
    )

    # Combine predicted contributions and predicted Sustainability Ratio in one table
    pred_df = pd.DataFrame(
        {
            "Year": future_years.flatten(),
            "Predicted Contributions Collected": predicted_contributions,
            "Predicted Sustainability Ratio": predicted_sustainability_ratio,
        }
    )

    # For display only, convert Year to string in a copy
    display_df = pred_df.copy()
    display_df["Year"] = display_df["Year"].astype(str)

    st.subheader("Predicted Contributions and Sustainability Ratio")
    st.write(display_df)

    # Visualize predictions
    fig, ax = plt.subplots()
    ax.plot(
        df["Year"],
        df["Contributions Collected"],
        "o-",
        label="Historical Contributions",
    )
    ax.plot(
        pred_df["Year"],
        pred_df["Predicted Contributions Collected"],
        "s--",
        label="Predicted Contributions",
    )
    ax.set_ylabel("Contributions Collected")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown(
        """
    **Explanation:**  
    - The model fits past contributions with a quadratic curve, capturing accelerating growth.  
    - Predictions show projected yearly contributions needed to keep financial inflows strong.  
    - By increasing contributions as predicted, the Sustainability Ratio is likely to improve, supporting long-term fund viability.  
    - Adjust the year range to explore different forecast periods.
    """
    )

st.markdown(
    "Built with advanced polynomial regression for accurate forecasting based on your uploaded data."
)
