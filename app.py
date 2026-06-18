import streamlit as st
import pandas as pd
import joblib

# 1. Set up the page layout
st.set_page_config(page_title="Commission Drag Predictor", layout="centered")
st.title("📊 Mutual Fund Commission Predictor")
st.write("Enter the structural and momentum details of a mutual fund to predict the expense ratio gap between its Direct and Regular plans.")

# 2. Load the exported AI model and features
@st.cache_resource # This caches the model so it doesn't reload on every button click
def load_assets():
    model = joblib.load('data/xgboost_commission_model.pkl')
    features = joblib.load('data/model_features.pkl')
    fund_houses = joblib.load('data/fund_houses.pkl')
    return model, features, fund_houses

model, features_to_use, fund_houses_list = load_assets()

# 3. Build the UI Input Fields
st.subheader("Fund Characteristics")
col1, col2 = st.columns(2)

with col1:
    # Dropdown menu for the AMC
    selected_amc = st.selectbox("Asset Management Company (AMC)", fund_houses_list)
    
    # Structural features
    days_alive = st.number_input("Days Since Launch", min_value=0, value=1500, step=100)
    risk_90d = st.number_input("90-Day Volatility (Risk %)", min_value=0.0, value=2.5, step=0.1)

with col2:
    # Momentum features
    ret_7d = st.number_input("7-Day Momentum (%)", value=1.5, step=0.5)
    ret_30d = st.number_input("30-Day Momentum (%)", value=4.0, step=0.5)
    ret_90d = st.number_input("90-Day Momentum (%)", value=10.0, step=0.5)

# 4. The Prediction Button
if st.button("Predict Commission Drag", type="primary"):
    
    # Create a dictionary starting with our base numeric features
    input_data = {
        'return_7d': ret_7d,
        'return_30d': ret_30d,
        'return_90d': ret_90d,
        'days_since_launch': days_alive,
        'risk_90d': risk_90d
    }
    
    # TRANSLATION TRICK: Loop through the required columns.
    # If the column is an AMC column, assign 1 if it matches the dropdown, otherwise 0.
    for col in features_to_use:
        if 'fund_house_' in col:
            if col == f"fund_house_{selected_amc}":
                input_data[col] = 1
            else:
                input_data[col] = 0
                
    # Convert the dictionary into a 1-row Pandas DataFrame using the exact column order
    input_df = pd.DataFrame([input_data])[features_to_use]
    
    # Run the AI Prediction
    prediction = model.predict(input_df)[0]
    
    # Display the result!
    st.divider()
    st.success(f"### Predicted Drag: {prediction:.3f}%")
    st.info("This is the structural premium the AMC is likely charging for the regular plan.")
