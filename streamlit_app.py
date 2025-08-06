import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(page_title="Defect Rate Predictor", layout="wide")
st.title("🏭 Smartphone Plastic Defect Rate Predictor")

# -------------------------------
# Sidebar Input Form
# -------------------------------
st.sidebar.header("Production Parameters")

# Collect production parameters from user via sliders/select boxes
temp = st.sidebar.slider("Temperature (°C)", 150, 250, 215)
pressure = st.sidebar.slider("Pressure (Pa)", 3000, 8000, 6500)
downtime = st.sidebar.slider("Downtime (Minutes)", 0, 200, 60)
quality = st.sidebar.slider("Raw Material Quality (Score)", 60, 100, 85)
cooling = st.sidebar.slider("Cooling Rate (°C/min)", 1, 10, 5)
speed = st.sidebar.slider("Machine Speed (RPM)", 1000, 2000, 1500)
humidity = st.sidebar.slider("Humidity (%)", 20, 80, 55)
ambient = st.sidebar.slider("Ambient Temp (°C)", 15, 35, 25)
maintenance = st.sidebar.slider("Maintenance (Days Since)", 0, 100, 30)
batch = st.sidebar.slider("Batch Size (Units)", 50, 200, 100)
energy = st.sidebar.slider("Energy Consumption (kWh)", 30, 100, 55)
shift = st.sidebar.selectbox("Operator Shift", ("Night", "Day"))
line = st.sidebar.selectbox("Production Line", ("Line 1", "Line 2", "Line 3"))


# -------------------------------
# Prediction Trigger & Workflow
# -------------------------------
if st.sidebar.button("Predict Defect Rate", type="primary"):
    """
    Trigger the prediction pipeline when the button is clicked:
    - Collects all user inputs into a structured format.
    - Transforms input data using the CustomData class.
    - Runs the prediction using the trained model via PredictPipeline.
    - Displays the predicted defect rate in the main panel.
    """
    # Step 1: Create an instance of CustomData using user inputs
    data = CustomData(
        **{
            "Temperature (°C)": temp,
            "Pressure (Pa)": pressure,
            "Cooling Rate (°C/min)": cooling,
            "Machine Speed (RPM)": speed,
            "Raw Material Quality (Score)": quality,
            "Humidity (%)": humidity,
            "Ambient Temperature (°C)": ambient,
            "Maintenance (Days Since)": maintenance,
            "Operator Shift": shift,
            "Batch Size (Units)": batch,
            "Energy Consumption (kWh)": energy,
            "Downtime (Minutes)": downtime,
            "Production Line": line
        }
    )
    
    # Step 2: Convert the custom data object to a pandas DataFrame
    data_df = data.get_data_as_dataframe()
    
    # Step 3: Load the prediction pipeline and make the prediction
    pipeline = PredictPipeline()
    prediction = pipeline.predict(data_df)[0]
    
    # Step 4: Display the prediction result in the UI
    st.subheader("Prediction Result")
    st.metric(label="Predicted Defect Rate", value=f"{prediction:.2f}%")