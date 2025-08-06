import joblib
import pandas as pd
import traceback
from flask import Flask, request, jsonify, render_template

# Initialize the Flask App
app = Flask(__name__)

# Load the Model and Preprocessor
try:
    model = joblib.load('model/defect_rate_predictor.joblib')
    preprocessor = joblib.load('model/preprocessor.joblib')
    print("✅ Model and preprocessor loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

# Web Page Routes
@app.route('/')
def home():
    """
    Renders the homepage of the application.

    Returns:
        HTML page: index.html from the templates folder.
    """
    return render_template('index.html')

# Define the Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives production data via a POST request, processes it, and returns a defect rate prediction.

    This endpoint is the core of the ML model's API. It expects a JSON payload
    containing a list with a single dictionary of production parameters. It then
    performs the same feature engineering and preprocessing steps used during training
    before making a prediction with the loaded model.

    Returns:
        flask.Response: A JSON response containing either the predicted defect rate
                        or an error message.
    """
    if not model or not preprocessor:
        return jsonify({"error": "Model or preprocessor not loaded"}), 500

    try:
        # 1. Get raw data from the request
        json_data = request.get_json()
        raw_data = pd.DataFrame(json_data)

        # --- Print the columns we received ---
        print(f"--- Columns received from frontend: {list(raw_data.columns)} ---")

        # 2. Engineer features just as training.
        raw_data['Energy_per_Unit'] = raw_data['Energy Consumption (kWh)'] / (raw_data['Batch Size (Units)'] + 1e-6)
        raw_data['Process_Stress_Index'] = raw_data['Temperature (°C)'] * raw_data['Pressure (Pa)']

        # 3. Use the preprocessor to transform the data
        processed_data = preprocessor.transform(raw_data)

        # 4. Convert to DataFrame to find and drop the unneeded columns by name
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out()
        all_feature_names = (
            list(preprocessor.named_transformers_['skewed'].feature_names_in_) +
            list(preprocessor.named_transformers_['symmetric'].feature_names_in_) +
            list(ohe_feature_names)
        )
        processed_df = pd.DataFrame(processed_data, columns=all_feature_names)

        # 5. Drop the columns the final model was NOT trained on
        cols_to_drop = ['Process_Stress_Index', 'Pressure (Pa)']
        final_data = processed_df.drop(columns=cols_to_drop)
        
        # 6. Make the prediction
        prediction = model.predict(final_data)

        # 7. Return the result
        return jsonify({"predicted_defect_rate": prediction.tolist()})

    except Exception as e:
        # Enhanced logging will print the full error to the Docker logs
        print("--- An error occurred during prediction ---")
        traceback.print_exc()
        print("--- End of error ---")
        return jsonify({"error": "An error occurred during prediction. Check logs."}), 500

# Run the Flask App
if __name__ == '__main__':
    """
    Runs the Flask application.

    The host '0.0.0.0' makes the server publicly available, which is necessary
    for it to be accessible from outside the Docker container.
    """
    app.run(host='0.0.0.0', port=5000)