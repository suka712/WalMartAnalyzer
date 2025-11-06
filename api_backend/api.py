import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import traceback
from predictor import WalmartSalesPredictor 

# ==============================================================================
# 1. Initialization
# ==============================================================================

app = Flask(__name__)

# --- Load the self-contained predictor object ---
try:
    # IMPORTANT: Update this path to match the name of your saved model directory
    MODEL_DIR = 'walmart_sales_model_20251027_155246' 
    PREDICTOR_PATH = os.path.join(MODEL_DIR, 'predictor.pkl')
    
    print(f"[*] Loading predictor from: {PREDICTOR_PATH}")
    predictor = joblib.load(PREDICTOR_PATH)
    print(f"[*] Predictor loaded successfully. Model type: {predictor.metadata.get('model_name', 'Unknown')}")

except FileNotFoundError:
    print(f"[ERROR] Predictor file not found at '{PREDICTOR_PATH}'.")
    print("[ERROR] Please ensure the model directory and predictor.pkl exist.")
    predictor = None
except Exception as e:
    print(f"[ERROR] An error occurred while loading the predictor: {e}")
    predictor = None

# ==============================================================================
# 2. API Endpoints
# ==============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to confirm the API is running and the model is loaded.
    """
    if predictor:
        return jsonify({
            "status": "ok",
            "message": "API is running and model is loaded.",
            "model_type": predictor.metadata.get('model_name', 'Unknown'),
            "model_train_date": predictor.metadata.get('train_date', 'Unknown')
        }), 200
    else:
        return jsonify({
            "status": "error",
            "message": "API is running, but the prediction model failed to load."
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint. Expects a JSON list of records.
    """
    if not predictor:
        return jsonify({"error": "Model is not loaded. Cannot make predictions."}), 503

    print("[*] Received a request on /predict endpoint.")

    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No input data provided."}), 400
        
        input_df = pd.DataFrame(json_data)
        print(f"[*] Converted JSON to DataFrame with {len(input_df)} rows.")

        predictions_df = predictor.predict(input_df)

        predictions_json = predictions_df.to_dict(orient='records')
        
        print(f"[*] Successfully generated {len(predictions_json)} predictions.")
        return jsonify(predictions_json), 200

    except Exception as e:
        print(f"[ERROR] An error occurred during prediction: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "An internal error occurred.",
            "message": str(e)
        }), 500

# ==============================================================================
# 3. Main Application Runner
# ==============================================================================

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5000))
    print(f"[*] Starting Flask server on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=True)