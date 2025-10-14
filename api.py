# Step 1: Make sure all imports are here
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Imports the fix

# Step 2: Initialize the Flask app
app = Flask(__name__)

# Step 3: APPLY THE FIX TO THE ENTIRE APP
# This is the crucial line that allows the browser to communicate with the server.
CORS(app)

# Load the saved model and scaler
try:
    model = pickle.load(open('fake_account_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("ERROR: Could not find model or scaler files. Please run train_model.py")
    exit()


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        feature_keys = [
            'username_length', 'num_posts', 'num_followers', 'num_following', 
            'account_age_days', 'has_profile_picture', 'has_bio', 
            'engagement_ratio', 'is_verified'
        ]
        
        features_list = [data.get(key, 0) for key in feature_keys]
        features = [np.array(features_list)]
        
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)[0]
        
        response = {
            'prediction': int(prediction[0]),
            'is_fake': bool(prediction[0] == 1),
            'probability_fake': probabilities[1],
            'confidence': {
                'real_account': probabilities[0],
                'fake_account': probabilities[1]
            },
            'input_data': data
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=True)