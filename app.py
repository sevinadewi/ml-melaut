

# from flask import Flask, request, jsonify
# import tensorflow as tf
# import joblib
# from google.cloud import storage

# app = Flask(__name__)

# # Konfigurasi bucket GCS
# MODEL_BUCKET_NAME = "bangkit-bucket-melaut"
# MODEL_FOLDER = "models/"

# # Path untuk model di direktori sementara
# classification_model_path = "/tmp/classification_model.h5"
# regression_model_path = "/tmp/deep_learning_regression_model.h5"
# scaler_classification_path = "/tmp/scaler_classification.pkl"
# scaler_deep_learning_regression_path = "/tmp/scaler_deep_learning_regression.pkl"


# # Unduh file dari Google Cloud Storage
# def download_file_from_gcs(bucket_name, blob_name, destination_file_name):
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(f"Downloaded {blob_name} to {destination_file_name}")


# # Unduh model dan scaler dari GCS dan load
# def initialize_models():
#     try:
#         download_file_from_gcs(MODEL_BUCKET_NAME, "models/classification_model.h5", classification_model_path)
#         download_file_from_gcs(MODEL_BUCKET_NAME, "models/deep_learning_regression_model.h5", regression_model_path)
#         download_file_from_gcs(MODEL_BUCKET_NAME, "models/scaler_classification.pkl", scaler_classification_path)
#         download_file_from_gcs(MODEL_BUCKET_NAME, "models/scaler_deep_learning_regression.pkl", scaler_deep_learning_regression_path)

#         global classification_model, regression_model
#         global scaler_classification, scaler_deep_learning_regression

#         classification_model = tf.keras.models.load_model(classification_model_path)
#         regression_model = tf.keras.models.load_model(regression_model_path)
#         scaler_classification = joblib.load(scaler_classification_path)
#         scaler_deep_learning_regression = joblib.load(scaler_deep_learning_regression_path)

#         print("Models loaded successfully.")
#     except Exception as e:
#         print(f"Error loading models: {e}")


# # Jalankan saat pertama kali server dinyalakan
# initialize_models()


# @app.route('/predict_classification', methods=["POST"])
# def predict_classification():
#     try:
#         data = request.get_json()
#         features = [data["feature_1"], data["feature_2"]]
#         scaled_features = scaler_classification.transform([features])
#         prediction = classification_model.predict(scaled_features)
#         return jsonify({"prediction": prediction.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/predict_regression', methods=["POST"])
# def predict_regression():
#     try:
#         data = request.get_json()
#         features = [data["feature_1"], data["feature_2"]]
#         scaled_features = scaler_deep_learning_regression.transform([features])
#         prediction = regression_model.predict(scaled_features)
#         return jsonify({"prediction": prediction.tolist()})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True, port=8080)

from flask import Flask, request, jsonify
from model import predict_rad, predict_condition
import logging

app = Flask(_name_)

# Konfigurasi logging
logging.basicConfig(filename='api.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s: %(message)s')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validasi input
        required_features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ff_avg']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing input features'}), 400
        
        # Periksa tipe data
        for feature in required_features:
            if not isinstance(data[feature], (int, float)):
                return jsonify({'error': f'Invalid data type for {feature}. Expected int or float'}), 400

        # Prediksi rad(m)
        predicted_rad_value = predict_rad(data, regression_scaler, regression_model)

        # Tambahkan predicted_rad ke data input untuk prediksi kondisi
        data['rad_m'] = predicted_rad_value

        # Prediksi kondisi
        predicted_condition_value = predict_condition(data, classification_scaler, classification_model)

        # Kembalikan hasil prediksi
        result = {
            'predicted_rad': float(predicted_rad_value),
            'condition': predicted_condition_value
        }
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error: {e}")  # Log the error
        return jsonify({'error': 'Internal server error'}), 500

if _name_ == '_main_':
    app.run(debug=True)