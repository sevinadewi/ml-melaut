from keras.models import load_model
import joblib
import pandas as pd
from google.cloud import storage
import numpy as np
import os


# Konfigurasi bucket GCS
MODEL_BUCKET_NAME = "bangkit-bucket-melaut"
MODEL_FOLDER = "models/"

# Path sementara untuk menyimpan model dan scaler
classification_model_path = "/tmp/classification_model.h5"
regression_model_path = "/tmp/deep_learning_regression_model.h5"
classification_scaler_path = "/tmp/scaler_classification.pkl"
regression_scaler_path = "/tmp/scaler_deep_learning_regression.pkl"

# Fungsi untuk mengunduh file dari GCS
def download_file_from_gcs(bucket_name, blob_name, destination_file_name):
    try:
        # Inisialisasi client GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download file ke direktori sementara
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {blob_name} to {destination_file_name}")
    except Exception as e:
        print(f"Error saat mengunduh file {blob_name}: {e}")
        raise

# Unduh model dan scaler dari GCS
try:
    # Download model dan scaler dari GCS
    download_file_from_gcs(MODEL_BUCKET_NAME, MODEL_FOLDER + "classification_model.h5", classification_model_path)
    download_file_from_gcs(MODEL_BUCKET_NAME, MODEL_FOLDER + "deep_learning_regression_model.h5", regression_model_path)
    download_file_from_gcs(MODEL_BUCKET_NAME, MODEL_FOLDER + "scaler_classification.pkl", classification_scaler_path)
    download_file_from_gcs(MODEL_BUCKET_NAME, MODEL_FOLDER + "scaler_deep_learning_regression.pkl", regression_scaler_path)
    
    # Memuat model dan scaler
    classification_model = load_model(classification_model_path)
    classification_scaler = joblib.load(classification_scaler_path)
    regression_model = load_model(regression_model_path)
    regression_scaler = joblib.load(regression_scaler_path)
    
    
    
    print("Model dan scaler berhasil di-load.")
except Exception as e:
    print(f"Error saat me-load model atau scaler: {e}")
    raise  # Raise the exception untuk menghentikan aplikasi

# Fungsi predict_rad()
def predict_rad(features, scaler, model):
    """
    Prediksi nilai rad(m) berdasarkan fitur cuaca input.

    Parameters:
        features (dict): Dictionary berisi nilai semua fitur (Tn, Tx, Tavg, RH_avg, RR, ss, ff_x, ddd_x, ff_avg).
        scaler (StandardScaler): Scaler yang digunakan untuk normalisasi fitur.
        model (Sequential): Model deep learning yang sudah dilatih.

    Returns:
        float: Nilai prediksi rad(m).
    """
    try:
        # Konversi input features ke DataFrame dengan nama kolom yang sama dengan yang diharapkan scaler
        feature_df = pd.DataFrame([features])

        # Penskalaan fitur input menggunakan scaler yang sudah dilatih
        scaled_features = scaler.transform(feature_df)

        # Prediksi rad(m) menggunakan model yang sudah dilatih
        predicted_value = model.predict(scaled_features)

        # Kembalikan nilai prediksi rad(m)
        return predicted_value[0][0]
    except Exception as e:
        print(f"Error dalam prediksi rad(m): {e}")
        raise

# Fungsi predict_condition()
def predict_condition(input_features, scaler, model):
    try:
        # Data input baru
        new_data = np.array([[input_features['Tn'], input_features['Tx'], input_features['Tavg'], 
                              input_features['RH_avg'], input_features['ff_avg'], input_features['rad_m']]])

        # Normalisasi data baru menggunakan scaler yang sama seperti yang digunakan saat pelatihan
        new_data_scaled = scaler.transform(new_data)

        # Reshape data untuk input LSTM (3D tensor)
        new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

        # Gunakan model yang dimuat untuk prediksi
        prediction = model.predict(new_data_reshaped)

        # Tampilkan hasil prediksi (nilai probabilitas, 0 atau 1)
        predicted_label = (prediction > 0.5).astype(int)
        return 'Aman' if predicted_label[0][0] == 0 else 'Tidak Aman'
    except Exception as e:
        print(f"Error dalam prediksi kondisi: {e}")
        raise 