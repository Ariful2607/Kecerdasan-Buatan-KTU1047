import joblib
import numpy as np

def load_model():
    scaler = joblib.load('scaler.pkl')
    centroids = np.load('centroids_optimized.npy')
    return scaler, centroids

def predict_cluster(sample, scaler, centroids):
    """
    sample: array 1D fitur input
    scaler: scaler yang sudah diload
    centroids: centroid hasil clustering (float16)
    """
    # Standarisasi data input
    sample_scaled = scaler.transform(sample.reshape(1, -1))
    # Hitung jarak Euclidean ke setiap centroid
    distances = np.linalg.norm(centroids - sample_scaled, axis=1)
    # Tentukan cluster terdekat
    cluster_id = np.argmin(distances)
    return cluster_id

# Contoh penggunaan inference
if __name__ == "__main__":
    scaler_loaded, centroids_loaded = load_model()
    sample_data = np.array([0.5, 0.2, 0.1, 0.7, 0.3])
    cluster_pred = predict_cluster(sample_data, scaler_loaded, centroids_loaded)
    print(f"Sample data masuk ke cluster: {cluster_pred}")
