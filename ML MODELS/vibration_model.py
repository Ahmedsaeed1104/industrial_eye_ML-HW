import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os

FIREBASE_PROJECT_ID = "graduation2-75f83"
FIREBASE_API_KEY = "AIzaSyC6tyMTiIbeznYYfPAaEka2d_9IIe28tw4"
FIREBASE_DATABASE_URL = f"https://{FIREBASE_PROJECT_ID}-default-rtdb.firebaseio.com"

SUPABASE_URL = "https://oyjcytbsblgcgnktlnbh.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im95amN5dGJzYmxnY2dua3RsbmJoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ4NDY0MzcsImV4cCI6MjA4MDQyMjQzN30.Ws7JQliOynykTy10Gow181qu6jsEpfCtM8x2I_adfis"
SUPABASE_BUCKET = "vibration_learn"

DEFAULT_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.15,
    'threshold_percentile': 95,
    'bottleneck_dim': 2,
    'hidden_dim': 4,
    'dropout_rate': 0.1,
    'learning_rate': 0.001,
    'use_data_augmentation': True,
    'augmentation_noise': 0.01,
    'use_early_stopping': True,
    'early_stopping_patience': 10,
}

def get_data_path_from_firestore():
    try:
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/learn_phase/data?key={FIREBASE_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        fields = data.get('fields', {})
        vibration_name = fields.get('vibration_name', {}).get('stringValue', 'vibration_data')
        phase_name = fields.get('phase_name', {}).get('stringValue', 'phase_1')
        data_path = f"{vibration_name}_{phase_name}"
        print(f"Data path from Firestore: {data_path}")
        return data_path
    except Exception as e:
        print(f"Error getting data path: {e}")
        return "vibrationSensorDataFeatures"


class VibrationAnomalyDetector:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.input_dim = 5
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.training_metrics = {}
        
    def load_data_from_firebase(self, data_path):
        all_features = []
        try:
            url = f"{FIREBASE_DATABASE_URL}/{data_path}.json?auth={FIREBASE_API_KEY}"
            print(f"Loading data from: {data_path}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data is None:
                print("No data found in Firebase")
                return np.array([])
            print(f"Number of entries: {len(data)}")
            for key, entry in data.items():
                try:
                    feature_vector = [float(entry['vibMagnitude']), float(entry['vibRMS_X']), float(entry['vibRMS_Y']), float(entry['vibRMS_Z']), float(entry['vibStdDev'])]
                    all_features.append(feature_vector)
                except (KeyError, ValueError):
                    continue
            if len(all_features) > 0:
                print(f"First sample: {all_features[0]}")
        except Exception as e:
            print(f"Error loading from Firebase: {e}")
            return np.array([])
        print(f"Successfully loaded {len(all_features)} samples")
        return np.array(all_features)

    def augment_data(self, data):
        if not self.config['use_data_augmentation']:
            return data
        noise_level = self.config['augmentation_noise']
        augmented = []
        for sample in data:
            augmented.append(sample)
            noisy = sample + np.random.normal(0, noise_level, sample.shape)
            augmented.append(noisy)
        print(f"Data augmented: {len(data)} -> {len(augmented)} samples")
        return np.array(augmented)
    
    def preprocess_data(self, data):
        return self.scaler.fit_transform(data)
    
    def build_autoencoder(self):
        hidden_dim = self.config['hidden_dim']
        bottleneck_dim = self.config['bottleneck_dim']
        dropout_rate = self.config['dropout_rate']
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(hidden_dim, activation='relu')(input_layer)
        encoded = layers.Dropout(dropout_rate)(encoded)
        encoded = layers.Dense(bottleneck_dim, activation='relu')(encoded)
        decoded = layers.Dense(hidden_dim, activation='relu')(encoded)
        decoded = layers.Dropout(dropout_rate)(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        autoencoder = keras.Model(input_layer, decoded)
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        autoencoder.compile(optimizer=optimizer, loss='mse')
        self.model = autoencoder
        return autoencoder

    def train(self, normal_data):
        if len(normal_data) == 0:
            print("No data to train on!")
            return None
        augmented_data = self.augment_data(normal_data)
        print(f"Training data shape: {augmented_data.shape}")
        self.processed_data = self.preprocess_data(augmented_data)
        callbacks = []
        if self.config['use_early_stopping']:
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config['early_stopping_patience'], restore_best_weights=True)
            callbacks.append(early_stopping)
            print(f"Early stopping enabled with patience={self.config['early_stopping_patience']}")
        else:
            print("Early stopping disabled - will train for all epochs")
        history = self.model.fit(self.processed_data, self.processed_data, epochs=self.config['epochs'], batch_size=self.config['batch_size'], validation_split=self.config['validation_split'], shuffle=True, verbose=1, callbacks=callbacks if callbacks else None)
        reconstructions = self.model.predict(self.processed_data, verbose=0)
        mse = np.mean(np.power(self.processed_data - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, self.config['threshold_percentile'])
        self._calculate_metrics(self.processed_data, reconstructions, mse, history)
        print(f"\nTraining completed!")
        print(f"Anomaly threshold: {self.threshold:.6f}")
        self._print_metrics()
        return history

    def _calculate_metrics(self, data, reconstructions, mse, history):
        reconstruction_error = np.mean(mse)
        max_possible_error = np.var(data)
        reconstruction_accuracy = max(0, 1 - (reconstruction_error / max_possible_error)) * 100
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        best_val_loss = min(history.history['val_loss'])
        predictions = mse > self.threshold
        normal_detected_as_normal = np.sum(~predictions) / len(predictions) * 100
        self.training_metrics = {'reconstruction_accuracy': reconstruction_accuracy, 'final_training_loss': final_loss, 'final_validation_loss': final_val_loss, 'best_validation_loss': best_val_loss, 'normal_detection_rate': normal_detected_as_normal, 'threshold': self.threshold, 'total_samples': len(data), 'epochs_trained': len(history.history['loss']), 'mean_reconstruction_error': reconstruction_error, 'config': self.config}
    
    def _print_metrics(self):
        m = self.training_metrics
        print("\n" + "="*50)
        print("MODEL ACCURACY METRICS")
        print("="*50)
        print(f"Reconstruction Accuracy: {m['reconstruction_accuracy']:.2f}%")
        print(f"Normal Detection Rate: {m['normal_detection_rate']:.2f}%")
        print(f"Final Training Loss: {m['final_training_loss']:.6f}")
        print(f"Final Validation Loss: {m['final_validation_loss']:.6f}")
        print(f"Best Validation Loss: {m['best_validation_loss']:.6f}")
        print(f"Mean Reconstruction Error: {m['mean_reconstruction_error']:.6f}")
        print(f"Epochs Trained: {m['epochs_trained']}")
        print(f"Total Samples: {m['total_samples']}")
        print("="*50 + "\n")

    def detect_anomaly(self, new_data):
        processed_data = self.scaler.transform([new_data])
        reconstruction = self.model.predict(processed_data, verbose=0)
        mse = np.mean(np.power(processed_data - reconstruction, 2))
        return mse > self.threshold, mse
    
    def convert_to_tflite(self, output_path='vibration_anomaly_detector.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"Model converted to TensorFlow Lite: {output_path}")
        return output_path
    
    def save_params(self, output_path='vibration_params.json'):
        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'var_'):
            params = {'mean': self.scaler.mean_.tolist(), 'std': np.sqrt(self.scaler.var_).tolist(), 'threshold': float(self.threshold), 'metrics': self.training_metrics, 'config': self.config}
            with open(output_path, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"Parameters saved: {output_path}")
            return output_path
        return None

    def generate_visualizations(self, history, output_dir='vibration_visualizations'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        reconstructions = self.model.predict(self.processed_data, verbose=0)
        mse = np.mean(np.power(self.processed_data - reconstructions, 2), axis=1)
        feature_names = ['VibMagnitude', 'VibRMS_X', 'VibRMS_Y', 'VibRMS_Z', 'VibStdDev']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#9C27B0')
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#E91E63')
        axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        loss_diff = np.array(history.history['val_loss']) - np.array(history.history['loss'])
        axes[1].plot(loss_diff, linewidth=2, color='#673AB7')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1].fill_between(range(len(loss_diff)), loss_diff, 0, where=(loss_diff > 0), alpha=0.3, color='red', label='Overfitting')
        axes[1].fill_between(range(len(loss_diff)), loss_diff, 0, where=(loss_diff <= 0), alpha=0.3, color='green', label='Good fit')
        axes[1].set_title('Validation - Training Loss Gap', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
        plt.show()
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(mse, bins=50, color='#9C27B0', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({self.threshold:.4f})')
        axes[0].set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Mean Squared Error')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        bp = axes[1].boxplot(mse, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#9C27B0')
        axes[1].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold')
        axes[1].set_title('Error Distribution Box Plot', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150)
        plt.show()
        plt.close()

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes[0].plot(mse, linewidth=1, color='#9C27B0', alpha=0.7)
        axes[0].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Anomaly Threshold')
        axes[0].fill_between(range(len(mse)), mse, self.threshold, where=(mse > self.threshold), alpha=0.3, color='red', label='Anomalies')
        axes[0].set_title('Reconstruction Error Timeline', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        window = min(50, len(mse) // 10)
        if window > 1:
            moving_avg = np.convolve(mse, np.ones(window)/window, mode='valid')
            axes[1].plot(moving_avg, linewidth=2, color='#4CAF50', label=f'Moving Average (window={window})')
            axes[1].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2)
            axes[1].set_title('Smoothed Error Trend', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_timeline.png'), dpi=150)
        plt.show()
        plt.close()

        percentile_95 = np.percentile(mse, 95)
        y_true = (mse > percentile_95).astype(int)
        fpr, tpr, _ = roc_curve(y_true, mse)
        roc_auc = auc(fpr, tpr)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(fpr, tpr, color='#9C27B0', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
        axes[0].fill_between(fpr, tpr, alpha=0.3, color='#9C27B0')
        axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        precisions, recalls, f1_scores = [], [], []
        threshold_range = np.linspace(np.min(mse), np.max(mse), 100)
        for thresh in threshold_range:
            pred = (mse > thresh).astype(int)
            tp = np.sum((pred == 1) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            fn = np.sum((pred == 0) & (y_true == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        axes[1].plot(threshold_range, precisions, label='Precision', linewidth=2, color='#4CAF50')
        axes[1].plot(threshold_range, recalls, label='Recall', linewidth=2, color='#FF9800')
        axes[1].plot(threshold_range, f1_scores, label='F1-Score', linewidth=2, color='#E91E63')
        axes[1].axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, label='Current Threshold')
        axes[1].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        plt.show()
        plt.close()

        y_pred = (mse > self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0], xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot_kws={'size': 14})
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1], xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot_kws={'size': 14})
        axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.show()
        plt.close()

        idx = 0
        x = np.arange(len(feature_names))
        width = 0.35
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes[0, 0].bar(x - width/2, self.processed_data[idx], width, label='Original', color='#9C27B0', alpha=0.8)
        axes[0, 0].bar(x + width/2, reconstructions[idx], width, label='Reconstructed', color='#E91E63', alpha=0.8)
        axes[0, 0].set_title(f'Sample {idx}: Original vs Reconstructed', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].scatter(self.processed_data.flatten(), reconstructions.flatten(), alpha=0.1, s=1, color='#9C27B0')
        min_val = min(self.processed_data.min(), reconstructions.min())
        max_val = max(self.processed_data.max(), reconstructions.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Reconstruction')
        axes[0, 1].set_title('Original vs Reconstructed (All Data)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Original Values')
        axes[0, 1].set_ylabel('Reconstructed Values')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        errors_per_feature = np.mean(np.abs(self.processed_data - reconstructions), axis=0)
        colors = plt.cm.RdYlGn_r(errors_per_feature / errors_per_feature.max())
        axes[1, 0].bar(feature_names, errors_per_feature, color=colors)
        axes[1, 0].set_title('Mean Absolute Error per Feature', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        errors = np.abs(self.processed_data - reconstructions)
        axes[1, 1].boxplot([errors[:, i] for i in range(len(feature_names))], labels=feature_names, patch_artist=True)
        axes[1, 1].set_title('Error Distribution per Feature', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstruction_comparison.png'), dpi=150)
        plt.show()
        plt.close()

        errors = np.abs(self.processed_data - reconstructions)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(errors[:100].T, ax=axes[0], cmap='YlOrRd', yticklabels=feature_names, cbar_kws={'label': 'Absolute Error'})
        axes[0].set_title('Error Heatmap (First 100 Samples)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Sample Index')
        error_corr = np.corrcoef(errors.T)
        sns.heatmap(error_corr, ax=axes[1], annot=True, fmt='.2f', cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names, center=0)
        axes[1].set_title('Error Correlation Between Features', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), dpi=150)
        plt.show()
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(0, 10)
        axes[0].axis('off')
        layers_info = [('Input\n(5 features)', 1, '#F3E5F5'), (f'Dense\n({self.config["hidden_dim"]} units)', 3, '#E1BEE7'), (f'Dropout\n({self.config["dropout_rate"]})', 4, '#CE93D8'), (f'Bottleneck\n({self.config["bottleneck_dim"]} units)', 5, '#BA68C8'), (f'Dense\n({self.config["hidden_dim"]} units)', 6, '#E1BEE7'), (f'Dropout\n({self.config["dropout_rate"]})', 7, '#CE93D8'), ('Output\n(5 features)', 9, '#F3E5F5')]
        for i, (name, y, color) in enumerate(layers_info):
            rect = plt.Rectangle((3, y-0.4), 4, 0.8, facecolor=color, edgecolor='black', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].text(5, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
            if i < len(layers_info) - 1:
                axes[0].annotate('', xy=(5, layers_info[i+1][1]-0.4), xytext=(5, y+0.4), arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        axes[0].set_title('Autoencoder Architecture', fontsize=14, fontweight='bold', pad=20)
        axes[1].axis('off')
        m = self.training_metrics
        metrics_text = f"""
╔══════════════════════════════════════════╗
║       VIBRATION MODEL METRICS             ║
╠══════════════════════════════════════════╣
║  Reconstruction Accuracy:  {m['reconstruction_accuracy']:>10.2f}%  ║
║  Normal Detection Rate:    {m['normal_detection_rate']:>10.2f}%  ║
║  Final Training Loss:      {m['final_training_loss']:>10.6f}   ║
║  Final Validation Loss:    {m['final_validation_loss']:>10.6f}   ║
║  Best Validation Loss:     {m['best_validation_loss']:>10.6f}   ║
║  Anomaly Threshold:        {m['threshold']:>10.6f}   ║
║  Total Samples:            {m['total_samples']:>10d}   ║
║  Epochs Trained:           {m['epochs_trained']:>10d}   ║
╚══════════════════════════════════════════╝
"""
        axes[1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center', transform=axes[1].transAxes, bbox=dict(boxstyle='round', facecolor='#f3e5f5', edgecolor='gray'))
        axes[1].set_title('Model Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_summary.png'), dpi=150)
        plt.show()
        plt.close()
        print(f"\nAll visualizations saved to: {output_dir}/")


def upload_to_supabase(file_path, file_name):
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{file_name}"
        headers = {"Authorization": f"Bearer {SUPABASE_ANON_KEY}", "apikey": SUPABASE_ANON_KEY, "x-upsert": "true"}
        if file_name.endswith('.json'):
            headers["Content-Type"] = 'application/json'
        elif file_name.endswith('.png'):
            headers["Content-Type"] = 'image/png'
        else:
            headers["Content-Type"] = 'application/octet-stream'
        with open(file_path, 'rb') as f:
            file_data = f.read()
        response = requests.post(url, headers=headers, data=file_data, timeout=120)
        if response.status_code in [200, 201]:
            print(f"Uploaded {file_name} to Supabase")
            return True
        else:
            print(f"Failed to upload {file_name}: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error uploading {file_name}: {e}")
        return False


def get_config_from_firestore():
    try:
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/vibration/config?key={FIREBASE_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print("No custom config found, using defaults")
            return {}
        data = response.json()
        fields = data.get('fields', {})
        config = {}
        if 'epochs' in fields:
            config['epochs'] = int(fields['epochs'].get('integerValue', 100))
        if 'batch_size' in fields:
            config['batch_size'] = int(fields['batch_size'].get('integerValue', 32))
        if 'threshold_percentile' in fields:
            config['threshold_percentile'] = int(fields['threshold_percentile'].get('integerValue', 95))
        if 'bottleneck_dim' in fields:
            config['bottleneck_dim'] = int(fields['bottleneck_dim'].get('integerValue', 2))
        if 'hidden_dim' in fields:
            config['hidden_dim'] = int(fields['hidden_dim'].get('integerValue', 4))
        if 'dropout_rate' in fields:
            config['dropout_rate'] = float(fields['dropout_rate'].get('doubleValue', 0.1))
        if 'learning_rate' in fields:
            config['learning_rate'] = float(fields['learning_rate'].get('doubleValue', 0.001))
        if 'use_data_augmentation' in fields:
            config['use_data_augmentation'] = fields['use_data_augmentation'].get('booleanValue', True)
        if 'use_early_stopping' in fields:
            config['use_early_stopping'] = fields['use_early_stopping'].get('booleanValue', True)
        if 'early_stopping_patience' in fields:
            config['early_stopping_patience'] = int(fields['early_stopping_patience'].get('integerValue', 10))
        print(f"Loaded config from Firebase: {config}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


if __name__ == "__main__":
    print("="*60)
    print("VIBRATION ANOMALY DETECTION MODEL - KAGGLE TRAINING")
    print("="*60)
    
    config = get_config_from_firestore()
    final_config = {**DEFAULT_CONFIG, **config}
    
    print("\nConfiguration:")
    for k, v in final_config.items():
        print(f"  {k}: {v}")
    
    detector = VibrationAnomalyDetector(config=final_config)
    detector.build_autoencoder()
    
    data_path = get_data_path_from_firestore()
    print(f"\nLoading data from Firebase: {data_path}")
    training_data = detector.load_data_from_firebase(data_path)
    
    if len(training_data) == 0:
        print("ERROR: No training data found!")
    else:
        print(f"\nStarting training with {len(training_data)} samples...")
        history = detector.train(training_data)
        
        if history:
            print("\nGenerating visualizations...")
            detector.generate_visualizations(history)
            
            print("\nSaving model files...")
            detector.convert_to_tflite('vibration_anomaly_detector.tflite')
            detector.save_params('vibration_params.json')
            
            print("\nUploading to Supabase...")
            upload_to_supabase('vibration_anomaly_detector.tflite', 'vibration_anomaly_detector.tflite')
            upload_to_supabase('vibration_params.json', 'vibration_params.json')
            
            viz_dir = 'vibration_visualizations'
            for viz_file in os.listdir(viz_dir):
                if viz_file.endswith('.png'):
                    upload_to_supabase(os.path.join(viz_dir, viz_file), f'visualizations/{viz_file}')
            
            print("\n" + "="*60)
            print("TRAINING COMPLETE!")
            print("="*60)
