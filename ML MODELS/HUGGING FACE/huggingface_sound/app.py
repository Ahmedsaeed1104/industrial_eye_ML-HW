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
import tempfile
import threading
import time
import gradio as gr

FIREBASE_PROJECT_ID = "graduation2-75f83"
FIREBASE_API_KEY = "AIzaSyC6tyMTiIbeznYYfPAaEka2d_9IIe28tw4"
FIREBASE_DATABASE_URL = f"https://{FIREBASE_PROJECT_ID}-default-rtdb.firebaseio.com"

SUPABASE_URL = "https://oyjcytbsblgcgnktlnbh.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im95amN5dGJzYmxnY2dua3RsbmJoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ4NDY0MzcsImV4cCI6MjA4MDQyMjQzN30.Ws7JQliOynykTy10Gow181qu6jsEpfCtM8x2I_adfis"
SUPABASE_BUCKET = "sound"

DEFAULT_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.15,
    'threshold_percentile': 95,
    'bottleneck_dim': 6,
    'hidden_dim': 12,
    'dropout_rate': 0.1,
    'learning_rate': 0.001,
    'use_data_augmentation': True,
    'augmentation_noise': 0.01,
    'use_early_stopping': True,
    'early_stopping_patience': 10,
}

status_info = {
    "message": "Waiting for learn=true in Firestore...",
    "is_training": False,
    "countdown": 0,
    "last_trained": None,
    "last_check": None,
    "metrics": None,
    "config": DEFAULT_CONFIG.copy(),
    "data_path": None
}


class AudioAnomalyDetector:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.input_dim = 16
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        self.training_metrics = {}
        self.processed_data = None

    def load_data_from_firebase(self, data_path):
        all_features = []
        try:
            url = f"{FIREBASE_DATABASE_URL}/{data_path}.json?auth={FIREBASE_API_KEY}"
            print(f"Loading data from: {data_path}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data is None:
                return np.array([])
            print(f"Number of entries: {len(data)}")
            for key, entry in data.items():
                try:
                    feature_vector = [
                        float(entry['audioMFCC1']), float(entry['audioMFCC2']),
                        float(entry['audioMFCC3']), float(entry['audioMFCC4']),
                        float(entry['audioMFCC5']), float(entry['audioMFCC6']),
                        float(entry['audioMFCC7']), float(entry['audioMFCC8']),
                        float(entry['audioPeak']), float(entry['audioRMS']),
                        float(entry['audioEnergy']), float(entry['audioSpectralCentroid']),
                        float(entry['audioSpectralFlatness']), float(entry['audioSpectralBandwidth']),
                        float(entry['audioSpectralRolloff']), float(entry['audioCrestFactor'])
                    ]
                    all_features.append(feature_vector)
                except (KeyError, ValueError):
                    continue
        except Exception as e:
            print(f"Error loading from Firebase: {e}")
            return np.array([])
        print(f"Successfully loaded {len(all_features)} samples")
        return np.array(all_features)

    def augment_data(self, data):
        if not self.config['use_data_augmentation']:
            return data
        augmented = []
        for sample in data:
            augmented.append(sample)
            noisy = sample + np.random.normal(0, self.config['augmentation_noise'], sample.shape)
            augmented.append(noisy)
        return np.array(augmented)

    def preprocess_data(self, data):
        return self.scaler.fit_transform(data)

    def build_autoencoder(self):
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(self.config['hidden_dim'], activation='relu')(input_layer)
        encoded = layers.Dropout(self.config['dropout_rate'])(encoded)
        encoded = layers.Dense(self.config['bottleneck_dim'], activation='relu')(encoded)
        decoded = layers.Dense(self.config['hidden_dim'], activation='relu')(encoded)
        decoded = layers.Dropout(self.config['dropout_rate'])(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']), loss='mse')
        return self.model

    def train(self, normal_data):
        if len(normal_data) == 0:
            return None
        augmented_data = self.augment_data(normal_data)
        self.processed_data = self.preprocess_data(augmented_data)
        callbacks = []
        if self.config['use_early_stopping']:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.config['early_stopping_patience'], restore_best_weights=True))
        history = self.model.fit(
            self.processed_data, self.processed_data,
            epochs=self.config['epochs'], batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'], shuffle=True, verbose=1,
            callbacks=callbacks if callbacks else None)
        reconstructions = self.model.predict(self.processed_data, verbose=0)
        mse = np.mean(np.power(self.processed_data - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, self.config['threshold_percentile'])
        self._calculate_metrics(self.processed_data, reconstructions, mse, history)
        return history

    def _calculate_metrics(self, data, reconstructions, mse, history):
        reconstruction_error = np.mean(mse)
        max_possible_error = np.var(data)
        reconstruction_accuracy = max(0, 1 - (reconstruction_error / max_possible_error)) * 100
        predictions = mse > self.threshold
        normal_detection_rate = np.sum(~predictions) / len(predictions) * 100
        self.training_metrics = {
            'reconstruction_accuracy': round(reconstruction_accuracy, 2),
            'final_training_loss': round(history.history['loss'][-1], 6),
            'final_validation_loss': round(history.history['val_loss'][-1], 6),
            'best_validation_loss': round(min(history.history['val_loss']), 6),
            'normal_detection_rate': round(normal_detection_rate, 2),
            'threshold': round(self.threshold, 6),
            'total_samples': len(data),
            'epochs_trained': len(history.history['loss']),
            'mean_reconstruction_error': round(reconstruction_error, 6)
        }

    def convert_to_tflite(self, output_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        return output_path

    def save_params(self, output_path, last_trained=None, data_path=None):
        params = {
            'mean': self.scaler.mean_.tolist(),
            'std': np.sqrt(self.scaler.var_).tolist(),
            'threshold': float(self.threshold),
            'metrics': self.training_metrics,
            'config': self.config,
            'last_trained': last_trained,
            'data_path': data_path
        }
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        return output_path

    def generate_visualizations(self, history, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        reconstructions = self.model.predict(self.processed_data, verbose=0)
        mse = np.mean(np.power(self.processed_data - reconstructions, 2), axis=1)
        feature_names = ['MFCC1', 'MFCC2', 'MFCC3', 'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8',
                         'Peak', 'RMS', 'Energy', 'SpectralCentroid', 'SpectralFlatness',
                         'SpectralBandwidth', 'SpectralRolloff', 'CrestFactor']

        # Training History
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#2196F3')
        axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#FF5722')
        axes[0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        loss_diff = np.array(history.history['val_loss']) - np.array(history.history['loss'])
        axes[1].plot(loss_diff, linewidth=2, color='#9C27B0')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1].fill_between(range(len(loss_diff)), loss_diff, 0, where=(loss_diff > 0), alpha=0.3, color='red', label='Overfitting')
        axes[1].fill_between(range(len(loss_diff)), loss_diff, 0, where=(loss_diff <= 0), alpha=0.3, color='green', label='Good fit')
        axes[1].set_title('Validation - Training Loss Gap', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
        plt.close()

        # Error Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(mse, bins=50, color='#2196F3', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({self.threshold:.4f})')
        axes[0].set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        bp = axes[1].boxplot(mse, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('#2196F3')
        axes[1].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[1].set_title('Error Distribution Box Plot', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150)
        plt.close()

        # Error Timeline
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes[0].plot(mse, linewidth=1, color='#2196F3', alpha=0.7)
        axes[0].axhline(y=self.threshold, color='red', linestyle='--', linewidth=2, label='Anomaly Threshold')
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
        plt.close()

        # ROC Curve
        percentile_95 = np.percentile(mse, 95)
        y_true = (mse > percentile_95).astype(int)
        fpr, tpr, _ = roc_curve(y_true, mse)
        roc_auc = auc(fpr, tpr)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(fpr, tpr, color='#2196F3', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
        axes[0].fill_between(fpr, tpr, alpha=0.3, color='#2196F3')
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
        axes[1].plot(threshold_range, f1_scores, label='F1-Score', linewidth=2, color='#9C27B0')
        axes[1].axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, label='Current Threshold')
        axes[1].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        plt.close()

        # Confusion Matrix
        y_pred = (mse > self.threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot_kws={'size': 14})
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'], annot_kws={'size': 14})
        axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

        # Reconstruction Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        idx = 0
        x = np.arange(len(feature_names))
        width = 0.35
        axes[0, 0].bar(x - width/2, self.processed_data[idx], width, label='Original', color='#2196F3', alpha=0.8)
        axes[0, 0].bar(x + width/2, reconstructions[idx], width, label='Reconstructed', color='#FF5722', alpha=0.8)
        axes[0, 0].set_title(f'Sample {idx}: Original vs Reconstructed', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 1].scatter(self.processed_data.flatten(), reconstructions.flatten(), alpha=0.1, s=1, color='#2196F3')
        min_val = min(self.processed_data.min(), reconstructions.min())
        max_val = max(self.processed_data.max(), reconstructions.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Reconstruction')
        axes[0, 1].set_title('Original vs Reconstructed (All Data)', fontsize=12, fontweight='bold')
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
        plt.close()

        # Feature Analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(errors[:100].T, ax=axes[0], cmap='YlOrRd', yticklabels=feature_names, cbar_kws={'label': 'Absolute Error'})
        axes[0].set_title('Error Heatmap (First 100 Samples)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Sample Index')
        error_corr = np.corrcoef(errors.T)
        sns.heatmap(error_corr, ax=axes[1], annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=feature_names, yticklabels=feature_names, center=0)
        axes[1].set_title('Error Correlation Between Features', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_analysis.png'), dpi=150)
        plt.close()

        # Model Summary
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(0, 10)
        axes[0].axis('off')
        layers_info = [
            ('Input\n(16 features)', 1, '#E3F2FD'),
            (f'Dense\n({self.config["hidden_dim"]} units)', 3, '#BBDEFB'),
            (f'Dropout\n({self.config["dropout_rate"]})', 4, '#90CAF9'),
            (f'Bottleneck\n({self.config["bottleneck_dim"]} units)', 5, '#64B5F6'),
            (f'Dense\n({self.config["hidden_dim"]} units)', 6, '#BBDEFB'),
            (f'Dropout\n({self.config["dropout_rate"]})', 7, '#90CAF9'),
            ('Output\n(16 features)', 9, '#E3F2FD')
        ]
        for i, (name, y, color) in enumerate(layers_info):
            rect = plt.Rectangle((3, y-0.4), 4, 0.8, facecolor=color, edgecolor='black', linewidth=2)
            axes[0].add_patch(rect)
            axes[0].text(5, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
            if i < len(layers_info) - 1:
                axes[0].annotate('', xy=(5, layers_info[i+1][1]-0.4), xytext=(5, y+0.4),
                                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        axes[0].set_title('Autoencoder Architecture', fontsize=14, fontweight='bold', pad=20)
        axes[1].axis('off')
        m = self.training_metrics
        metrics_text = f"""
    SOUND MODEL METRICS
    ----------------------------------------
    Reconstruction Accuracy:  {m['reconstruction_accuracy']:>10.2f}%
    Normal Detection Rate:    {m['normal_detection_rate']:>10.2f}%
    Final Training Loss:      {m['final_training_loss']:>10.6f}
    Final Validation Loss:    {m['final_validation_loss']:>10.6f}
    Best Validation Loss:     {m['best_validation_loss']:>10.6f}
    Anomaly Threshold:        {m['threshold']:>10.6f}
    Total Samples:            {m['total_samples']:>10d}
    Epochs Trained:           {m['epochs_trained']:>10d}
    ----------------------------------------
"""
        axes[1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center',
                    transform=axes[1].transAxes, bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='gray'))
        axes[1].set_title('Model Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_summary.png'), dpi=150)
        plt.close()

        return ['training_history.png', 'error_distribution.png', 'error_timeline.png', 'roc_curve.png',
                'confusion_matrix.png', 'reconstruction_comparison.png', 'feature_analysis.png', 'model_summary.png']


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
            response = requests.post(url, headers=headers, data=f.read(), timeout=120)
        if response.status_code in [200, 201]:
            print(f"Uploaded: {file_name}")
            return True
        return False
    except Exception as e:
        print(f"Upload error {file_name}: {e}")
        return False


def check_learn_condition():
    try:
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/sound/data?key={FIREBASE_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        fields = data.get('fields', {})
        learn = fields.get('learn', {}).get('booleanValue', False)
        time_field = fields.get('time', {})
        countdown_time = int(time_field.get('integerValue', 0) or time_field.get('doubleValue', 0))
        return learn, countdown_time
    except:
        return False, 0


def get_learn_phase_info():
    try:
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/learn_phase/data?key={FIREBASE_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        fields = data.get('fields', {})
        sound_name = fields.get('sound_name', {}).get('stringValue', 'sound_data')
        phase_name = fields.get('phase_name', {}).get('stringValue', 'phase_1')
        return f"{sound_name}_{phase_name}"
    except:
        return "soundSensorDataFeatures"


def get_phase_name_from_firestore(phase_type='learning'):
    """Fetch phase name from Firestore. phase_type: 'learning' or 'logging'"""
    try:
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/learn_phase/data?key={FIREBASE_API_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        fields = data.get('fields', {})
        if phase_type == 'learning':
            return fields.get('phase_name', {}).get('stringValue', '')
        else:
            return fields.get('logphase_name', {}).get('stringValue', '')
    except:
        return ''


def get_user_config_from_firestore():
    try:
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/sound/config?key={FIREBASE_API_KEY}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {}
        fields = response.json().get('fields', {})
        config = {}
        for key in ['epochs', 'batch_size', 'threshold_percentile', 'bottleneck_dim', 'hidden_dim', 'early_stopping_patience']:
            if key in fields:
                config[key] = int(fields[key].get('integerValue', 0))
        for key in ['dropout_rate', 'learning_rate']:
            if key in fields:
                config[key] = float(fields[key].get('doubleValue', 0))
        for key in ['use_data_augmentation', 'use_early_stopping']:
            if key in fields:
                config[key] = fields[key].get('booleanValue', True)
        return config
    except:
        return {}


def run_training(custom_config=None, custom_phase=None):
    global status_info
    status_info["is_training"] = True
    status_info["message"] = "Training in progress..."
    temp_dir = tempfile.gettempdir()
    viz_dir = os.path.join(temp_dir, 'sound_visualizations')
    try:
        firestore_config = get_user_config_from_firestore()
        final_config = {**DEFAULT_CONFIG, **firestore_config, **(custom_config or {})}
        status_info["config"] = final_config
        detector = AudioAnomalyDetector(config=final_config)
        detector.build_autoencoder()

        status_info["message"] = "Getting data path..."
        if custom_phase and custom_phase.strip():
            # Manual training with custom phase
            try:
                url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/learn_phase/data?key={FIREBASE_API_KEY}"
                response = requests.get(url, timeout=10)
                fields = response.json().get('fields', {})
                sound_name = fields.get('sound_name', {}).get('stringValue', 'sound_data')
                data_path = f"{sound_name}_{custom_phase.strip()}"
            except:
                data_path = f"sound_data_{custom_phase.strip()}"
        else:
            data_path = get_learn_phase_info()

        status_info["data_path"] = data_path
        status_info["message"] = f"Loading data from {data_path}..."
        training_data = detector.load_data_from_firebase(data_path)

        if len(training_data) == 0:
            status_info["message"] = "Error: No training data found"
            status_info["is_training"] = False
            return False

        status_info["message"] = f"Training with {len(training_data)} samples..."
        history = detector.train(training_data)

        if history is None:
            status_info["message"] = "Error: Training failed"
            status_info["is_training"] = False
            return False

        status_info["metrics"] = detector.training_metrics
        status_info["message"] = "Generating visualizations..."
        viz_files = detector.generate_visualizations(history, viz_dir)

        status_info["message"] = "Saving model files..."
        tflite_path = os.path.join(temp_dir, 'sound_anomaly_detector.tflite')
        params_path = os.path.join(temp_dir, 'sound_params.json')
        detector.convert_to_tflite(tflite_path)
        last_trained_time = time.strftime("%Y-%m-%d %H:%M:%S")
        detector.save_params(params_path, last_trained=last_trained_time, data_path=data_path)

        status_info["message"] = "Uploading to Supabase..."
        upload_to_supabase(tflite_path, 'sound_anomaly_detector.tflite')
        upload_to_supabase(params_path, 'sound_params.json')

        for viz_file in viz_files:
            viz_path = os.path.join(viz_dir, viz_file)
            if os.path.exists(viz_path):
                upload_to_supabase(viz_path, f'visualizations/{viz_file}')

        status_info["last_trained"] = last_trained_time
        status_info["message"] = "Training complete!"
        status_info["is_training"] = False
        status_info["training_just_completed"] = True
        return True
    except Exception as e:
        print(f"TRAINING ERROR: {e}")
        status_info["message"] = f"Error: {str(e)}"
        status_info["is_training"] = False
        return False


def monitor_firestore():
    global status_info
    while True:
        try:
            if status_info["is_training"]:
                time.sleep(5)
                continue
            learn, countdown_minutes = check_learn_condition()
            status_info["last_check"] = time.strftime("%H:%M:%S")
            if learn:
                countdown_seconds = max(0, (countdown_minutes * 60) - 30)
                if countdown_seconds > 0:
                    for remaining in range(countdown_seconds, 0, -1):
                        status_info["countdown"] = remaining
                        status_info["message"] = f"Training starts in {remaining // 60}m {remaining % 60}s..."
                        time.sleep(1)
                        if remaining % 10 == 0:
                            if not check_learn_condition()[0]:
                                status_info["message"] = "Cancelled"
                                status_info["countdown"] = 0
                                break
                    else:
                        status_info["countdown"] = 0
                        run_training()
                else:
                    run_training()
            else:
                # Don't overwrite "Training complete!" message
                if not status_info.get("training_just_completed", False):
                    status_info["message"] = "Waiting for learn=true..."
                status_info["countdown"] = 0
            time.sleep(5)
        except Exception as e:
            time.sleep(5)


monitor_thread = threading.Thread(target=monitor_firestore, daemon=True)
monitor_thread.start()


# ============== GRADIO INTERFACE ==============

def get_last_training_info_from_supabase():
    """Fetch last trained time and data path from Supabase params file"""
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/sound_params.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            params = response.json()
            last_trained = params.get('last_trained', '')
            data_path = params.get('data_path', '')
            return last_trained, data_path
    except Exception as e:
        print(f"Error fetching from Supabase: {e}")
    return "", ""


def get_status_html():
    # Get info from Supabase params file
    last_trained, data_path = get_last_training_info_from_supabase()
    
    # Use local status_info as fallback
    if not last_trained:
        last_trained = status_info.get("last_trained", "Never")
    if not data_path:
        data_path = status_info.get("data_path", "Not set")
    
    countdown_text = ""
    if status_info["countdown"] > 0:
        mins = status_info["countdown"] // 60
        secs = status_info["countdown"] % 60
        countdown_text = f'<div style="background:#fff3cd;padding:10px;border-radius:5px;margin:10px 0;"><strong>COUNTDOWN:</strong> {mins}m {secs}s</div>'

    status_color = "#28a745" if not status_info["is_training"] else "#007bff"
    if "Error" in status_info["message"]:
        status_color = "#dc3545"

    return f"""
    <div style="padding:15px;">
        <div style="background:{status_color};color:white;padding:12px;border-radius:5px;margin-bottom:15px;">
            <strong>Status:</strong> {status_info["message"]}
        </div>
        {countdown_text}
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:15px;">
            <div style="background:#e3f2fd;padding:10px;border-radius:5px;text-align:center;border:1px solid #2196F3;">
                <div style="font-size:12px;color:#1565c0;font-weight:bold;">Last Trained</div>
                <div style="font-size:14px;font-weight:bold;color:#0d47a1;">{last_trained or "Never"}</div>
            </div>
            <div style="background:#e8f5e9;padding:10px;border-radius:5px;text-align:center;border:1px solid #4CAF50;">
                <div style="font-size:12px;color:#2e7d32;font-weight:bold;">Data Path</div>
                <div style="font-size:14px;font-weight:bold;color:#1b5e20;">{data_path}</div>
            </div>
        </div>
    </div>
    """


def fetch_params_from_supabase():
    """Fetch the params JSON file from Supabase to get config and metrics"""
    try:
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/sound_params.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Supabase fetch failed with status: {response.status_code}")
    except Exception as e:
        print(f"Supabase fetch error: {e}")
    return None


def get_config_html():
    params = fetch_params_from_supabase()
    if params and 'config' in params:
        config = params['config']
        source = "from Supabase"
    else:
        config = status_info.get("config", DEFAULT_CONFIG)
        source = "default"
    return f"""
    <div style="padding:15px;">
        <h4 style="margin:0 0 10px 0;color:#333;">Last Training Configuration ({source})</h4>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Epochs</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('epochs', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Batch Size</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('batch_size', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Validation Split</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('validation_split', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Threshold Percentile</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('threshold_percentile', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Bottleneck Dim</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('bottleneck_dim', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Hidden Dim</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('hidden_dim', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Dropout Rate</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('dropout_rate', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Learning Rate</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('learning_rate', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Data Augmentation</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('use_data_augmentation', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Early Stopping</td><td style="padding:5px;border-bottom:1px solid #eee;">{config.get('use_early_stopping', '-')}</td></tr>
            <tr><td style="padding:5px;">Early Stopping Patience</td><td style="padding:5px;">{config.get('early_stopping_patience', '-')}</td></tr>
        </table>
    </div>
    """


def get_metrics_html():
    params = fetch_params_from_supabase()
    if params and 'metrics' in params:
        m = params['metrics']
        source = "from Supabase"
    elif status_info.get("metrics"):
        m = status_info["metrics"]
        source = "from current session"
    else:
        return "<div style='padding:15px;color:#666;'>No training metrics available yet. Train a model first.</div>"
    return f"""
    <div style="padding:15px;">
        <h4 style="margin:0 0 10px 0;color:#333;">Training Metrics ({source})</h4>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Reconstruction Accuracy</td><td style="padding:5px;border-bottom:1px solid #eee;"><strong>{m.get('reconstruction_accuracy', '-')}%</strong></td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Normal Detection Rate</td><td style="padding:5px;border-bottom:1px solid #eee;"><strong>{m.get('normal_detection_rate', '-')}%</strong></td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Final Training Loss</td><td style="padding:5px;border-bottom:1px solid #eee;">{m.get('final_training_loss', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Final Validation Loss</td><td style="padding:5px;border-bottom:1px solid #eee;">{m.get('final_validation_loss', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Best Validation Loss</td><td style="padding:5px;border-bottom:1px solid #eee;">{m.get('best_validation_loss', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Anomaly Threshold</td><td style="padding:5px;border-bottom:1px solid #eee;">{m.get('threshold', '-')}</td></tr>
            <tr><td style="padding:5px;border-bottom:1px solid #eee;">Total Samples</td><td style="padding:5px;border-bottom:1px solid #eee;">{m.get('total_samples', '-')}</td></tr>
            <tr><td style="padding:5px;">Epochs Trained</td><td style="padding:5px;">{m.get('epochs_trained', '-')}</td></tr>
        </table>
    </div>
    """


def refresh_status():
    return get_status_html(), get_config_html(), get_metrics_html()


def fetch_learning_phase():
    return get_phase_name_from_firestore('learning')


def fetch_logging_phase():
    return get_phase_name_from_firestore('logging')


def manual_train_with_config(epochs, batch_size, threshold_pct, bottleneck, hidden, dropout, lr, augment, early_stop, patience, custom_phase):
    if status_info["is_training"]:
        return "Training already in progress. Please wait."
    custom_config = {
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'threshold_percentile': int(threshold_pct),
        'bottleneck_dim': int(bottleneck),
        'hidden_dim': int(hidden),
        'dropout_rate': float(dropout),
        'learning_rate': float(lr),
        'use_data_augmentation': augment,
        'use_early_stopping': early_stop,
        'early_stopping_patience': int(patience)
    }
    phase = custom_phase.strip() if custom_phase else None
    threading.Thread(target=run_training, args=(custom_config, phase)).start()
    return "Training started. Use the Refresh button to monitor progress."


# Build the interface
with gr.Blocks(title="Sound Anomaly Model Trainer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Sound Anomaly Detection Model Trainer")
    gr.Markdown("Autoencoder-based anomaly detection for audio sensor data")

    with gr.Row():
        # Left Column - Status
        with gr.Column(scale=1):
            gr.Markdown("### System Status")
            status_html = gr.HTML(get_status_html())
            refresh_btn = gr.Button("Refresh Status", variant="secondary")

            gr.Markdown("### Configuration")
            config_html = gr.HTML(get_config_html())

            gr.Markdown("### Training Metrics")
            metrics_html = gr.HTML(get_metrics_html())

        # Right Column - Manual Training
        with gr.Column(scale=1):
            gr.Markdown("### Manual Training")

            with gr.Group():
                gr.Markdown("#### Data Source")
                custom_phase_input = gr.Textbox(label="Custom Phase Name", placeholder="Enter phase name or use buttons below", value="")
                with gr.Row():
                    learning_phase_btn = gr.Button("Use Last Learning Phase", size="sm")
                    logging_phase_btn = gr.Button("Use Last Logging Phase", size="sm")

            with gr.Group():
                gr.Markdown("#### Model Parameters")
                with gr.Row():
                    epochs_input = gr.Slider(25, 200, value=100, step=25, label="Epochs")
                    batch_size_input = gr.Slider(16, 64, value=32, step=8, label="Batch Size")
                with gr.Row():
                    threshold_input = gr.Slider(90, 99, value=95, step=1, label="Threshold Percentile")
                    bottleneck_input = gr.Slider(3, 8, value=6, step=1, label="Bottleneck Dim")
                with gr.Row():
                    hidden_input = gr.Slider(6, 16, value=12, step=2, label="Hidden Dim")
                    dropout_input = gr.Slider(0, 0.3, value=0.1, step=0.05, label="Dropout Rate")
                with gr.Row():
                    lr_input = gr.Number(value=0.001, label="Learning Rate")
                    patience_input = gr.Slider(5, 30, value=10, step=5, label="Early Stop Patience")
                with gr.Row():
                    augment_input = gr.Checkbox(value=True, label="Data Augmentation")
                    early_stop_input = gr.Checkbox(value=True, label="Early Stopping")

            result_box = gr.Textbox(label="Training Status", interactive=False)
            train_btn = gr.Button("Start Training", variant="primary")

    # Event handlers
    refresh_btn.click(fn=refresh_status, outputs=[status_html, config_html, metrics_html])
    learning_phase_btn.click(fn=fetch_learning_phase, outputs=custom_phase_input)
    logging_phase_btn.click(fn=fetch_logging_phase, outputs=custom_phase_input)
    train_btn.click(
        fn=manual_train_with_config,
        inputs=[epochs_input, batch_size_input, threshold_input, bottleneck_input, hidden_input,
                dropout_input, lr_input, augment_input, early_stop_input, patience_input, custom_phase_input],
        outputs=result_box
    )

demo.launch()
