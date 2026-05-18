"""
TRAINING 5 MODELS FOR KRL PASSENGER PREDICTION
Tanpa TensorFlow - Menggunakan Scikit-learn + NumPy
Algoritma: Linear Regression, ANN (MLPRegressor), RNN (Gradient Boosting untuk time series), 
           K-Means, Backpropagation Manual
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING 5 MODELS FOR KRL PASSENGER PREDICTION")
print("(Tanpa TensorFlow - Scikit-learn + NumPy)")
print("="*60)

# Buat folder models jika belum ada
os.makedirs('models', exist_ok=True)

# ============================================
# 1. LOAD & PREPROCESS DATA
# ============================================
print("\n[1/7] Loading data...")

# Cari file Excel
current_dir = os.path.dirname(os.path.abspath(__file__))
excel_file = None

for file in os.listdir(current_dir):
    if file.endswith('.xlsx') and 'KRL' in file:
        excel_file = os.path.join(current_dir, file)
        break

if excel_file is None:
    print("❌ File Excel tidak ditemukan!")
    exit(1)

print(f"✅ File ditemukan: {excel_file}")

df = pd.read_excel(excel_file, sheet_name='Sheet1')

# Filter Jabodetabek
df_jabodetabek = df[df['Wilayah'] == 'Jabodetabek'].copy()
df_jabodetabek['Tanggal'] = pd.to_datetime(df_jabodetabek['Tanggal'])
df_jabodetabek.set_index('Tanggal', inplace=True)

print(f"Data shape: {df_jabodetabek.shape}")
print(f"Date range: {df_jabodetabek.index.min()} to {df_jabodetabek.index.max()}")

# Feature engineering untuk time series
df_jabodetabek['year'] = df_jabodetabek.index.year
df_jabodetabek['month'] = df_jabodetabek.index.month
df_jabodetabek['dayofyear'] = df_jabodetabek.index.dayofyear
df_jabodetabek['quarter'] = df_jabodetabek.index.quarter

# Lag features (data historis)
df_jabodetabek['lag_1'] = df_jabodetabek['Jumlah'].shift(1)
df_jabodetabek['lag_2'] = df_jabodetabek['Jumlah'].shift(2)
df_jabodetabek['lag_3'] = df_jabodetabek['Jumlah'].shift(3)
df_jabodetabek['lag_6'] = df_jabodetabek['Jumlah'].shift(6)
df_jabodetabek['lag_12'] = df_jabodetabek['Jumlah'].shift(12)

# Rolling statistics
df_jabodetabek['rolling_mean_3'] = df_jabodetabek['Jumlah'].rolling(3).mean()
df_jabodetabek['rolling_mean_6'] = df_jabodetabek['Jumlah'].rolling(6).mean()
df_jabodetabek['rolling_std_3'] = df_jabodetabek['Jumlah'].rolling(3).std()
df_jabodetabek['rolling_std_6'] = df_jabodetabek['Jumlah'].rolling(6).std()

# Drop NaN values
df_jabodetabek.dropna(inplace=True)

# Prepare features
feature_cols = ['year', 'month', 'dayofyear', 'quarter', 
                'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3', 'rolling_std_6']

X = df_jabodetabek[feature_cols].values
y = df_jabodetabek['Jumlah'].values

print(f"Jumlah fitur: {len(feature_cols)}")
print(f"Total sampel setelah feature engineering: {len(X)}")

# Train-test split (80-20 untuk time series)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalisasi dengan MinMaxScaler untuk Backpropagation (range 0-1)
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

X_train_scaled_standard = scaler_standard.fit_transform(X_train)
X_test_scaled_standard = scaler_standard.transform(X_test)

X_train_scaled_minmax = scaler_minmax.fit_transform(X_train)
X_test_scaled_minmax = scaler_minmax.transform(X_test)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Save scaler
joblib.dump(scaler_standard, 'models/scaler.pkl')
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

# ============================================
# 2. LINEAR REGRESSION (Model 1) - Ridge
# ============================================
print("\n[2/7] Training Linear Regression (Ridge)...")
lr_model = Ridge(alpha=1.0)
lr_model.fit(X_train_scaled_standard, y_train)

y_pred_lr = lr_model.predict(X_test_scaled_standard)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"  MAE: {mae_lr:.2f}")
print(f"  RMSE: {rmse_lr:.2f}")
print(f"  R2 Score: {r2_lr:.4f}")

joblib.dump(lr_model, 'models/lr_model.pkl')

# ============================================
# 3. ARTIFICIAL NEURAL NETWORK (ANN) - Model 2
# ============================================
print("\n[3/7] Training Artificial Neural Network (MLPRegressor)...")
ann_model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=False
)

ann_model.fit(X_train_scaled_standard, y_train)

y_pred_ann = ann_model.predict(X_test_scaled_standard)
mae_ann = mean_absolute_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
r2_ann = r2_score(y_test, y_pred_ann)

print(f"  MAE: {mae_ann:.2f}")
print(f"  RMSE: {rmse_ann:.2f}")
print(f"  R2 Score: {r2_ann:.4f}")
print(f"  Iterations: {ann_model.n_iter_}")

joblib.dump(ann_model, 'models/ann_model.pkl')

# ============================================
# 4. RNN (Gradient Boosting) - Model 3
# ============================================
print("\n[4/7] Training RNN (Gradient Boosting untuk Time Series)...")

rnn_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

rnn_model.fit(X_train_scaled_standard, y_train)

y_pred_rnn = rnn_model.predict(X_test_scaled_standard)
mae_rnn = mean_absolute_error(y_test, y_pred_rnn)
rmse_rnn = np.sqrt(mean_squared_error(y_test, y_pred_rnn))
r2_rnn = r2_score(y_test, y_pred_rnn)

print(f"  MAE: {mae_rnn:.2f}")
print(f"  RMSE: {rmse_rnn:.2f}")
print(f"  R2 Score: {r2_rnn:.4f}")

joblib.dump(rnn_model, 'models/rnn_model.pkl')

# ============================================
# 5. K-MEANS CLUSTERING (Model 4)
# ============================================
print("\n[5/7] Training K-Means Clustering...")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_scaled_standard)

cluster_means = []
for i in range(3):
    cluster_mask = (clusters == i)
    if np.sum(cluster_mask) > 0:
        cluster_means.append(np.mean(y_train[cluster_mask]))
    else:
        cluster_means.append(0)

print(f"  Cluster 0 (n={np.sum(clusters==0)}): rata-rata penumpang = {cluster_means[0]:.0f}")
print(f"  Cluster 1 (n={np.sum(clusters==1)}): rata-rata penumpang = {cluster_means[1]:.0f}")
print(f"  Cluster 2 (n={np.sum(clusters==2)}): rata-rata penumpang = {cluster_means[2]:.0f}")

X_train_cluster = np.column_stack([X_train_scaled_standard, clusters])
X_test_cluster = np.column_stack([X_test_scaled_standard, kmeans.predict(X_test_scaled_standard)])

lr_cluster = Ridge(alpha=1.0)
lr_cluster.fit(X_train_cluster, y_train)

y_pred_cluster = lr_cluster.predict(X_test_cluster)
mae_cluster = mean_absolute_error(y_test, y_pred_cluster)
rmse_cluster = np.sqrt(mean_squared_error(y_test, y_pred_cluster))
r2_cluster = r2_score(y_test, y_pred_cluster)

print(f"\n  K-Means + Regression MAE: {mae_cluster:.2f}")
print(f"  K-Means + Regression RMSE: {rmse_cluster:.2f}")
print(f"  K-Means + Regression R2 Score: {r2_cluster:.4f}")

joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(lr_cluster, 'models/kmeans_regression.pkl')

# ============================================
# 6. BACKPROPAGATION MANUAL (Model 5) - DIPERBAIKI
# ============================================
print("\n[6/7] Training Backpropagation Manual (dari scratch dengan NumPy)...")

class BackpropagationNN:
    def __init__(self, input_size, hidden_size=16, output_size=1, learning_rate=0.0001):
        # Inisialisasi weight dengan nilai kecil
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate
        self.losses = []
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Gradient clipping untuk mencegah exploding gradient
        dZ2 = output - y.reshape(-1, 1)
        dZ2 = np.clip(dZ2, -1, 1)  # Gradient clipping
        
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.z1)
        dZ1 = np.clip(dZ1, -1, 1)  # Gradient clipping
        
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Update weights dengan learning rate kecil
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=200, batch_size=16, verbose=True):
        n_samples = X.shape[0]
        y_mean = np.mean(y)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward(X_batch)
                
                # Cegah nilai loss yang terlalu besar
                loss = np.mean((output - y_batch.reshape(-1, 1))**2)
                if np.isnan(loss) or loss > 1e10:
                    loss = 1e10
                
                epoch_loss += loss
                n_batches += 1
                
                self.backward(X_batch, y_batch, output)
            
            avg_loss = epoch_loss / n_batches
            self.losses.append(avg_loss)
            
            if verbose and (epoch % 50 == 0 or epoch == epochs-1):
                print(f"    Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return self.losses
    
    def predict(self, X):
        return self.forward(X).flatten()

# Train backpropagation dengan MinMaxScaler (agar nilai input dalam range 0-1)
print("  Melakukan training Backpropagation...")
bp_model = BackpropagationNN(
    input_size=X_train_scaled_minmax.shape[1],
    hidden_size=16,
    output_size=1,
    learning_rate=0.0001
)

losses = bp_model.train(X_train_scaled_minmax, y_train, epochs=200, batch_size=16, verbose=True)

# Predict dengan MinMaxScaler
y_pred_bp = bp_model.predict(X_test_scaled_minmax)
y_pred_bp = np.nan_to_num(y_pred_bp, nan=np.mean(y_train))  # Ganti NaN dengan mean

mae_bp = mean_absolute_error(y_test, y_pred_bp)
rmse_bp = np.sqrt(mean_squared_error(y_test, y_pred_bp))
r2_bp = r2_score(y_test, y_pred_bp)

print(f"\n  MAE: {mae_bp:.2f}")
print(f"  RMSE: {rmse_bp:.2f}")
print(f"  R2 Score: {r2_bp:.4f}")

# Save backpropagation weights
with open('models/bp_weights.pkl', 'wb') as f:
    pickle.dump({
        'W1': bp_model.W1, 'b1': bp_model.b1,
        'W2': bp_model.W2, 'b2': bp_model.b2,
        'losses': bp_model.losses
    }, f)

# ============================================
# 7. HASIL PERBANDINGAN & VISUALISASI
# ============================================
print("\n" + "="*60)
print("HASIL PERBANDINGAN 5 MODEL")
print("="*60)

results = pd.DataFrame({
    'Model': ['Linear Regression (Ridge)', 'ANN (MLPRegressor)', 'RNN (Gradient Boosting)', 
              'K-Means + Regresi', 'Backpropagation Manual'],
    'MAE': [mae_lr, mae_ann, mae_rnn, mae_cluster, mae_bp],
    'RMSE': [rmse_lr, rmse_ann, rmse_rnn, rmse_cluster, rmse_bp],
    'R2 Score': [r2_lr, r2_ann, r2_rnn, r2_cluster, r2_bp]
})

print(results.to_string(index=False))

# Simpan hasil
results.to_csv('models/model_comparison.csv', index=False)

# Plot perbandingan
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Perbandingan Kinerja 5 Model Prediksi Jumlah Penumpang KRL', fontsize=14)

metrics = ['MAE', 'RMSE', 'R2 Score']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

for i, metric in enumerate(metrics):
    bars = axes[i].bar(results['Model'], results[metric], color=colors)
    axes[i].set_title(f'Perbandingan {metric}', fontsize=12)
    axes[i].set_xlabel('Model', fontsize=10)
    axes[i].set_ylabel(metric, fontsize=10)
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
    
    for bar, val in zip(bars, results[metric]):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (abs(bar.get_height())*0.02),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('models/model_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Grafik perbandingan disimpan di 'models/model_comparison.png'")

# Plot training loss untuk Backpropagation
plt.figure(figsize=(10, 5))
plt.plot(bp_model.losses, color='#f39c12', linewidth=2)
plt.title('Backpropagation Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('models/bp_training_loss.png', dpi=150, bbox_inches='tight')
print("📈 Backpropagation loss curve disimpan di 'models/bp_training_loss.png'")

# Plot aktual vs prediksi untuk model terbaik
best_idx = np.argmax(results['R2 Score'].values)
best_model_name = results.iloc[best_idx, 0]
print(f"\n🏆 Model terbaik: {best_model_name} dengan R² Score = {results['R2 Score'].max():.4f}")

# Plot untuk Linear Regression
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5, color='#3498db')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Aktual', fontsize=12)
plt.ylabel('Prediksi', fontsize=12)
plt.title('Linear Regression: Aktual vs Prediksi', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_pred_lr
plt.hist(residuals, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
plt.xlabel('Residual', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.title('Distribusi Residual - Linear Regression', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/prediction_analysis.png', dpi=150, bbox_inches='tight')
print("📉 Analisis prediksi disimpan di 'models/prediction_analysis.png'")

# Plot feature importance
plt.figure(figsize=(12, 6))
importances = rnn_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
plt.barh(range(len(indices)), importances[indices], color='#3498db')
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Importance')
plt.title('Feature Importance - RNN (Gradient Boosting)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
print("📊 Feature importance disimpan di 'models/feature_importance.png'")

plt.show()

print("\n" + "="*60)
print("✅ SEMUA MODEL TELAH DILATIH DAN DISIMPAN!")
print("="*60)
print("\n📁 File yang disimpan di folder 'models/':")
print("   - lr_model.pkl (Linear Regression Ridge)")
print("   - ann_model.pkl (ANN - MLPRegressor)")
print("   - rnn_model.pkl (RNN - Gradient Boosting)")
print("   - kmeans_model.pkl + kmeans_regression.pkl (K-Means)")
print("   - bp_weights.pkl (Backpropagation weights)")
print("   - scaler.pkl (StandardScaler)")
print("   - feature_columns.pkl")
print("   - model_comparison.csv")
print("   - model_comparison.png")
print("   - bp_training_loss.png")
print("   - prediction_analysis.png")
print("   - feature_importance.png")