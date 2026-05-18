from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Load semua model dan preprocessor
print("Loading models...")

# Mendapatkan path absolut
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, 'models')

# Load semua model
lr_model = joblib.load(os.path.join(models_dir, 'lr_model.pkl'))
ann_model = joblib.load(os.path.join(models_dir, 'ann_model.pkl'))
rnn_model = joblib.load(os.path.join(models_dir, 'rnn_model.pkl'))
kmeans = joblib.load(os.path.join(models_dir, 'kmeans_model.pkl'))
kmeans_regression = joblib.load(os.path.join(models_dir, 'kmeans_regression.pkl'))
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

with open(os.path.join(models_dir, 'feature_columns.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

# Load backpropagation weights
with open(os.path.join(models_dir, 'bp_weights.pkl'), 'rb') as f:
    bp_weights = pickle.load(f)

# Load historical data untuk mendapatkan rata-rata lag
excel_path = os.path.join(base_dir, '../Dataset KRL 2006-2023.xlsx')
df_hist = pd.read_excel(excel_path, sheet_name='Sheet1')
df_hist_jabodetabek = df_hist[df_hist['Wilayah'] == 'Jabodetabek'].copy()
df_hist_jabodetabek['Tanggal'] = pd.to_datetime(df_hist_jabodetabek['Tanggal'])
df_hist_jabodetabek.set_index('Tanggal', inplace=True)
df_hist_jabodetabek = df_hist_jabodetabek.sort_index()

# Hitung rata-rata untuk prediksi masa depan
avg_passengers = df_hist_jabodetabek['Jumlah'].mean()
last_12_values = df_hist_jabodetabek['Jumlah'].tail(12).values.tolist()
last_3_values = df_hist_jabodetabek['Jumlah'].tail(3).values.tolist()

print(f"✅ Models loaded successfully!")
print(f"📊 Average passengers: {avg_passengers:.0f}")
print(f"📅 Data range: {df_hist_jabodetabek.index.min()} to {df_hist_jabodetabek.index.max()}")

# Backpropagation class untuk inference
class BackpropagationNN:
    def __init__(self, weights):
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2.flatten()

bp_model = BackpropagationNN(bp_weights)

# Data untuk template
template_data = {
    'last_data': {
        'last_date': df_hist_jabodetabek.index.max().strftime('%B %Y'),
        'last_value': int(last_12_values[-1]) if last_12_values else 0,
        'avg_value': int(avg_passengers)
    }
}

def predict_future(year, month, last_actual_values=None):
    """Prediksi untuk tahun yang akan datang"""
    if last_actual_values is None:
        lag_1 = last_12_values[-1] if len(last_12_values) > 0 else avg_passengers
        lag_2 = last_12_values[-2] if len(last_12_values) > 1 else avg_passengers
        lag_3 = last_12_values[-3] if len(last_12_values) > 2 else avg_passengers
        lag_6 = last_12_values[-6] if len(last_12_values) > 5 else avg_passengers
        lag_12 = last_12_values[-12] if len(last_12_values) > 11 else avg_passengers
        rolling_mean_3 = np.mean(last_3_values) if len(last_3_values) == 3 else avg_passengers
        rolling_mean_6 = np.mean(last_12_values[-6:]) if len(last_12_values) >= 6 else avg_passengers
        rolling_std_3 = np.std(last_3_values) if len(last_3_values) == 3 else avg_passengers * 0.1
        rolling_std_6 = np.std(last_12_values[-6:]) if len(last_12_values) >= 6 else avg_passengers * 0.1
    else:
        lag_1, lag_2, lag_3, lag_6, lag_12 = last_actual_values[:5]
        rolling_mean_3 = np.mean(last_actual_values[:3]) if len(last_actual_values) >= 3 else avg_passengers
        rolling_mean_6 = np.mean(last_actual_values[:6]) if len(last_actual_values) >= 6 else avg_passengers
        rolling_std_3 = np.std(last_actual_values[:3]) if len(last_actual_values) >= 3 else avg_passengers * 0.1
        rolling_std_6 = np.std(last_actual_values[:6]) if len(last_actual_values) >= 6 else avg_passengers * 0.1
    
    date = datetime(year, month, 1)
    dayofyear = date.timetuple().tm_yday
    quarter = (month - 1) // 3 + 1
    
    features = np.array([[year, month, dayofyear, quarter, 
                          lag_1, lag_2, lag_3, lag_6, lag_12,
                          rolling_mean_3, rolling_mean_6, rolling_std_3, rolling_std_6]])
    
    features_scaled = scaler.transform(features)
    
    pred_lr = lr_model.predict(features_scaled)[0]
    pred_ann = ann_model.predict(features_scaled)[0]
    pred_rnn = rnn_model.predict(features_scaled)[0]
    
    cluster = kmeans.predict(features_scaled)[0]
    features_cluster = np.column_stack([features_scaled, [[cluster]]])
    pred_cluster = kmeans_regression.predict(features_cluster)[0]
    
    pred_bp = bp_model.predict(features_scaled)[0]
    
    return {
        'linear_regression': max(0, round(float(pred_lr), 0)),
        'ann': max(0, round(float(pred_ann), 0)),
        'rnn': max(0, round(float(pred_rnn), 0)),
        'kmeans_regression': max(0, round(float(pred_cluster), 0)),
        'backpropagation': max(0, round(float(pred_bp), 0))
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            prediction_type = request.form.get('prediction_type', 'single')
            
            if prediction_type == 'single':
                year = int(request.form['year'])
                month = int(request.form['month'])
                
                lag_1 = float(request.form.get('lag_1', last_12_values[-1] if last_12_values else avg_passengers))
                lag_2 = float(request.form.get('lag_2', last_12_values[-2] if len(last_12_values) > 1 else avg_passengers))
                lag_3 = float(request.form.get('lag_3', last_12_values[-3] if len(last_12_values) > 2 else avg_passengers))
                lag_6 = float(request.form.get('lag_6', last_12_values[-6] if len(last_12_values) > 5 else avg_passengers))
                lag_12 = float(request.form.get('lag_12', last_12_values[-12] if len(last_12_values) > 11 else avg_passengers))
                rolling_mean_3 = float(request.form.get('rolling_mean_3', np.mean(last_3_values) if last_3_values else avg_passengers))
                rolling_mean_6 = float(request.form.get('rolling_mean_6', avg_passengers))
                rolling_std_3 = float(request.form.get('rolling_std_3', avg_passengers * 0.1))
                rolling_std_6 = float(request.form.get('rolling_std_6', avg_passengers * 0.1))
                
                date = datetime(year, month, 1)
                dayofyear = date.timetuple().tm_yday
                quarter = (month - 1) // 3 + 1
                
                features = np.array([[year, month, dayofyear, quarter, 
                                      lag_1, lag_2, lag_3, lag_6, lag_12,
                                      rolling_mean_3, rolling_mean_6, rolling_std_3, rolling_std_6]])
                
                features_scaled = scaler.transform(features)
                
                pred_lr = lr_model.predict(features_scaled)[0]
                pred_ann = ann_model.predict(features_scaled)[0]
                pred_rnn = rnn_model.predict(features_scaled)[0]
                
                cluster = kmeans.predict(features_scaled)[0]
                features_cluster = np.column_stack([features_scaled, [[cluster]]])
                pred_cluster = kmeans_regression.predict(features_cluster)[0]
                pred_bp = bp_model.predict(features_scaled)[0]
                
                predictions = {
                    'linear_regression': max(0, round(float(pred_lr), 0)),
                    'ann': max(0, round(float(pred_ann), 0)),
                    'rnn': max(0, round(float(pred_rnn), 0)),
                    'kmeans_regression': max(0, round(float(pred_cluster), 0)),
                    'backpropagation': max(0, round(float(pred_bp), 0))
                }
                
                return render_template('predict.html', 
                                     predictions=predictions,
                                     prediction_type='single', 
                                     year=year, 
                                     month=month,
                                     last_data=template_data['last_data'])
            
            elif prediction_type == 'future_year':
                start_year = int(request.form['start_year'])
                future_predictions = []
                
                current_lags = last_12_values.copy()
                
                for month in range(1, 13):
                    pred = predict_future(start_year, month, current_lags[-12:] if len(current_lags) >= 12 else current_lags)
                    
                    avg_pred = np.mean([pred['linear_regression'], pred['ann'], 
                                       pred['rnn'], pred['backpropagation']])
                    current_lags.append(avg_pred)
                    
                    future_predictions.append({
                        'month': month,
                        'month_name': datetime(start_year, month, 1).strftime('%B'),
                        'predictions': pred
                    })
                
                return render_template('predict.html', 
                                     future_predictions=future_predictions,
                                     prediction_type='future_year', 
                                     start_year=start_year,
                                     last_data=template_data['last_data'])
            
            elif prediction_type == 'multi_year':
                start_year = int(request.form['start_year'])
                num_years = int(request.form['num_years'])
                all_predictions = []
                
                current_lags = last_12_values.copy()
                
                for year_offset in range(num_years):
                    year = start_year + year_offset
                    year_predictions = []
                    
                    for month in range(1, 13):
                        pred = predict_future(year, month, current_lags[-12:] if len(current_lags) >= 12 else current_lags)
                        
                        avg_pred = np.mean([pred['linear_regression'], pred['ann'], 
                                           pred['rnn'], pred['backpropagation']])
                        current_lags.append(avg_pred)
                        
                        year_predictions.append({
                            'month': month,
                            'month_name': datetime(year, month, 1).strftime('%B'),
                            'predictions': pred
                        })
                    
                    all_predictions.append({
                        'year': year,
                        'months': year_predictions,
                        'year_avg': np.mean([p['predictions']['linear_regression'] for p in year_predictions])
                    })
                
                return render_template('predict.html', 
                                     all_predictions=all_predictions,
                                     prediction_type='multi_year', 
                                     start_year=start_year, 
                                     num_years=num_years,
                                     last_data=template_data['last_data'])
            
        except Exception as e:
            return render_template('predict.html', 
                                 error=str(e),
                                 last_data=template_data['last_data'])
    
    return render_template('predict.html', last_data=template_data['last_data'])

@app.route('/compare')
def compare():
    comparison_df = pd.read_csv(os.path.join(models_dir, 'model_comparison.csv'))
    models = comparison_df.to_dict('records')
    return render_template('compare.html', models=models)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)