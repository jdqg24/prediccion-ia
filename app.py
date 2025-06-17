from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
from datetime import datetime, timedelta
import pandas as pd

app = Flask(__name__, static_folder='static')
MODEL_DIR = 'models'
CSV_PATH = 'data.csv'
SEQUENCE_LENGTH = 7

# Cargar el dataframe una vez al iniciar
df = pd.read_csv(CSV_PATH)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y/%m/%d')
df.sort_values('fecha', inplace=True)

def load_scaler_and_model(rio):
    scaler_path = os.path.join(MODEL_DIR, f'{rio}_scaler.save')
    model_path = os.path.join(MODEL_DIR, f'{rio}_best_model.h5')
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        return None, None
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    return scaler, model

def get_last_values(rio):
    river_data = df[['fecha', rio]].dropna()
    last_values = river_data[rio].tail(SEQUENCE_LENGTH).values
    last_date = river_data['fecha'].tail(1).iloc[0]
    return last_values.tolist(), last_date

def predict_sequence(scaler, model, initial_values, days):
    predictions = []
    window = list(initial_values)
    for _ in range(days):
        input_scaled = scaler.transform(np.array(window).reshape(-1,1)).reshape(1, SEQUENCE_LENGTH, 1)
        pred_scaled = model.predict(input_scaled, verbose=0)
        pred = scaler.inverse_transform(pred_scaled).flatten()[0]
        predictions.append(pred)
        window = window[1:] + [pred]
    return predictions

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/last-values', methods=['GET'])
def get_last_values_endpoint():
    rios = [col for col in df.columns if col != 'fecha']
    last_values = {}
    for rio in rios:
        values, last_date = get_last_values(rio)
        last_values[rio] = {
            'values': values,
            'last_date': last_date.strftime('%Y-%m-%d')
        }
    return jsonify(last_values)

@app.route('/api/predict-multiple', methods=['POST'])
def predict_multiple():
    try:
        data = request.get_json(force=True)
        rios = data.get('cities')
        days = int(data.get('days', 7))
        start_date_str = data.get('start_date')  # formato "YYYY-MM-DD"

        if not rios:
            return jsonify({'error': 'Debe enviar "cities"'}), 400

        if not start_date_str:
            return jsonify({'error': 'Debe enviar "start_date" en formato YYYY-MM-DD'}), 400

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

        predictions_all = {}
        start_dates = {}

        for rio in rios:
            scaler, model = load_scaler_and_model(rio)
            if scaler is None or model is None:
                return jsonify({'error': f'Modelo o scaler para el río "{rio}" no encontrados.'}), 404

            initial_values, last_date = get_last_values(rio)
            river_data = df[['fecha', rio]].dropna().sort_values('fecha')

            if start_date < last_date:
                # Predicción hacia atrás
                days_to_predict_back = (last_date - start_date).days
                available_past = river_data[river_data['fecha'] < last_date].tail(SEQUENCE_LENGTH)

                if len(available_past) < SEQUENCE_LENGTH:
                    return jsonify({'error': f'No hay suficientes datos anteriores para predecir hacia atrás para el río "{rio}".'}), 400

                input_values = available_past[rio].values[::-1]  # invertir
                pred_backwards = predict_sequence(scaler, model, input_values, days_to_predict_back)
                pred_backwards = pred_backwards[::-1]  # revertimos orden

                predictions = []
                for i in range(days_to_predict_back):
                    date_pred = last_date - timedelta(days=i+1)
                    real_val_row = river_data[river_data['fecha'] == date_pred]
                    real_val = real_val_row[rio].values[0] if not real_val_row.empty else None
                    predictions.append({
                        'fecha': date_pred.strftime("%Y-%m-%d"),
                        'predicted_level': float(pred_backwards[i]),
                        'real_level': float(real_val) if real_val is not None else None
                    })

                predictions_all[rio] = predictions[::-1]
                start_dates[rio] = start_date
                continue

            # Predicción hacia adelante (como antes)
            initial_values, last_date = get_last_values(rio)
            start_dates[rio] = last_date + timedelta(days=1)

            preds = predict_sequence(scaler, model, initial_values, days)

            predictions_all[rio] = []
            for i, val in enumerate(preds):
                date_pred = start_dates[rio] + timedelta(days=i)
                predictions_all[rio].append({
                    'fecha': date_pred.strftime("%Y-%m-%d"),
                    'predicted_level': float(val)
                })

        return jsonify({
            'predictions': predictions_all,
            'start_dates': {k: v.strftime('%Y-%m-%d') for k, v in start_dates.items()}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
