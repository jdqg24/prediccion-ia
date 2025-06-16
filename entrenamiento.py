import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

CSV_PATH = 'data.csv'
MODEL_DIR = 'models'
SEQUENCE_LENGTH = 7
EPOCHS = 100
BATCH_SIZE = 32

os.makedirs(MODEL_DIR, exist_ok=True)

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

df = pd.read_csv(CSV_PATH)
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y/%m/%d')
df.sort_values('fecha', inplace=True)

# Todas las columnas menos 'fecha' son ríos
rios = [col for col in df.columns if col != 'fecha']

for rio in rios:
    print(f"Entrenando modelo para río: {rio}")
    data = df[[rio]].dropna().values

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH)
    X_val, y_val = create_sequences(val_scaled, SEQUENCE_LENGTH)
    X_test, y_test = create_sequences(test_scaled, SEQUENCE_LENGTH)

    X_train = X_train.reshape((-1, SEQUENCE_LENGTH, 1))
    X_val = X_val.reshape((-1, SEQUENCE_LENGTH, 1))
    X_test = X_test.reshape((-1, SEQUENCE_LENGTH, 1))

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint_path = os.path.join(MODEL_DIR, f'{rio}_best_model.h5')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=2
    )

    joblib.dump(scaler, os.path.join(MODEL_DIR, f'{rio}_scaler.save'))

    print(f"Modelo y scaler guardados para río: {rio}")
