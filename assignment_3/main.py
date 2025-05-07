#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_and_preprocess(path, n_steps=24):
    # 1. 读取
    df = pd.read_csv(path)
    
    # 2. 解析 datetime
    # 假设 date 列格式为 "YYYY-MM-DD HH:MM:SS" 或 "YYYY/MM/DD HH:MM"
    df['datetime'] = pd.to_datetime(df['date'])
    df.set_index('datetime', inplace=True)
    df.drop(columns=['date'], inplace=True)
    
    # 3. 缺失值填充
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # 4. 独热编码风向
    df = pd.get_dummies(df, columns=['wnd_dir'])
    
    # 5. 标准化
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    
    # 6. 构建 supervised 序列
    X, y = [], []
    data = df_scaled.values
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps, :])
        y.append(data[i+n_steps, 0])  # pm2.5 在第 0 列，现在即 pollution
    return np.array(X), np.array(y), scaler, df_scaled.columns

def build_model(n_steps, n_features):
    model = Sequential([
        LSTM(64, input_shape=(n_steps, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_history(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def main():
    # 超参数
    DATA_PATH = './LSTM-Multivariate_pollution.csv'
    N_STEPS = 24
    TEST_RATIO = 0.2
    BATCH_SIZE = 32
    EPOCHS = 10
    PATIENCE = 5

    # 加载与预处理
    X, y, scaler, col_names = load_and_preprocess(DATA_PATH, N_STEPS)
    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    n_features = X_train.shape[2]

    # 构建模型
    model = build_model(N_STEPS, n_features)
    model.summary()

    # 训练
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=2
    )

    # 可视化
    plot_history(history)

    # 评估
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'Validation RMSE: {rmse:.3f}')

    # 下小时预测
    last_seq = X[-1].reshape((1, N_STEPS, n_features))
    pred_scaled = model.predict(last_seq)[0,0]
    # 反归一化
    min0, scale0 = scaler.data_min_[0], scaler.scale_[0]
    pred_pm25 = pred_scaled/scale0 + min0
    print(f'Next-hour PM2.5 预测值：{pred_pm25:.2f}')
    
    r2 = r2_score(y_val, y_pred)
    print(f'Validation R² Score: {r2:.3f}')

if __name__ == '__main__':
    main()
