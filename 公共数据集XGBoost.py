import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import re

# ==================== 随机种子配置 ====================
current_seed = random.randint(0, 99999)

random.seed(current_seed)
np.random.seed(current_seed)
tf.random.set_seed(current_seed)


# ==================== 数据预处理 ====================
def load_and_preprocess(file_path):
    # 读取数据
    data = pd.read_excel(file_path, engine='openpyxl')

    # 列名标准化处理（关键修正）
    data.columns = [
        re.sub(r'[^a-zA-Z0-9]+', '_', col).strip('_').lower()  # 统一转换为小写
        for col in data.columns
    ]

    # 验证必需列（修正列名大小写问题）
    required_columns = {
        'lv_activepower_kw',
        'wind_speed_m_s',
        'theoretical_power_curve_kwh',
        'wind_direction'
    }
    missing = required_columns - set(data.columns.str.lower())
    if missing:
        raise KeyError(f"缺失列：{missing}，实际列名：{data.columns.tolist()}")

    # 定义输入输出（修正大小写匹配）
    input_features = [
        'wind_speed_m_s',
        'theoretical_power_curve_kwh',
        'wind_direction'
    ]
    output_column = 'lv_activepower_kw'

    # 数值特征标准化（关键修正）
    global scaler
    scaler = StandardScaler()
    data[input_features] = scaler.fit_transform(data[input_features])

    return data[input_features], data[output_column]


# ==================== 模型构建 ====================
def build_model(input_shape):
    initializer = tf.keras.initializers.GlorotNormal(seed=current_seed)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(0.01, 0.01),
                              input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3, seed=current_seed),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(32, activation='relu',
                              kernel_initializer=initializer),
        tf.keras.layers.Dense(1, kernel_initializer=initializer)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


def predict_power(model):
    try:
        # 获取用户输入（增加单位提示）
        wind_speed = float(input("风速（m/s）: "))
        power_curve = float(input("理论功率曲线（KWh）: "))
        wind_dir = float(input("风向（角度）: "))

        # 构建特征数组（修正输入顺序）
        raw_features = np.array([[wind_speed, power_curve, wind_dir]])

        # 标准化处理（关键修正）
        scaled_features = scaler.transform(raw_features)

        # 预测
        prediction = model.predict(scaled_features.astype(np.float32), verbose=0)[0][0]
        print(f"\n预测功率：{prediction:.2f} kW")

    except Exception as e:
        print(f"输入错误：{str(e)}")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 数据准备
    file_path = r'C:\Users\朱许杨\Desktop\毕业项目\T1.xlsx'
    X, y = load_and_preprocess(file_path)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 构建模型
    model = build_model((X_train.shape[1],))

    # 训练配置（添加早停机制）
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=50, restore_best_weights=True)

    history = model.fit(
        X_train.values.astype(np.float32),
        y_train.values.astype(np.float32),
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # 评估模型
    test_loss, test_mae = model.evaluate(X_test.values.astype(np.float32),
                                         y_test.values.astype(np.float32))
    print(f"\n测试集评估：\nMSE：{test_loss:.2f}\nMAE：{test_mae:.2f}")

    # 交互预测
    while True:
        predict_power(model)
        if input("继续预测？(y/n)：").lower() != 'y':
            break