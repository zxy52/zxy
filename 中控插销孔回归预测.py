import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 读取数据集
file_path = 'C:/Users/朱许杨/Desktop/毕业项目/中控插销孔.xlsx'
data = pd.read_excel(file_path)

# 特征编码
label_encoders = {}
for column in ['子/母', '前板/后板', '孔型']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 特征和目标变量
features = ['厚度', '子/母', '前板/后板', '孔型']
targets = ['A', 'B']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data[features], data[targets], test_size=0.2, random_state=42)

# 转换为 NumPy 数组
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(targets))
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X_train_np, y_train_np, epochs=10000, verbose=1)

# 用测试集评估模型
test_loss = model.evaluate(X_test_np, y_test_np, verbose=1)
print(f'Test Loss: {test_loss}')

thickness_input = float(input("请输入厚度："))
type_input = input("请输入子/母：")
plate_input = input("请输入前板/后板：")
hole_type_input = input("请输入孔型：")
板材长cm = float(input("请输入板材长cm: "))

# 构建输入数据的 NumPy 数组
input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                                   label_encoders["前板/后板"].transform([plate_input])[0],
                                   label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

# 使用 TensorFlow 模型进行预测
predicted_values = model.predict(input_data_np)[0]

# 输出预测结果
# print(f'预测的A值:{predicted_values[0]}')
# print(f"预测的B值:{predicted_values[1]}")

opening_direction = input("请输入开向：")

def calculate_A(predicted_values):
    A = predicted_values[0]
    return A

A_result = calculate_A(predicted_values)

def calculate_B(predicted_values, 板材长cm):
    if type_input == '子' and plate_input == '前板' and (opening_direction == '外左' or opening_direction == '内右'):
        B = 板材长cm - predicted_values[1]
    else:
        B = predicted_values[1]
    return B

B_result = calculate_B(predicted_values, 板材长cm)

print(f"中控插销孔A的值为: {A_result}")
print(f"中控插销孔B的值为: {B_result}")
