import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 读取数据集
file_path = 'C:/Users/朱许杨/Desktop/毕业项目/副锁孔.xlsx'
data = pd.read_excel(file_path)

# 特征编码
label_encoders = {}
for column in ['子/母', '前板/后板', '孔型']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 特征和目标变量
features = ['厚度', '子/母', '前板/后板', '孔型']
target = ['A']

X = data[features]
y = data[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=10000, batch_size=32, verbose=0)

# 用测试集评估模型
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# 使用模型进行预测
def predict_a(thickness, type_, plate, hole_type):
    input_data = pd.DataFrame([[thickness, label_encoders["子/母"].transform([type_])[0],
                                label_encoders["前板/后板"].transform([plate])[0],
                                label_encoders["孔型"].transform([hole_type])[0]]], columns=features)
    predicted_a = model.predict(input_data).item()
    return predicted_a

# 输入特征值进行预测
thickness_input = float(input("请输入厚度："))
type_input = input("请输入子/母：")
plate_input = input("请输入前板/后板：")
hole_type_input = input("请输入孔型：")
主锁孔B = float(input("请输入主锁孔B的值: "))

predicted_value = predict_a(thickness_input, type_input, plate_input, hole_type_input)

def calculate_A(predicted_value):
    A = predicted_value
    return A

A_result = calculate_A(predicted_value)
def calculate_B1(predicted_value, 主锁孔B):
    threshold = 1  # 设置阈值，表示足够接近零的范围
    if abs(predicted_value) <= threshold:
        B1 = 0
    else:
        B1 = 主锁孔B - 600
    return B1

B1_result = calculate_B1(predicted_value, 主锁孔B)
def calculate_B2(predicted_value, 主锁孔B):
    threshold = 1  # 设置阈值，表示足够接近零的范围
    if abs(predicted_value) <= threshold:
        B2 = 0
    else:
        B2 = 主锁孔B + 600
    return B2

B2_result = calculate_B2(predicted_value, 主锁孔B)

print(f'副锁孔A的值为：{A_result}')
print(f'副锁孔B1的值为：{B1_result}')
print(f'副锁孔B2的值为：{B2_result}')
