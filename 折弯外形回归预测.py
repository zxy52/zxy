import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 读取数据集
file_path = 'C:/Users/朱许杨/Desktop/毕业项目/折弯外形.xlsx'
data = pd.read_excel(file_path)

# 特征编码
label_encoders = {}
for column in ['子/母', '前板/后板', '孔型']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 特征和目标变量
features = ['厚度', '子/母', '前板/后板', '孔型']
targets = ['A1', 'A21', 'A22', 'A23', 'A24', 'A3', 'A41', 'A42', 'A43', 'A44', 'A51', 'A52', 'A53', 'A54',
           'A61', 'A62', 'A63', 'A64', 'A8', 'A9', 'B1']

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
test_loss = model.evaluate(X_test_np, y_test_np, verbose=0)
print(f'Test Loss: {test_loss}')

thickness_input = float(input("请输入厚度："))
type_input = input("请输入子/母：")
plate_input = input("请输入前板/后板：")
hole_type_input = input("请输入孔型：")
门体结构 = input("请输入门体结构: ")
规格宽cm = float(input("请输入规格宽cm: "))
规格高cm = float(input("请输入规格高cm: "))

# 构建输入数据的 NumPy 数组
input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                                   label_encoders["前板/后板"].transform([plate_input])[0],
                                   label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

# 使用 TensorFlow 模型进行预测
predicted_values = model.predict(input_data_np)[0]

# 输出预测结果
# print(f'预测的A1值:{predicted_values[0]}')
# print(f'预测的A21值：{predicted_values[1]}')
# print(f"预测的A22值: {predicted_values[2]}")
# print(f"预测的A23值: {predicted_values[3]}")
# print(f"预测的A24值: {predicted_values[4]}")
# print(f'预测的A3值：{predicted_values[5]}')
# print(f'预测的A41值：{predicted_values[6]}')
# print(f"预测的A42值: {predicted_values[7]}")
# print(f"预测的A43值: {predicted_values[8]}")
# print(f"预测的A44值: {predicted_values[9]}")
# print(f'预测的A51值：{predicted_values[10]}')
# print(f"预测的A52值: {predicted_values[11]}")
# print(f"预测的A53值: {predicted_values[12]}")
# print(f"预测的A54值: {predicted_values[13]}")
# print(f'预测的A61值：{predicted_values[14]}')
# print(f"预测的A62值: {predicted_values[15]}")
# print(f"预测的A63值: {predicted_values[16]}")
# print(f"预测的A64值: {predicted_values[17]}")
# print(f"预测的A8值: {predicted_values[18]}")
# print(f"预测的A9值: {predicted_values[19]}")
# print(f"预测的B1值: {predicted_values[20]}")

def calculate_A(门体结构, 规格宽cm, type_input, plate_input, predicted_values):
    A = 0
    if type_input == '子' and plate_input == '前板':
        if 门体结构 == "子母门":
            子门款式宽度 = float(input("请输入子门款式宽度: "))
            if 子门款式宽度 < 230:
                if 规格宽cm * 10 < 1250:
                    # A = 规格宽cm * 10 - 104 - A21
                    A = 规格宽cm * 10 - 104 - predicted_values[1]
                else:
                    # A = 规格宽cm * 10 - 104 - A22
                    A = 规格宽cm * 10 - 104 - predicted_values[2]
            else:
                if 规格宽cm * 10 < 1250:
                    # A = 规格宽cm * 10 - 104 - A23
                    A = 规格宽cm * 10 - 104 - predicted_values[3]
                else:
                    # A = 规格宽cm * 10 - 104 - A24
                    A = 规格宽cm * 10 - 104 - predicted_values[4]
        elif 门体结构 == "对开门":
            封板结构 = input("请输入封板结构: ")
            if 封板结构 == "无":
                # A = (规格宽cm * 10 - 100) / 2 + A3
                A = predicted_values[5] + (规格宽cm * 10 - 100) / 2
            elif 封板结构 == "中间封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 1920:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A42
                        A = predicted_values[7]
                    else:
                        # A = (规格宽cm * 10 - 102 - 274) / 2 + A41
                        A = (规格宽cm * 10 - 102 - 274) / 2 + predicted_values[6]
                elif 规格宽cm * 10 < 2120:
                    # A = A43
                    A = predicted_values[8]
                else:
                    # A = A44
                    A = predicted_values[9]
            elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 1930:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A52
                        A = predicted_values[11]
                    else:
                        # A = (规格宽cm * 10 - 102 - 284) / 2 + A51
                        A = (规格宽cm * 10 - 102 - 284) / 2 + predicted_values[10]
                elif 规格宽cm * 10 < 2130:
                    # A = A53
                    A = predicted_values[12]
                else:
                    # A = A54
                    A = predicted_values[13]
            elif 封板结构 == "两边封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 2200:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A62
                        A = predicted_values[15]
                    else:
                        # A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + A61
                        A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + predicted_values[14]
                elif 规格宽cm * 10 < 2400:
                    # A = A63
                    A = predicted_values[16]
                else:
                    # A = A64
                    A = predicted_values[17]
        elif 门体结构 == "三开子母门":
            子门款式 = input("请输入子门款式: ")
            if 规格宽cm * 10 < 1930:
                if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                    # A = A52
                    A = predicted_values[11]
                else:
                    # A = (规格宽cm * 10 - 102 - 284) / 2 + A51
                    A = (规格宽cm * 10 - 102 - 284) / 2 + predicted_values[10]
            elif 规格宽cm * 10 < 2130:
                # A = A53
                A = predicted_values[12]
            else:
                # A = A54
                A = predicted_values[13]
        elif 门体结构 == "四开子母门":
            子门款式 = input("请输入子门款式: ")
            if 规格宽cm * 10 < 2200:
                if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                    # A = A62
                    A = predicted_values[15]
                else:
                    # A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + A61
                    A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + predicted_values[14]
            elif 规格宽cm * 10 < 2400:
                # A = A63
                A = predicted_values[16]
            else:
                # A = A64
                A = predicted_values[17]
        elif 门体结构 == "三开门":
            # A = (规格宽cm * 10 - 102) / 3 + A8
            A = (规格宽cm * 10 - 102) / 3 + predicted_values[18]
        elif 门体结构 == "四开门":
            # A = (规格宽cm * 10 - 104) / 4 + A9
            A = (规格宽cm * 10 - 104) / 4 + predicted_values[19]
    elif type_input == '母':
        if 门体结构 == "单门":
            # A = 规格宽cm * 10 - A1
            A = 规格宽cm * 10 -  predicted_values[0]
        elif 门体结构 == "子母门":
            子门款式宽度 = float(input("请输入子门款式宽度: "))
            if 子门款式宽度 < 230:
                if 规格宽cm * 10 < 1250:
                    # A = A21
                    A = predicted_values[1]
                else:
                    # A = A22
                    A = predicted_values[2]
            else:
                if 规格宽cm * 10 < 1250:
                    # A = A23
                    A = predicted_values[3]
                else:
                    # A = A24
                    A = predicted_values[4]
        elif 门体结构 == "对开门":
            封板结构 = input("请输入封板结构: ")
            if 封板结构 == "无":
                # A = (规格宽cm * 10 - 100) / 2 + A3
                A = predicted_values[5] + (规格宽cm * 10 - 100) / 2
            elif 封板结构 == "中间封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 1920:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A42
                        A = predicted_values[7]
                    else:
                        # A = (规格宽cm * 10 - 102 - 274) / 2 + A41
                        A = (规格宽cm * 10 - 102 - 274) / 2 + predicted_values[6]
                elif 规格宽cm * 10 < 2120:
                    # A = A43
                    A = predicted_values[8]
                else:
                    # A = A44
                    A = predicted_values[9]
            elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 1930:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A52
                        A = predicted_values[11]
                    else:
                        # A = (规格宽cm * 10 - 102 - 284) / 2 + A51
                        A = (规格宽cm * 10 - 102 - 284) / 2 + predicted_values[10]
                elif 规格宽cm * 10 < 2130:
                    # A = A53
                    A = predicted_values[12]
                else:
                    # A = A54
                    A = predicted_values[13]
            elif 封板结构 == "两边封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 2200:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A62
                        A = predicted_values[15]
                    else:
                        # A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + A61
                        A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + predicted_values[14]
                elif 规格宽cm * 10 < 2400:
                    # A = A63
                    A = predicted_values[16]
                else:
                    # A = A64
                    A = predicted_values[17]
        elif 门体结构 == "三开子母门":
            子门款式 = input("请输入子门款式: ")
            if 规格宽cm * 10 < 1930:
                if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                    # A = A52
                    A = predicted_values[11]
                else:
                    # A = (规格宽cm * 10 - 102 - 284) / 2 + A51
                    A = (规格宽cm * 10 - 102 - 284) / 2 + predicted_values[10]
            elif 规格宽cm * 10 < 2130:
                # A = A53
                A = predicted_values[12]
            else:
                # A = A54
                A = predicted_values[13]
        elif 门体结构 == "四开子母门":
            子门款式 = input("请输入子门款式: ")
            if 规格宽cm * 10 < 2200:
                if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                    # A = A62
                    A = predicted_values[15]
                else:
                    # A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + A61
                    A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + predicted_values[14]
            elif 规格宽cm * 10 < 2400:
                # A = A63
                A = predicted_values[16]
            else:
                # A = A64
                A = predicted_values[17]
        elif 门体结构 == "三开门":
            # A = (规格宽cm * 10 - 102) / 3 + A8
            A = (规格宽cm * 10 - 102) / 3 + predicted_values[18]
        elif 门体结构 == "四开门":
            # A = (规格宽cm * 10 - 104) / 4 + A9
            A = (规格宽cm * 10 - 104) / 4 + predicted_values[19]
    elif type_input == '子' and plate_input == '后板':
        if 门体结构 == "子母门":
            子门款式宽度 = float(input("请输入子门款式宽度: "))
            if 子门款式宽度 < 230:
                if 规格宽cm * 10 < 1250:
                    # A = 规格宽cm * 10 - 104 - A21
                    A = 规格宽cm * 10 - 104 - predicted_values[1]
                else:
                    # A = 规格宽cm * 10 - 104 - A22
                    A = 规格宽cm * 10 - 104 - predicted_values[2]
            else:
                if 规格宽cm * 10 < 1250:
                    # A = 规格宽cm * 10 - 104 - A23
                    A = 规格宽cm * 10 - 104 - predicted_values[3]
                else:
                    # A = 规格宽cm * 10 - 104 - A24
                    A = 规格宽cm * 10 - 104 - predicted_values[4]
        elif 门体结构 == "对开门":
            封板结构 = input("请输入封板结构: ")
            if 封板结构 == "无":
                # A = (规格宽cm * 10 - 100) / 2 + A3
                A = predicted_values[5] + (规格宽cm * 10 - 100) / 2
            elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 1930:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A52
                        A = predicted_values[11]
                    else:
                        # A = (规格宽cm * 10 - 102 - 284) / 2 + A51
                        A = (规格宽cm * 10 - 102 - 284) / 2 + predicted_values[10]
                elif 规格宽cm * 10 < 2130:
                    # A = A53
                    A = predicted_values[12]
                else:
                    # A = A54
                    A = predicted_values[13]
            elif 封板结构 == "两边封板":
                子门款式 = input("请输入子门款式: ")
                if 规格宽cm * 10 < 2200:
                    if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                        # A = A62
                        A = predicted_values[15]
                    else:
                        # A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + A61
                        A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + predicted_values[14]
                elif 规格宽cm * 10 < 2400:
                    # A = A63
                    A = predicted_values[16]
                else:
                    # A = A64
                    A = predicted_values[17]
        elif 门体结构 == "三开子母门":
            子门款式 = input("请输入子门款式: ")
            if 规格宽cm * 10 < 1930:
                if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                    # A = A52
                    A = predicted_values[11]
                else:
                    # A = (规格宽cm * 10 - 102 - 284) / 2 + A51
                    A = (规格宽cm * 10 - 102 - 284) / 2 + predicted_values[10]
            elif 规格宽cm * 10 < 2130:
                # A = A53
                A = predicted_values[12]
            else:
                # A = A54
                A = predicted_values[13]
        elif 门体结构 == "四开子母门":
            子门款式 = input("请输入子门款式: ")
            if 规格宽cm * 10 < 2200:
                if 子门款式 == "平板" or 子门款式 == "条形" or 子门款式 == "GF010" or 子门款式 == "CM23":
                    # A = A62
                    A = predicted_values[15]
                else:
                    # A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + A61
                    A = (规格宽cm * 10 - 104 - 276 - 276) / 2 + predicted_values[14]
            elif 规格宽cm * 10 < 2400:
                # A = A63
                A = predicted_values[16]
            else:
                # A = A64
                A = predicted_values[17]
        elif 门体结构 == "三开门":
            # A = (规格宽cm * 10 - 102) / 3 + A8
            A = (规格宽cm * 10 - 102) / 3 + predicted_values[18]
        elif 门体结构 == "四开门":
            # A = (规格宽cm * 10 - 104) / 4 + A9
            A = (规格宽cm * 10 - 104) / 4 + predicted_values[19]
    return A

A_result = calculate_A(门体结构, 规格宽cm, type_input, plate_input, predicted_values)

def calculate_B(规格高cm, predicted_values):
    # B = 规格高cm * 10 - B1
    B = 规格高cm * 10 - predicted_values[20]
    return B

B_result = calculate_B(规格高cm, predicted_values)

print(f"折弯外形A的值为: {A_result}")
print(f"折弯外形B的值为: {B_result}")
