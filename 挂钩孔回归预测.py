import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 读取数据集
file_path = 'C:/Users/朱许杨/Desktop/毕业项目/挂钩孔.xlsx'
data = pd.read_excel(file_path)

# 特征编码
label_encoders = {}
for column in ['子/母', '前板/后板', '孔型']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 特征和目标变量
features = ['厚度', '子/母', '前板/后板', '孔型']
targets = ['A11', 'A12', 'A21', 'A22', 'A31', 'A32', 'A51', 'A52', 'A61', 'A62', 'A7', 'B1', 'B2']

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
板材长cm = float(input("请输入板材长cm: "))
板材宽cm = float(input("请输入板材宽cm: "))

# 构建输入数据的 NumPy 数组
input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                                   label_encoders["前板/后板"].transform([plate_input])[0],
                                   label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

# 使用 TensorFlow 模型进行预测
predicted_values = model.predict(input_data_np)[0]

# 输出预测结果
# print(f'预测的A11值:{predicted_values[0]}')
# print(f"预测的A12值:{predicted_values[1]}")
# print(f"预测的A21值:{predicted_values[2]}")
# print(f"预测的A22值:{predicted_values[3]}")
# print(f"预测的A31值:{predicted_values[4]}")
# print(f'预测的A32值：{predicted_values[5]}')
# print(f"预测的A51值: {predicted_values[6]}")
# print(f"预测的A52值: {predicted_values[7]}")
# print(f"预测的A61值: {predicted_values[8]}")
# print(f"预测的A62值: {predicted_values[9]}")
# print(f'预测的A7值：{predicted_values[10]}')
# print(f"预测的B1值: {predicted_values[11]}")
# print(f"预测的B2值: {predicted_values[12]}")

opening_direction = input("请输入开向：")

def calculate_A1_A2(门体结构, 规格宽cm, 板材宽cm, type_input, plate_input, predicted_values):
    A1 = 0
    A2 = 0
    if 板材宽cm >= 801 and 板材宽cm < 1142:
        if type_input == '子' and plate_input == '后板':
            if 门体结构 == "对开门":
                封板结构 = input("请输入封板结构: ")
                if 封板结构 == "无" or 封板结构 == "中间封板":
                    if 规格宽cm * 10 <= 1720:
                        # A1 = A31
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[4]
                        A2 = 板材宽cm - 360 - A1
                    else:
                        # A1 = (规格宽cm * 10 - 1720) / 4 + A32
                        # A2 = 板材宽cm - 360 - A1
                        A1 = (规格宽cm * 10 - 1720) / 4 + predicted_values[5]
                        A2 = 板材宽cm - 360 - A1
                elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                    if 规格宽cm * 10 <= 2130:
                        # A1 = A51
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[6]
                        A2 = 板材宽cm - 360 - A1
                    else:
                        # A1 = A52
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[7]
                        A2 = 板材宽cm - 360 - A1
                elif 封板结构 == "两边封板":
                    if 规格宽cm * 10 <= 2400:
                        # A1 = A61
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[8]
                        A2 = 板材宽cm - 360 - A1
                    else:
                        # A1 = A62
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[9]
                        A2 = 板材宽cm - 360 - A1
            elif 门体结构 == "三开子母门":
                if 规格宽cm * 10 <= 2130:
                    # A1 = A51
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[6]
                    A2 = 板材宽cm - 360 - A1
                else:
                    # A1 = A52
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[7]
                    A2 = 板材宽cm - 360 - A1
            elif 门体结构 == "四开子母门":
                if 规格宽cm * 10 <= 2400:
                    # A1 = A61
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[8]
                    A2 = 板材宽cm - 360 - A1
                else:
                    # A1 = A62
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[9]
                    A2 = 板材宽cm - 360 - A1
            elif 门体结构 == "拼接门":
                if thickness_input > 70:
                    # A1 = A7
                    # A2 = A7
                    A1 = predicted_values[10]
                    A2 = predicted_values[10]
                else:
                    # A1 = 0
                    # A2 = 0
                    A1 = 0
                    A2 = 0
        elif type_input == '母' and plate_input == '后板':
            if 门体结构 == "单门":
                if 规格宽cm * 10 <= 880:
                    # A1 = A11
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[0]
                    A2 = 板材宽cm - 360 - A1
                else:
                    # A1 = 规格宽cm * 5 - 440 + A12
                    # A2 = 板材宽cm - 360 - A1
                    A1 = 规格宽cm * 5 - 440 + predicted_values[1]
                    A2 = 板材宽cm - 360 - A1
            elif 门体结构 == "子母门":
                if 规格宽cm * 10 <= 1250:
                    # A1 = A21
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[2]
                    A2 = 板材宽cm - 360 - A1
                else:
                    # A1 = A22
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[3]
                    A2 = 板材宽cm - 360 - A1
            elif 门体结构 == "对开门":
                封板结构 = input("请输入封板结构: ")
                if 封板结构 == "无" or 封板结构 == "中间封板":
                    if 规格宽cm * 10 <= 1720:
                        # A1 = A31
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[4]
                        A2 = 板材宽cm - 360 - A1
                    else:
                        # A1 = (规格宽cm * 10 - 1720) / 4 + A32
                        # A2 = 板材宽cm - 360 - A1
                        A1 = (规格宽cm * 10 - 1720) / 4 + predicted_values[5]
                        A2 = 板材宽cm - 360 - A1
                elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                    if 规格宽cm * 10 <= 2130:
                        # A1 = A51
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[6]
                        A2 = 板材宽cm - 360 - A1
                    else:
                        # A1 = A52
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[7]
                        A2 = 板材宽cm - 360 - A1
                elif 封板结构 == "两边封板":
                    if 规格宽cm * 10 <= 2400:
                        # A1 = A61
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[8]
                        A2 = 板材宽cm - 360 - A1
                    else:
                        # A1 = A62
                        # A2 = 板材宽cm - 360 - A1
                        A1 = predicted_values[9]
                        A2 = 板材宽cm - 360 - A1
            elif 门体结构 == "三开子母门":
                if 规格宽cm * 10 <= 2130:
                    # A1 = A51
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[6]
                    A2 = 板材宽cm - 360 - A1
                else:
                    # A1 = A52
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[7]
                    A2 = 板材宽cm - 360 - A1
            elif 门体结构 == "四开子母门":
                if 规格宽cm * 10 <= 2400:
                    # A1 = A61
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[8]
                    A2 = 板材宽cm - 360 - A1
                else:
                    # A1 = A62
                    # A2 = 板材宽cm - 360 - A1
                    A1 = predicted_values[9]
                    A2 = 板材宽cm - 360 - A1
    else:
        A1 = 0
        A2 = 0
    return A1, A2

A1_result, A2_result= calculate_A1_A2(门体结构, 规格宽cm, 板材宽cm, type_input, plate_input, predicted_values)

def calculate_B(门体结构, 板材长cm, type_input, plate_input, opening_direction, predicted_values):
    B = 0
    if type_input == '子' and plate_input == '后板':
        if opening_direction == '外右' or opening_direction == '内左':
            if 门体结构 == "拼接门":
                if thickness_input > 70:
                    # B = B2
                    B = predicted_values[12]
                else:
                    B = 0
            else:
                # B = B1
                B = predicted_values[11]
        elif opening_direction == '外左' or opening_direction == '内右':
            if 门体结构 == "拼接门":
                if thickness_input > 70:
                    # B = 板材长cm - B2
                    B = 板材长cm - predicted_values[12]
                else:
                    B = 0
            else:
                # B = 板材长 - B1
                B = 板材长cm - predicted_values[11]
    elif type_input == '母' and plate_input == '后板':
        if opening_direction == '外右' or opening_direction == '内左':
            if 门体结构 == "拼接门":
                B = 0
            else:
                # B = 板材长 - B1
                B = 板材长cm - predicted_values[11]
        elif opening_direction == '外左' or opening_direction == '内右':
            if 门体结构 == "拼接门":
                B = 0
            else:
                # B = B1
                B = predicted_values[11]
    return B

B_result= calculate_B(门体结构, 板材长cm, type_input, plate_input, opening_direction, predicted_values)

print(f"挂钩孔A1的值: {A1_result}")
print(f"挂钩孔A2的值: {A2_result}")
print(f"挂钩孔B的值: {B_result}")
