import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 读取商标孔数据集
file_path = 'C:/Users/朱许杨/Desktop/毕业项目/商标.xlsx'
data = pd.read_excel(file_path)

# 特征编码
label_encoders = {}
for column in ['子/母', '前板/后板', '孔型']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# 特征和目标变量
features = ['厚度', '子/母', '前板/后板', '孔型']
targets = ['A11', 'A12', 'A13', 'A14', 'A15', 'A21', 'A22', 'A23', 'A24', 'A25', 'A31', 'A32', 'A33', 'A34', 'A35',
           'A41', 'A42', 'A43', 'A44', 'A45', 'A51', 'A52', 'A53', 'A54', 'A55', 'A61', 'A62', 'A63', 'A64', 'A65',
           'A5补', 'A6补', 'B1', 'B2', 'B3']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data[features], data[targets], test_size=0.2, random_state=42)

# 转换为 NumPy 数组
X_train_np = X_train.values
y_train_np = y_train.values
X_test_np = X_test.values
y_test_np = y_test.values

# 加载预训练的猫眼孔模型
cat_eye_model = tf.keras.models.load_model('C:/Users/朱许杨/Desktop/毕业项目/my_model.h5')

# 冻结猫眼孔模型的前几层（不训练这些层）
for layer in cat_eye_model.layers[:-1]:  # 保留最后一层以外的所有层
    layer.trainable = False

# 添加商标孔特定的输出层
new_output = tf.keras.layers.Dense(len(targets), activation=None)(cat_eye_model.layers[-2].output)

# 构建新的模型
model = tf.keras.models.Model(inputs=cat_eye_model.input, outputs=new_output)

# 编译新模型
model.compile(optimizer='adam', loss='mse')

# 迁移学习 - 在商标孔数据集上进行训练
model.fit(X_train_np, y_train_np, epochs=10000, verbose=1)

# 用商标孔测试集评估模型
test_loss = model.evaluate(X_test_np, y_test_np, verbose=0)
print(f'Test Loss on 商标孔数据集: {test_loss}')

thickness_input = float(input("请输入厚度："))
type_input = input("请输入子/母：")
plate_input = input("请输入前板/后板：")
hole_type_input = input("请输入孔型：")
门体结构 = input("请输入门体结构: ")
门面款式宽度 = float(input("请输入门面款式宽度: "))
规格宽cm = float(input("请输入规格宽cm: "))
规格高cm = float(input("请输入规格高cm: "))
板材长cm = float(input("请输入板材长cm: "))

# 构建输入数据的 NumPy 数组
input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                                   label_encoders["前板/后板"].transform([plate_input])[0],
                                   label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

# 使用 TensorFlow 模型进行预测
predicted_values = model.predict(input_data_np)[0]

# 输出预测结果
# print(f'预测的A11值:{predicted_values[0]}')
# print(f"预测的A12值:{predicted_values[1]}")
# print(f"预测的A13值:{predicted_values[2]}")
# print(f"预测的A14值:{predicted_values[3]}")
# print(f"预测的A15值:{predicted_values[4]}")
# print(f'预测的A21值：{predicted_values[5]}')
# print(f"预测的A22值: {predicted_values[6]}")
# print(f"预测的A23值: {predicted_values[7]}")
# print(f"预测的A24值: {predicted_values[8]}")
# print(f"预测的A25值: {predicted_values[9]}")
# print(f'预测的A31值：{predicted_values[10]}')
# print(f"预测的A32值: {predicted_values[11]}")
# print(f"预测的A33值: {predicted_values[12]}")
# print(f"预测的A34值: {predicted_values[13]}")
# print(f"预测的A35值: {predicted_values[14]}")
# print(f'预测的A41值：{predicted_values[15]}')
# print(f"预测的A42值: {predicted_values[16]}")
# print(f"预测的A43值: {predicted_values[17]}")
# print(f"预测的A44值: {predicted_values[18]}")
# print(f"预测的A45值: {predicted_values[19]}")
# print(f'预测的A51值：{predicted_values[20]}')
# print(f"预测的A52值: {predicted_values[21]}")
# print(f"预测的A53值: {predicted_values[22]}")
# print(f"预测的A54值: {predicted_values[23]}")
# print(f"预测的A55值: {predicted_values[24]}")
# print(f'预测的A61值：{predicted_values[25]}')
# print(f"预测的A62值: {predicted_values[26]}")
# print(f"预测的A63值: {predicted_values[27]}")
# print(f"预测的A64值: {predicted_values[28]}")
# print(f"预测的A65值: {predicted_values[29]}")
# print(f"预测的A5补值: {predicted_values[30]}")
# print(f"预测的A6补值: {predicted_values[31]}")
# print(f"预测的B1值: {predicted_values[32]}")
# print(f"预测的B2值: {predicted_values[33]}")
# print(f"预测的B3值: {predicted_values[34]}")
opening_direction = input("请输入开向：")

def calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction, predicted_values):
    A = 0
    if type_input == '子' and plate_input == '前板' and (opening_direction == '外左' or opening_direction == '外右'):
        if 门体结构 == "对开门":
            封板结构 = input("请输入封板结构: ")
            if 封板结构 == "无":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1720:
                        # A = (规格宽cm * 10 - 100) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 100) / 4
                    else:
                        # A = A31
                        A = predicted_values[10]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1740:
                        # A = (规格宽cm * 10 - 100) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 100) / 4
                    else:
                        # A = A32
                        A = predicted_values[11]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1760:
                        # A = (规格宽cm * 10 - 100) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 100) / 4
                    else:
                        # A = A33
                        A = predicted_values[12]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1780:
                        # A = (规格宽cm * 10 - 100) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 100) / 4
                    else:
                        # A = A34
                        A = predicted_values[13]
            elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A51
                        A = predicted_values[20]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 封板结构 == "两边封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A61
                        A = predicted_values[25]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A62
                        A = predicted_values[26]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A63
                        A = predicted_values[27]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A64
                        A = predicted_values[28]
        elif 门体结构 == "三开子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A51
                    A = predicted_values[20]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A52
                    A = predicted_values[21]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A53
                    A = predicted_values[22]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A54
                    A = predicted_values[23]
        elif 门体结构 == "四开子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A61
                    A = predicted_values[25]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A62
                    A = predicted_values[26]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A63
                    A = predicted_values[27]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A64
                    A = predicted_values[28]
    elif type_input == '母' and plate_input == '前板' and (opening_direction == '外左' or opening_direction == '外右'):
        if 门体结构 == "单门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 880:
                    # A = (规格宽cm * 10 + A15) / 2 + 12
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 + 12
                else:
                    # A = A11
                    A = predicted_values[0]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 900:
                    # A = (规格宽cm * 10 + A15) / 2 + 12
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 + 12
                else:
                    # A = A12
                    A = predicted_values[1]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 900:
                    # A = (规格宽cm * 10 + A15) / 2 + 12
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 + 12
                else:
                    # A = A13
                    A = predicted_values[2]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 910:
                    # A = (规格宽cm * 10 + A15) / 2 + 12
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 + 12
                else:
                    # A = A14
                    A = predicted_values[3]
        elif 门体结构 == "子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A21
                    A = predicted_values[5]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A22
                    A = predicted_values[6]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A23
                    A = predicted_values[7]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A24
                    A = predicted_values[8]
        elif 门体结构 == "对开门":
            封板结构 = input("请输入封板结构: ")
            if 封板结构 == "无":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1720:
                        # A = (规格宽cm * 5 + A35) / 2 + 12
                        A = (规格宽cm * 5 + predicted_values[14]) / 2 + 12
                    else:
                        # A = A31
                        A = predicted_values[10]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1740:
                        # A = (规格宽cm * 5 + A35) / 2 + 12
                        A = (规格宽cm * 5 + predicted_values[14]) / 2 + 12
                    else:
                        # A = A32
                        A = predicted_values[11]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1760:
                        # A = (规格宽cm * 5 + A35) / 2 + 12
                        A = (规格宽cm * 5 + predicted_values[14]) / 2 + 12
                    else:
                        # A = A33
                        A = predicted_values[12]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1780:
                        # A = (规格宽cm * 5 + A35) / 2 + 12
                        A = (规格宽cm * 5 + predicted_values[14]) / 2 + 12
                    else:
                        # A = A34
                        A = predicted_values[13]
            elif 封板结构 == "中间封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A41
                        A = predicted_values[15]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A42
                        A = predicted_values[16]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A43
                        A = predicted_values[17]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A44
                        A = predicted_values[18]
            elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A51
                        A = predicted_values[20]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 封板结构 == "两边封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A61
                        A = predicted_values[25]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A62
                        A = predicted_values[26]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A63
                        A = predicted_values[27]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A64
                        A = predicted_values[28]
        elif 门体结构 == "三开子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A51
                    A = predicted_values[20]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A52
                    A = predicted_values[21]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A53
                    A = predicted_values[22]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A54
                    A = predicted_values[23]
        elif 门体结构 == "四开子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A61
                    A = predicted_values[25]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A62
                    A = predicted_values[26]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A63
                    A = predicted_values[27]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A64
                    A = predicted_values[28]
    elif type_input == '母' and plate_input == '后板' and (opening_direction == '内左' or opening_direction == '内右'):
        if 门体结构 == "单门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 880:
                    # A = (规格宽cm * 10 + A15) / 2 - 17.5
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 - 17.5
                else:
                    # A = A11
                    A = predicted_values[0]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 900:
                    # A = (规格宽cm * 10 + A15) / 2 - 17.5
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 - 17.5
                else:
                    # A = A12
                    A = predicted_values[1]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 900:
                    # A = (规格宽cm * 10 + A15) / 2 - 17.5
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 - 17.5
                else:
                    # A = A13
                    A = predicted_values[2]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 910:
                    # A = (规格宽cm * 10 + A15) / 2 - 17.5
                    A = (规格宽cm * 10 + predicted_values[4]) / 2 - 17.5
                else:
                    # A = A14
                    A = predicted_values[3]
        elif 门体结构 == "子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A21
                    A = predicted_values[5]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A22
                    A = predicted_values[6]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A23
                    A = predicted_values[7]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 1250:
                    # A = A25
                    A = predicted_values[9]
                else:
                    # A = A24
                    A = predicted_values[8]
        elif 门体结构 == "对开门":
            封板结构 = input("请输入封板结构: ")
            if 封板结构 == "无":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1720:
                        # A = (规格宽cm * 10 ) / 4 + A35 / 2 -17.5
                        A = predicted_values[14] / 2 + (规格宽cm * 10) / 4 - 17.5
                    else:
                        # A = A31
                        A = predicted_values[10]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1740:
                        # A = (规格宽cm * 10 ) / 4 + A35 / 2 -17.5
                        A = predicted_values[14] / 2 + (规格宽cm * 10) / 4 - 17.5
                    else:
                        # A = A32
                        A = predicted_values[11]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1760:
                        # A = (规格宽cm * 10 ) / 4 + A35 / 2 -17.5
                        A = predicted_values[14] / 2 + (规格宽cm * 10) / 4 - 17.5
                    else:
                        # A = A33
                        A = predicted_values[12]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1780:
                        # A = (规格宽cm * 10 ) / 4 + A35 / 2 -17.5
                        A = predicted_values[14] / 2 + (规格宽cm * 10) / 4 - 17.5
                    else:
                        # A = A34
                        A = predicted_values[13]
            elif 封板结构 == "中间封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A41
                        A = predicted_values[15]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A42
                        A = predicted_values[16]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A43
                        A = predicted_values[17]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2120:
                        # A = A45
                        A = predicted_values[19]
                    else:
                        # A = A44
                        A = predicted_values[18]
            elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A51
                        A = predicted_values[20]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 封板结构 == "两边封板":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A61
                        A = predicted_values[25]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A62
                        A = predicted_values[26]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A63
                        A = predicted_values[27]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A64
                        A = predicted_values[28]
        elif 门体结构 == "三开子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A51
                    A = predicted_values[20]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A52
                    A = predicted_values[21]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A53
                    A = predicted_values[22]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 2130:
                    # A = A55
                    A = predicted_values[24]
                else:
                    # A = A54
                    A = predicted_values[23]
        elif 门体结构 == "四开子母门":
            if 门面款式宽度 == 520:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A61
                    A = predicted_values[25]
            elif 门面款式宽度 == 526:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A62
                    A = predicted_values[26]
            elif 门面款式宽度 == 540:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A63
                    A = predicted_values[27]
            elif 门面款式宽度 == 550:
                if 规格宽cm * 10 >= 2400:
                    # A = A65
                    A = predicted_values[29]
                else:
                    # A = A64
                    A = predicted_values[28]
    elif type_input == '子' and plate_input == '后板' and (opening_direction == '内左' or opening_direction == '内右'):
        if 门体结构 == "对开门":
            封板结构 = input("请输入封板结构: ")
            if 封板结构 == "无":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1720:
                        # A = (规格宽cm * 10 - 116) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 116) / 4
                    else:
                        # A = (规格宽cm * 10 - 100) / 2 -405 + A31
                        A = predicted_values[10] + (规格宽cm * 10 - 100) / 2 - 405
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1740:
                        # A = (规格宽cm * 10 - 116) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 116) / 4
                    else:
                        # A = (规格宽cm * 10 - 100) / 2 -408 + A32
                        A = predicted_values[11] + (规格宽cm * 10 - 116) / 2 - 408
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1760:
                        # A = (规格宽cm * 10 - 116) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 116) / 4
                    else:
                        # A = (规格宽cm * 10 - 100) / 2 -415 + A33
                        A = predicted_values[12] + (规格宽cm * 10 - 116) / 2 - 415
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1780:
                        # A = (规格宽cm * 10 - 116) / 4 + A35
                        A = predicted_values[14] + (规格宽cm * 10 - 116) / 4
                    else:
                        # A = (规格宽cm * 10 - 100) / 2 -420 + A34
                        A = predicted_values[13] + (规格宽cm * 10 - 116) / 2 - 420
            elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                子门款式宽度 = float(input("请输入子门款式宽度: "))
                if 门面款式宽度 == 520:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        else:
                            # A = A51
                            A = predicted_values[20]
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        elif 规格宽cm * 10 < 1930:
                            # A = 107 - 965 + 规格宽cm * 5 + 260 + A5补
                            A = 107 - 965 + 规格宽cm * 5 + 260 + predicted_values[30]
                        else:
                            # A = A51
                            A = predicted_values[20]
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        elif 规格宽cm * 10 < 1930:
                            # A = 101 - 965 + 规格宽cm * 5 + 263 + A5补
                            A = 101 - 965 + 规格宽cm * 5 + 263 + predicted_values[30]
                        else:
                            # A = A52
                            A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        elif 规格宽cm * 10 < 1930:
                            # A = 87 - 965 + 规格宽cm * 5 + 270 + A5补
                            A = 87 - 965 + 规格宽cm * 5 + 270 + predicted_values[30]
                        else:
                            # A = A53
                            A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        else:
                            # A = A54
                            A = predicted_values[23]
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55
                            A = predicted_values[24]
                        elif 规格宽cm * 10 < 1930:
                            # A = 77 - 965 + 规格宽cm * 5 + 275 + A5补
                            A = 77 - 965 + 规格宽cm * 5 + 275 + predicted_values[30]
                        else:
                            # A = A54
                            A = predicted_values[23]
            elif 封板结构 == "两边封板":
                子门款式宽度 = float(input("请输入子门款式宽度: "))
                if 门面款式宽度 == 520:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        else:
                            # A = A61
                            A = predicted_values[25]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        elif 规格宽cm * 10 < 2200:
                            # A = 95 - 1100 + 规格宽cm * 5 + 260 + A6补
                            A = 95 - 1100 + 规格宽cm * 5 + 260 + predicted_values[31]
                        else:
                            # A = A61
                            A = predicted_values[25]
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        else:
                            # A = A62
                            A = predicted_values[26]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        elif 规格宽cm * 10 < 2200:
                            # A = 89 - 1100 + 规格宽cm * 5 + 263 + A6补
                            A = 89 - 1100 + 规格宽cm * 5 + 263 + predicted_values[31]
                        else:
                            # A = A62
                            A = predicted_values[26]
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        else:
                            # A = A63
                            A = predicted_values[27]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        elif 规格宽cm * 10 < 2200:
                            # A = 75 - 1100 + 规格宽cm * 5 + 270 + A6补
                            A = 75 - 1100 + 规格宽cm * 5 + 270 + predicted_values[31]
                        else:
                            # A = A64
                            A = predicted_values[27]
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        else:
                            # A = A62
                            A = predicted_values[28]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[29]
                        elif 规格宽cm * 10 < 2200:
                            # A = 65 - 1100 + 规格宽cm * 5 + 275 + A6补
                            A = 65 - 1100 + 规格宽cm * 5 + 275 + predicted_values[31]
                        else:
                            # A = A62
                            A = predicted_values[28]
        elif 门体结构 == "三开子母门":
            子门款式宽度 = float(input("请输入子门款式宽度: "))
            if 门面款式宽度 == 520:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A51
                        A = predicted_values[20]
                else:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    elif 规格宽cm * 10 < 1930:
                        # A = 107 - 965 + 规格宽cm * 5 + 260 + A5补
                        A = 107 - 965 + 规格宽cm * 5 + 260 + predicted_values[30]
                    else:
                        # A = A51
                        A = predicted_values[20]
            elif 门面款式宽度 == 526:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A52
                        A = predicted_values[21]
                else:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    elif 规格宽cm * 10 < 1930:
                        # A = 101 - 965 + 规格宽cm * 5 + 263 + A5补
                        A = 101 - 965 + 规格宽cm * 5 + 263 + predicted_values[30]
                    else:
                        # A = A52
                        A = predicted_values[21]
            elif 门面款式宽度 == 540:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A53
                        A = predicted_values[22]
                else:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    elif 规格宽cm * 10 < 1930:
                        # A = 87 - 965 + 规格宽cm * 5 + 270 + A5补
                        A = 87 - 965 + 规格宽cm * 5 + 270 + predicted_values[30]
                    else:
                        # A = A53
                        A = predicted_values[22]
            elif 门面款式宽度 == 550:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    else:
                        # A = A54
                        A = predicted_values[23]
                else:
                    if 规格宽cm * 10 >= 2130:
                        # A = A55
                        A = predicted_values[24]
                    elif 规格宽cm * 10 < 1930:
                        # A = 77 - 965 + 规格宽cm * 5 + 275 + A5补
                        A = 77 - 965 + 规格宽cm * 5 + 275 + predicted_values[30]
                    else:
                        # A = A54
                        A = predicted_values[23]
        elif 门体结构 == "四开子母门":
            子门款式宽度 = float(input("请输入子门款式宽度: "))
            if 门面款式宽度 == 520:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A61
                        A = predicted_values[25]
                else:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    elif 规格宽cm * 10 < 2200:
                        # A = 95 - 1100 + 规格宽cm * 5 + 260 + A6补
                        A = 95 - 1100 + 规格宽cm * 5 + 260 + predicted_values[31]
                    else:
                        # A = A61
                        A = predicted_values[25]
            elif 门面款式宽度 == 526:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A62
                        A = predicted_values[26]
                else:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    elif 规格宽cm * 10 < 2200:
                        # A = 89 - 1100 + 规格宽cm * 5 + 263 + A6补
                        A = 89 - 1100 + 规格宽cm * 5 + 263 + predicted_values[31]
                    else:
                        # A = A62
                        A = predicted_values[26]
            elif 门面款式宽度 == 540:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A63
                        A = predicted_values[27]
                else:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    elif 规格宽cm * 10 < 2200:
                        # A = 75 - 1100 + 规格宽cm * 5 + 270 + A6补
                        A = 75 - 1100 + 规格宽cm * 5 + 270 + predicted_values[31]
                    else:
                        # A = A64
                        A = predicted_values[27]
            elif 门面款式宽度 == 550:
                if 子门款式宽度 < 220:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    else:
                        # A = A62
                        A = predicted_values[28]
                else:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[29]
                    elif 规格宽cm * 10 < 2200:
                        # A = 65 - 1100 + 规格宽cm * 5 + 275 + A6补
                        A = 65 - 1100 + 规格宽cm * 5 + 275 + predicted_values[31]
                    else:
                        # A = A62
                        A = predicted_values[28]
    return A

A_result = calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction, predicted_values)

def calculate_B(规格高cm, 板材长cm, opening_direction, predicted_values):
    B = 0
    if type_input == '子' and plate_input == '前板':
        if opening_direction == '外右':
            if 规格高cm * 10 >= 1970:
                # B = B1
                B = predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = B2
                B = predicted_values[33]
            else:
                # B = B3
                B = predicted_values[34]
        elif opening_direction == '外左':
            if 规格高cm * 10 >= 1970:
                # B = 板材长 - B1
                B = 板材长cm - predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = 板材长 - B2
                B = 板材长cm - predicted_values[33]
            else:
                # B = 板材长 - B3
                B = 板材长cm - predicted_values[34]
    elif type_input == '母' and plate_input == '前板':
        if opening_direction == '外左':
            if 规格高cm * 10 >= 1970:
                # B = B1
                B = predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = B2
                B = predicted_values[33]
            else:
                # B = B3
                B = predicted_values[34]
        elif opening_direction == '外右':
            if 规格高cm * 10 >= 1970:
                # B = 板材长 - B1
                B = 板材长cm - predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = 板材长 - B2
                B = 板材长cm - predicted_values[33]
            else:
                # B = 板材长 - B3
                B = 板材长cm - predicted_values[34]
    elif type_input == '母' and plate_input == '后板':
        if opening_direction == '内左':
            if 规格高cm * 10 >= 1970:
                # B = B1
                B = predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = B2
                B = predicted_values[33]
            else:
                # B = B3
                B = predicted_values[34]
        elif opening_direction == '内右':
            if 规格高cm * 10 >= 1970:
                # B = 板材长 - B1
                B = 板材长cm - predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = 板材长 - B2
                B = 板材长cm - predicted_values[33]
            else:
                # B = 板材长 - B3
                B = 板材长cm - predicted_values[34]
    elif type_input == '子' and plate_input == '后板':
        if opening_direction == '内右':
            if 规格高cm * 10 >= 1970:
                # B = B1
                B = predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = B2
                B = predicted_values[33]
            else:
                # B = B3
                B = predicted_values[34]
        elif opening_direction == '内左':
            if 规格高cm * 10 >= 1970:
                # B = 板材长 - B1
                B = 板材长cm - predicted_values[32]
            elif 规格高cm * 10 >= 1920:
                # B = 板材长 - B2
                B = 板材长cm - predicted_values[33]
            else:
                # B = 板材长 - B3
                B = 板材长cm - predicted_values[34]
    return B

B_result = calculate_B(规格高cm, 板材长cm, opening_direction, predicted_values)

print(f"商标A的值为: {A_result}")
print(f"商标B的值为: {B_result}")
