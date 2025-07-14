from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
def collect_inputs():
    inputs = {}
    inputs['厚度'] = float(input("请输入厚度："))
    inputs['子/母板'] = input("请输入子/母板：")
    inputs['前板/后板'] = input("请输入前板/后板：")
    inputs['锁型'] = input("请输入锁型：")
    inputs['铰链种类'] = input("请输入铰链种类：")
    inputs['门体结构'] = input("请输入门体结构: ")
    inputs['封板结构'] = input("请输入封板结构: ")
    inputs['门面款式'] = input("请输入门面款式: ")
    inputs['门面款式宽度'] = float(input("请输入门面款式宽度: "))
    inputs['子门款式'] = input("请输入子门款式: ")
    inputs['子门款式宽度'] = float(input("请输入子门款式宽度: "))
    inputs['花纹长度'] = float(input("请输入花纹长度: "))
    inputs['花纹宽度'] = float(input("请输入花纹宽度: "))
    inputs['规格宽cm'] = float(input("请输入规格宽cm: "))
    inputs['规格高cm'] = float(input("请输入规格高cm: "))
    inputs['板材长cm'] = float(input("请输入板材长cm: "))
    inputs['板材宽cm'] = float(input("请输入板材宽cm: "))
    inputs['开向'] = input("请输入开向：")
    return inputs
# thickness_input = float(input("请输入厚度："))
# type_input = input("请输入子/母：")
# plate_input = input("请输入前板/后板：")
# lock_type = input("请输入锁型：")
# hinge_type = input("请输入铰链种类：")
# 门体结构 = input("请输入门体结构: ")
# 封板结构 = input("请输入封板结构: ")
# 门面款式宽度 = float(input("请输入门面款式宽度: "))
# 子门款式 = input("请输入子门款式: ")
# 子门款式宽度 = float(input("请输入子门款式宽度: "))
# 花纹长度 = float(input("请输入花纹长度: "))
# 花纹宽度 = float(input("请输入花纹宽度: "))
# 规格宽cm = float(input("请输入规格宽cm: "))
# 规格高cm = float(input("请输入规格高cm: "))
# 板材长cm = float(input("请输入板材长cm: "))
# 板材宽 = float(input("请输入板材宽: "))
# opening_direction_input = input("请输入开向：")
def 上下插销孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/上下插销孔.xlsx'):
    # 读取数据集
    data = pd.read_excel(file_path)

    # 特征编码
    label_encoders = {}
    for column in ['子/母', '前板/后板', '孔型']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # 特征和目标变量
    features = ['厚度', '子/母', '前板/后板', '孔型']
    targets = ['A1', 'A2', 'B1', 'B2']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[targets], test_size=0.2, random_state=42)

    # 转换为 NumPy 数组
    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values
    y_test_np = y_test.values

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(targets))
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    # 提取输入数据
    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "上下插销孔"
    门体结构 = inputs['门体结构']

    # 构建输入数据的 NumPy 数组
    input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                               label_encoders["前板/后板"].transform([plate_input])[0],
                               label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

    # 使用 TensorFlow 模型进行预测
    predicted_values = model.predict(input_data_np)[0]
    # print(f'TensorFlow 预测的A1值：{predicted_values[0]}')
    # print(f"TensorFlow 预测的A2值: {predicted_values[1]}")
    # print(f"TensorFlow 预测的B1值: {predicted_values[2]}")
    # print(f"TensorFlow 预测的B2值: {predicted_values[3]}")

    # 计算 A 和 B 值
    def calculate_A(门体结构, predicted_values, thickness_input):
        if 门体结构 == '拼接门':
            if thickness_input > 70:
                # A = A1
                A = predicted_values[0]
            else:
                A = 0
        else:
            # A = A2
            A = predicted_values[1]
        return A

    def calculate_B(门体结构, predicted_values, thickness_input):
        if 门体结构 == '拼接门':
            if thickness_input > 70:
                # B = B2
                B = predicted_values[3]
            else:
                B = 0
        else:
            # B = B1
            B = predicted_values[2]
        return B

    A_result = calculate_A(门体结构, predicted_values, thickness_input)
    B_result = calculate_B(门体结构, predicted_values, thickness_input)

    return A_result, B_result
def 中控插销孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/中控插销孔.xlsx'):
    # 读取数据集
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

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(targets))
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    # 从inputs字典中提取用户输入
    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "中控插销孔"
    板材长cm = inputs['板材长cm']
    opening_direction = inputs['开向']

    # 构建输入数据的 NumPy 数组
    input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                               label_encoders["前板/后板"].transform([plate_input])[0],
                               label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

    # 使用 TensorFlow 模型进行预测
    predicted_values = model.predict(input_data_np)[0]

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

    return A_result, B_result
def 主锁孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/主锁孔.xlsx'):
    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    # 数据预处理
    label_encoders = {}
    categorical_columns = ['子/母', '前板/后板', '孔型', '锁型']
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    X = data[['厚度', '子/母', '前板/后板', '孔型', '锁型']]
    y = data[['A', 'B']]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=22)

    # 定义神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),  # 输入特征数为 5，隐藏层神经元个数为 64
        tf.keras.layers.Dense(2)  # 输出 A 和 B 两个值
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    num_epochs = 10000
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=16, verbose=0)

    # 用测试集评估模型
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    # 进行预测
    def predict_A_and_B(thickness, type_input, plate_input, hole_type, lock_type, label_encoders):
        categorical_data = [type_input, plate_input, hole_type, lock_type]
        encoded_data = []
        for i, col in enumerate(['子/母', '前板/后板', '孔型', '锁型']):
            try:
                encoded_val = label_encoders[col].transform([categorical_data[i]])
                encoded_data.extend(encoded_val)
            except ValueError:
                label_encoders[col].fit([categorical_data[i]])
                encoded_val = label_encoders[col].transform([categorical_data[i]])
                encoded_data.extend(encoded_val)

        numerical_data = [thickness]
        scaled_data = scaler.transform([numerical_data + encoded_data])

        predicted = model.predict(scaled_data)
        return predicted[0]

    # 从inputs字典中提取用户输入
    thickness = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type = "主锁孔"
    lock_type = inputs['锁型']
    板材长cm = inputs['板材长cm']
    opening_direction = inputs['开向']

    # 处理lock_type的特殊情况
    if lock_type in ['大插芯锁', '五舌锁', '豪华插芯锁', '霸王锁', '防爆锁', '特能电控锁', '5001', '5003', '5010', '5020', 'A5', 'A6', 'A7']:
        lock_type = '特能锁'

    # 使用训练时的 label_encoders 对象进行预测
    predicted_values = predict_A_and_B(thickness, type_input, plate_input, hole_type, lock_type, label_encoders)

    def calculate_A(predicted_values):
        A = predicted_values[0]
        return A

    A_result = calculate_A(predicted_values)

    def calculate_B(type_input, plate_input, opening_direction, 板材长cm, predicted_values):
        if type_input == '子' and plate_input == '前板' and (opening_direction == '外左' or opening_direction == '内右'):
            B = 板材长cm - predicted_values[1]
        elif type_input == '母' and plate_input == '后板' and (opening_direction == '外左' or opening_direction == '内右'):
            B = 板材长cm - predicted_values[1]
        else:
            B = predicted_values[1]
        return B

    B_result = calculate_B(type_input, plate_input, opening_direction, 板材长cm, predicted_values)

    return A_result, B_result
def 副锁孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/副锁孔.xlsx'):
    # 读取数据集
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

    # 使用模型进行预测
    def predict_a(thickness, type_, plate, hole_type):
        input_data = pd.DataFrame([[thickness, label_encoders["子/母"].transform([type_])[0],
                                    label_encoders["前板/后板"].transform([plate])[0],
                                    label_encoders["孔型"].transform([hole_type])[0]]], columns=features)
        predicted_a = model.predict(input_data).item()
        return predicted_a

    # 从inputs字典中提取用户输入
    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "副锁孔"
    A_result, 主锁孔B = 主锁孔(inputs)

    predicted_value = predict_a(thickness_input, type_input, plate_input, hole_type_input)

    # 计算副锁孔A
    def calculate_A(predicted_value):
        A = predicted_value
        return A

    A_result = calculate_A(predicted_value)

    # 计算副锁孔B1
    def calculate_B1(predicted_value, 主锁孔B):
        threshold = 1  # 设置阈值，表示足够接近零的范围
        if abs(predicted_value) <= threshold:
            B1 = 0
        else:
            B1 = 主锁孔B - 600
        return B1

    B1_result = calculate_B1(predicted_value, 主锁孔B)

    # 计算副锁孔B2
    def calculate_B2(predicted_value, 主锁孔B):
        threshold = 1  # 设置阈值，表示足够接近零的范围
        if abs(predicted_value) <= threshold:
            B2 = 0
        else:
            B2 = 主锁孔B + 600
        return B2

    B2_result = calculate_B2(predicted_value, 主锁孔B)

    return A_result, B1_result, B2_result
def 拉手孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/拉手孔.xlsx'):
    # 读取 Excel 文件
    data = pd.read_excel(file_path)
    # 数据预处理
    label_encoders = {}
    categorical_columns = ['子/母', '前板/后板', '孔型', '锁型']
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    X = data[['厚度', '子/母', '前板/后板', '孔型', '锁型']]
    y = data[['A', 'B']]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=22)
    # 定义神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),  # 输入特征数为 5，隐藏层神经元个数为 100
        tf.keras.layers.Dense(2)  # 输出 A 和 B 两个值
    ])
    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    # 训练模型
    num_epochs = 10000
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=16, verbose=0)
    # 用测试集评估模型
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    # 进行预测
    def predict_A_and_B(thickness, type_input, plate_input, hole_type, lock_type, label_encoders):
        if lock_type in ['5001', '5003', '霸王锁', '插芯锁', '五舌锁']:
            lock_type = '插芯锁'
        if lock_type in ['5010', '5020', '特能锁']:
            lock_type = '特能锁'
        categorical_data = [type_input, plate_input, hole_type, lock_type]
        encoded_data = []
        for i, col in enumerate(['子/母', '前板/后板', '孔型', '锁型']):
            try:
                encoded_val = label_encoders[col].transform([categorical_data[i]])
                encoded_data.extend(encoded_val)
            except ValueError:
                label_encoders[col].fit([categorical_data[i]])
                encoded_val = label_encoders[col].transform([categorical_data[i]])
                encoded_data.extend(encoded_val)

        numerical_data = [thickness]
        scaled_data = scaler.transform([numerical_data + encoded_data])

        predicted = model.predict(scaled_data)
        return predicted[0]

    # 从inputs字典中提取用户输入
    thickness = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type = "拉手孔"
    lock_type = inputs['锁型']
    板材长cm = inputs['板材长cm']
    opening_direction = inputs['开向']

    # 使用训练时的 label_encoders 对象进行预测
    predicted_values = predict_A_and_B(thickness, type_input, plate_input, hole_type, lock_type, label_encoders)

    def calculate_A(predicted_values):
        A = predicted_values[0]
        return A

    A_result = calculate_A(predicted_values)

    def calculate_B(type_input, plate_input, opening_direction, 板材长cm, predicted_values):
        if type_input == '子' and plate_input == '前板' and (opening_direction == '外左' or opening_direction == '内右'):
            B = 板材长cm - predicted_values[1]
        elif type_input == '子' and plate_input == '后板' and (opening_direction == '外右' or opening_direction == '内左'):
            B = 板材长cm - predicted_values[1]
        else:
            B = predicted_values[1]
        return B

    B_result = calculate_B(type_input, plate_input, opening_direction, 板材长cm, predicted_values)

    return A_result, B_result
def 暗铰链孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/暗铰链孔.xlsx'):
    # 读取数据集
    data = pd.read_excel(file_path)

    # 特征编码
    label_encoders = {}
    for column in ['子/母', '前板/后板', '孔型']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # 特征和目标变量
    features = ['厚度', '子/母', '前板/后板', '孔型']
    targets = ['A', 'B', 'C', 'D', 'E']

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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    # 从inputs字典中提取用户输入
    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "暗铰链孔"
    hinge_type = inputs['铰链种类']
    板材长cm = inputs['板材长cm']
    opening_direction = inputs['开向']

    if hinge_type in ['轴承铰链', '轴承铰链304不锈钢', '军工轴承铰链', '金刚轴承铰链', '轴承铰链镀锌', '6052铰链轴承铰链镀锌']:
        hinge_type = '轴承铰链'
    else:
        hinge_type = '其他'

    # 构建输入数据的 NumPy 数组
    input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                                label_encoders["前板/后板"].transform([plate_input])[0],
                                label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

    # 使用 TensorFlow 模型进行预测
    predicted_values = model.predict(input_data_np)[0]

    def calculate_A(hinge_type, predicted_values):
        if hinge_type == '轴承铰链':
            A = predicted_values[0]
        else:
            A = 0
        return A
    A_result = calculate_A(hinge_type, predicted_values)

    def calculate_B(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == '轴承铰链':
            if type_input == '子' and (opening_direction == '外左' or opening_direction == '内右'):
                B = 板材长cm - predicted_values[1]
            elif type_input == '母' and (opening_direction == '外右' or opening_direction == '内左'):
                B = 板材长cm - predicted_values[1]
            else:
                B = predicted_values[1]
        else:
            B = 0
        return B
    B_result = calculate_B(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)

    def calculate_C(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == '轴承铰链':
            if type_input == '子' and (opening_direction == '外左' or opening_direction == '内右'):
                C = 板材长cm - predicted_values[2]
            elif type_input == '母' and (opening_direction == '外右' or opening_direction == '内左'):
                C = 板材长cm - predicted_values[2]
            else:
                C = predicted_values[2]
        else:
            C = 0
        return C
    C_result = calculate_C(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)

    def calculate_D(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == '轴承铰链':
            if type_input == '子' and (opening_direction == '外右' or opening_direction == '内左'):
                D = 板材长cm - predicted_values[3]
            elif type_input == '母' and (opening_direction == '外左' or opening_direction == '内右'):
                D = 板材长cm - predicted_values[3]
            else:
                D = predicted_values[3]
        else:
            D = 0
        return D
    D_result = calculate_D(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)

    def calculate_E(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == '轴承铰链':
            if type_input == '子' and (opening_direction == '外右' or opening_direction == '内左'):
                E = 板材长cm - predicted_values[4]
            elif type_input == '母' and (opening_direction == '外左' or opening_direction == '内右'):
                E = 板材长cm - predicted_values[4]
            else:
                E = predicted_values[4]
        else:
            E = 0
        return E
    E_result = calculate_E(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)
    return A_result, B_result, C_result, D_result, E_result
def 明铰链孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/明铰链孔.xlsx'):
    # 读取数据集
    data = pd.read_excel(file_path)

    # 特征编码
    label_encoders = {}
    for column in ['子/母', '前板/后板', '孔型']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # 特征和目标变量
    features = ['厚度', '子/母', '前板/后板', '孔型']
    targets = ['A', 'B', 'C', 'D', 'E']

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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    # 从inputs字典中提取用户输入
    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "明铰链孔"
    hinge_type = inputs['铰链种类']
    板材长cm = inputs['板材长cm']
    opening_direction = inputs['开向']

    if hinge_type in ['H型明铰链304不锈钢', '液压明铰链', '不锈钢镀青铜液压明铰链', '普通型明铰链304不锈钢', '普通型明铰链', '外门液压明铰链',
                      '内门液压明铰链', '外门普通型明铰链', '内门普通型明铰链', '内外明铰链', '旗形铰链', '普通型明铰链304不锈钢(仿红铜)',
                      '黑金刚液压明铰链']:
        hinge_type = 'H型明铰链'
    else:
        hinge_type = '其他'

    # 构建输入数据的 NumPy 数组
    input_data_np = np.array([[thickness_input, label_encoders["子/母"].transform([type_input])[0],
                               label_encoders["前板/后板"].transform([plate_input])[0],
                               label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

    # 使用 TensorFlow 模型进行预测
    predicted_values = model.predict(input_data_np)[0]

    def calculate_A(hinge_type, predicted_values):
        if hinge_type == 'H型明铰链':
            A = predicted_values[0]
        else:
            A = 0
        return A

    A_result = calculate_A(hinge_type, predicted_values)

    def calculate_B(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == 'H型明铰链':
            if type_input == '子' and (opening_direction == '外左' or opening_direction == '内右'):
                B = 板材长cm - predicted_values[1]
            elif type_input == '母' and (opening_direction == '外右' or opening_direction == '内左'):
                B = 板材长cm - predicted_values[1]
            else:
                B = predicted_values[1]
        else:
            B = 0
        return B

    B_result = calculate_B(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)

    def calculate_C(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == 'H型明铰链':
            if type_input == '子' and (opening_direction == '外左' or opening_direction == '内右'):
                C = 板材长cm - predicted_values[2]
            elif type_input == '母' and (opening_direction == '外右' or opening_direction == '内左'):
                C = 板材长cm - predicted_values[2]
            else:
                C = predicted_values[2]
        else:
            C = 0
        return C

    C_result = calculate_C(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)

    def calculate_D(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == 'H型明铰链':
            if type_input == '子' and (opening_direction == '外右' or opening_direction == '内左'):
                D = 板材长cm - predicted_values[3]
            elif type_input == '母' and (opening_direction == '外左' or opening_direction == '内右'):
                D = 板材长cm - predicted_values[3]
            else:
                D = predicted_values[3]
        else:
            D = 0
        return D

    D_result = calculate_D(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)

    def calculate_E(type_input, opening_direction, 板材长cm, predicted_values, hinge_type):
        if hinge_type == 'H型明铰链':
            if type_input == '子' and (opening_direction == '外右' or opening_direction == '内左'):
                E = 板材长cm - predicted_values[4]
            elif type_input == '母' and (opening_direction == '外左' or opening_direction == '内右'):
                E = 板材长cm - predicted_values[4]
            else:
                E = predicted_values[4]
        else:
            E = 0
        return E

    E_result = calculate_E(type_input, opening_direction, 板材长cm, predicted_values, hinge_type)

    return A_result, B_result, C_result, D_result, E_result
def 硬标(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/硬标.xlsx'):
    # 读取数据集
    data = pd.read_excel(file_path)

    # 特征编码
    label_encoders = {}
    for column in ['子/母', '前板/后板', '孔型']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # 特征和目标变量
    features = ['厚度', '子/母', '前板/后板', '孔型']
    targets = ['A1', 'A21', 'A22', 'A3', 'B1', 'B2']

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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    # 从inputs字典中提取用户输入
    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "硬标"
    门面款式 = inputs['门面款式']
    门体结构 = inputs['门体结构']
    规格宽cm = inputs['规格宽cm']
    规格高cm = inputs['规格高cm']
    板材长cm = inputs['板材长cm']
    opening_direction = inputs['开向']

    # 构建输入数据的 NumPy 数组
    input_data_np = np.array([[thickness_input,
                               label_encoders["子/母"].transform([type_input])[0],
                               label_encoders["前板/后板"].transform([plate_input])[0],
                               label_encoders["孔型"].transform([hole_type_input])[0]]], dtype=np.float32)

    predicted_values = model.predict(input_data_np)[0]

    def calculate_A(门面款式, 门体结构, 规格宽cm, type_input, plate_input, predicted_values, opening_direction):
        A = 0
        if type_input == '子' and plate_input == '前板' and (opening_direction == '外左' or opening_direction == '外右'):
            if 门面款式 == 'GF010' and 门体结构 == '对开门':
                A = predicted_values[0]
            elif 门面款式 == 'L908' and 规格宽cm * 10 >= 1760 and 门体结构 == '对开门':
                A = predicted_values[1]
            elif 门面款式 == 'L908' and 规格宽cm * 10 < 1760 and 门体结构 == '对开门':
                A = predicted_values[2] + 规格宽cm * 5 - 880
            elif 门面款式 == '常规花纹' and 门体结构 == '对开门':
                A = predicted_values[3]
        elif type_input == '子' and plate_input == '后板' and (opening_direction == '内左' or opening_direction == '内右'):
            if 门面款式 == 'GF010' and 门体结构 == '对开门':
                A = predicted_values[0]
            elif 门面款式 == 'L908' and 规格宽cm * 10 >= 1760 and 门体结构 == '对开门':
                A = predicted_values[1]
            elif 门面款式 == 'L908' and 规格宽cm * 10 < 1760 and 门体结构 == '对开门':
                A = predicted_values[2] + 规格宽cm * 5 - 880
            elif 门面款式 == '常规花纹' and 门体结构 == '对开门':
                A = predicted_values[3]
        elif type_input == '母' and plate_input == '前板' and (opening_direction == '外左' or opening_direction == '外右'):
            if 门面款式 == 'GF010':
                A = predicted_values[0]
            elif 门面款式 == 'L908' and 规格宽cm * 10 >= 900:
                A = predicted_values[1]
            elif 门面款式 == 'L908' and 规格宽cm * 10 < 900:
                A = predicted_values[2] + 规格宽cm * 10 - 900
            elif 门面款式 == '常规花纹':
                A = predicted_values[3]
        elif type_input == '母' and plate_input == '后板' and (opening_direction == '内左' or opening_direction == '内右'):
            if 门面款式 == 'GF010':
                A = predicted_values[0]
            elif 门面款式 == 'L908' and 规格宽cm * 10 >= 900:
                A = predicted_values[1]
            elif 门面款式 == 'L908' and 规格宽cm * 10 < 900:
                A = predicted_values[2] + 规格宽cm * 10 - 900
            elif 门面款式 == '常规花纹':
                A = predicted_values[3]
        return A

    A_result = calculate_A(门面款式, 门体结构, 规格宽cm, type_input, plate_input, predicted_values, opening_direction)

    def calculate_B(门体结构, 规格高cm, type_input, plate_input, predicted_values, 板材长cm, opening_direction):
        if type_input == '子' and plate_input == '前板':
            if opening_direction == '外左':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = predicted_values[4]
                else:
                    B = predicted_values[5]
            elif opening_direction == '外右':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = 板材长cm - predicted_values[4]
                else:
                    B = predicted_values[5]
            else:
                B = 0
        elif type_input == '子' and plate_input == '后板':
            if opening_direction == '内左':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = predicted_values[4]
                else:
                    B = predicted_values[5]
            elif opening_direction == '内右':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = 板材长cm - predicted_values[4]
                else:
                    B = predicted_values[5]
            else:
                B = 0
        elif type_input == '母' and plate_input == '前板':
            if opening_direction == '外右':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = predicted_values[4]
                else:
                    B = predicted_values[5]
            elif opening_direction == '外左':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = 板材长cm - predicted_values[4]
                else:
                    B = predicted_values[5]
            else:
                B = 0
        elif type_input == '母' and plate_input == '后板':
            if opening_direction == '内右':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = predicted_values[4]
                else:
                    B = predicted_values[5]
            elif opening_direction == '内左':
                if 规格高cm * 10 >= 1920 and 门体结构 == '对开门':
                    B = 板材长cm - predicted_values[4]
                else:
                    B = predicted_values[5]
            else:
                B = 0
        return B

    B_result = calculate_B(门体结构, 规格高cm, type_input, plate_input, predicted_values, 板材长cm, opening_direction)

    return A_result, B_result
def 猫眼孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/猫眼孔.xlsx'):
    # 读取数据集
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
               'A5补', 'A6补', 'B']

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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "猫眼孔"
    门体结构 = inputs['门体结构']
    门面款式宽度 = inputs['门面款式宽度']
    规格宽cm = inputs['规格宽cm']
    板材长cm = inputs['板材长cm']

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
    # print(f"预测的B值: {predicted_values[32]}")

    opening_direction = inputs['开向']

    def calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, predicted_values):
        if type_input == '子' and plate_input == '前板':
            A = 0
            if 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
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
        elif type_input == '母' and plate_input == '前板':
            A = 0
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
                封板结构 = inputs['封板结构']
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
        elif type_input == '母' and plate_input == '后板':
            A = 0
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
                封板结构 = inputs['封板结构']
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
        elif type_input == '子' and plate_input == '后板':
            A = 0
            if 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
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
                    子门款式宽度 = inputs['子门款式宽度']
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
                    子门款式宽度 = inputs['子门款式宽度']
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
                子门款式宽度 = inputs['子门款式宽度']
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
                子门款式宽度 = inputs['子门款式宽度']
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

    A_result = calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, predicted_values)

    def calculate_B(type_input, plate_input, predicted_values, 板材长cm, opening_direction):
        if type_input == '子' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '内右'):
            B = 板材长cm - predicted_values[32]
        elif type_input == '子' and plate_input == '后板' and (
                opening_direction == '外右' or opening_direction == '内左'):
            B = 板材长cm - predicted_values[32]
        elif type_input == '母' and plate_input == '前板' and (
                opening_direction == '外右' or opening_direction == '内左'):
            B = 板材长cm - predicted_values[32]
        elif type_input == '母' and plate_input == '后板' and (
                opening_direction == '外左' or opening_direction == '内右'):
            B = 板材长cm - predicted_values[32]
        else:
            B = predicted_values[32]
        return B

    B_result = calculate_B(type_input, plate_input, predicted_values, 板材长cm, opening_direction)

    return A_result, B_result
def 商标(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/商标.xlsx'):
    # 读取数据集
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

    # 构建神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(len(targets))
    ])
    # 编译模型
    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "商标"
    门体结构 = inputs['门体结构']
    门面款式宽度 = inputs['门面款式宽度']
    规格宽cm = inputs['规格宽cm']
    规格高cm = inputs['规格高cm']
    板材长cm = inputs['板材长cm']

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
    opening_direction = inputs['开向']

    def calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction, predicted_values):
        A = 0
        if type_input == '子' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '外右'):
            if 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
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
        elif type_input == '母' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '外右'):
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
                封板结构 = inputs['封板结构']
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
        elif type_input == '母' and plate_input == '后板' and (
                opening_direction == '内左' or opening_direction == '内右'):
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
                封板结构 = inputs['封板结构']
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
        elif type_input == '子' and plate_input == '后板' and (
                opening_direction == '内左' or opening_direction == '内右'):
            if 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
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
                    子门款式宽度 = inputs['子门款式宽度']
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
                    子门款式宽度 = inputs['子门款式宽度']
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
                子门款式宽度 = inputs['子门款式宽度']
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
                子门款式宽度 = inputs['子门款式宽度']
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

    A_result = calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction,
                           predicted_values)

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

    return A_result, B_result
def 压纹(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/压纹.xlsx'):
    # 读取数据集
    data = pd.read_excel(file_path)
    # 特征编码
    label_encoders = {}
    for column in ['子/母', '前板/后板', '孔型']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # 特征和目标变量
    features = ['厚度', '子/母', '前板/后板', '孔型']
    targets = ['A11', 'A12', 'A13', 'A14', 'A15', 'A21', 'A22', 'A23', 'A24', 'A25', 'A31', 'A32', 'A33', 'A34', 'A35',
               'A41', 'A42', 'A43', 'A44', 'A45', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58',
               'A61', 'A62', 'A63', 'A64', 'A65', 'A66', 'A67', 'A68', 'A5补', 'A6补', 'B']

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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "压纹"
    门体结构 = inputs['门体结构']
    门面款式宽度 = inputs['门面款式宽度']
    规格宽cm = inputs['规格宽cm']
    板材长cm = inputs['板材长cm']

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
    # print(f"预测的A56值: {predicted_values[25]}")
    # print(f"预测的A57值: {predicted_values[26]}")
    # print(f"预测的A58值: {predicted_values[27]}")
    # print(f'预测的A61值：{predicted_values[28]}')
    # print(f"预测的A62值: {predicted_values[29]}")
    # print(f"预测的A63值: {predicted_values[30]}")
    # print(f"预测的A64值: {predicted_values[31]}")
    # print(f"预测的A65值: {predicted_values[32]}")
    # print(f"预测的A66值: {predicted_values[33]}")
    # print(f"预测的A67值: {predicted_values[34]}")
    # print(f"预测的A68值: {predicted_values[35]}")
    # print(f"预测的A5补值: {predicted_values[36]}")
    # print(f"预测的A6补值: {predicted_values[37]}")
    # print(f"预测的B值: {predicted_values[38]}")

    opening_direction = inputs['开向']

    def calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction, predicted_values):
        A = 0
        if type_input == '子' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '外右'):
            if 门体结构 == "子母门":
                子门款式宽度 = inputs['子门款式宽度']
                花纹宽度 = inputs['花纹宽度']
                if 门面款式宽度 == 520:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A22
                            A = predicted_values[6] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A22
                            A = predicted_values[6] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A23
                            A = predicted_values[7] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A23
                            A = predicted_values[7] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A24
                            A = predicted_values[8] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A24
                            A = predicted_values[8] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A25
                            A = predicted_values[9] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A25
                            A = predicted_values[9] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 5 - 570) / 2 + A32
                            A = predicted_values[11] + (规格宽cm * 5 - 570) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 5 - 576) / 2 + A33
                            A = predicted_values[12] + (规格宽cm * 5 - 576) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 5 - 590) / 2 + A34
                            A = predicted_values[13] + (规格宽cm * 5 - 590) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 5 - 600) / 2 + A35
                            A = predicted_values[14] + (规格宽cm * 5 - 600) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
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
                            # A = A56
                            A = predicted_values[25]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58
                            A = predicted_values[27]
                        else:
                            # A = A54
                            A = predicted_values[23]
                elif 封板结构 == "两边封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        else:
                            # A = A61
                            A = predicted_values[28]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[33]
                        else:
                            # A = A62
                            A = predicted_values[29]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        else:
                            # A = A63
                            A = predicted_values[30]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        else:
                            # A = A64
                            A = predicted_values[31]
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
                        # A = A56
                        A = predicted_values[25]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A57
                        A = predicted_values[26]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A58
                        A = predicted_values[27]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 门体结构 == "四开子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[32]
                    else:
                        # A = A61
                        A = predicted_values[28]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A66
                        A = predicted_values[33]
                    else:
                        # A = A62
                        A = predicted_values[29]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A67
                        A = predicted_values[34]
                    else:
                        # A = A63
                        A = predicted_values[30]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A68
                        A = predicted_values[35]
                    else:
                        # A = A64
                        A = predicted_values[31]
        elif type_input == '母' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '外右'):
            if 门体结构 == "单门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 880:
                        # A = (规格宽cm * 10 + A12 -520) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[1] - 520) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A13 -526) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[2] - 526) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A14 -540) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[3] - 540) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 910:
                        # A = (规格宽cm * 10 + A15 -550) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[4] - 550) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
            elif 门体结构 == "子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1250:
                        # A = A22
                        A = predicted_values[6]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1250:
                        # A = A23
                        A = predicted_values[7]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1250:
                        # A = A24
                        A = predicted_values[8]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1250:
                        # A = A25
                        A = predicted_values[9]
                    else:
                        # A = A21
                        A = predicted_values[5]
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 5 + A32 - 520) / 2 + 12
                            A = (规格宽cm * 5 + predicted_values[11] - 520) / 4 + 12
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 5 + A33 - 526) / 2 + 12
                            A = (规格宽cm * 5 + predicted_values[12] - 526) / 4 + 12
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 5 + A34 - 540) / 2 + 12
                            A = (规格宽cm * 5 + predicted_values[13] - 540) / 4 + 12
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 5 + A35 - 550) / 2 + 12
                            A = (规格宽cm * 5 + predicted_values[14] - 550) / 4 + 12
                        else:
                            # A = A31
                            A = predicted_values[10]
                elif 封板结构 == "中间封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2120:
                            # A = A42
                            A = predicted_values[16]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2120:
                            # A = A43
                            A = predicted_values[17]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2120:
                            # A = A44
                            A = predicted_values[18]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2120:
                            # A = A45
                            A = predicted_values[19]
                        else:
                            # A = A41
                            A = predicted_values[15]
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
                            # A = A56
                            A = predicted_values[25]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2130:
                            # A = A59
                            A = predicted_values[27]
                        else:
                            # A = A54
                            A = predicted_values[23]
                elif 封板结构 == "两边封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        else:
                            # A = A61
                            A = predicted_values[28]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66
                            A = predicted_values[33]
                        else:
                            # A = A62
                            A = predicted_values[29]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        else:
                            # A = A63
                            A = predicted_values[30]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        else:
                            # A = A64
                            A = predicted_values[31]
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
                        # A = A56
                        A = predicted_values[25]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A57
                        A = predicted_values[26]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A59
                        A = predicted_values[27]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 门体结构 == "四开子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[32]
                    else:
                        # A = A61
                        A = predicted_values[28]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A66
                        A = predicted_values[33]
                    else:
                        # A = A62
                        A = predicted_values[29]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A67
                        A = predicted_values[34]
                    else:
                        # A = A63
                        A = predicted_values[30]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A68
                        A = predicted_values[35]
                    else:
                        # A = A64
                        A = predicted_values[31]
        elif type_input == '母' and plate_input == '后板' and (
                opening_direction == '内左' or opening_direction == '内右'):
            if 门体结构 == "单门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 880:
                        # A = (规格宽cm * 10 + A12 - 520) / 2 - 17.5
                        A = (规格宽cm * 10 + predicted_values[1] - 520) / 2 - 17.5
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A13 - 526) / 2 - 17.5
                        A = (规格宽cm * 10 + predicted_values[2] - 526) / 2 - 17.5
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A14 - 540) / 2 - 17.5
                        A = (规格宽cm * 10 + predicted_values[3] - 540) / 2 - 17.5
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 910:
                        # A = (规格宽cm * 10 + A15 - 550) / 2 - 17.5
                        A = (规格宽cm * 10 + predicted_values[4] - 550) / 2 - 17.5
                    else:
                        # A = A11
                        A = predicted_values[0]
            elif 门体结构 == "子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1250:
                        # A = A22
                        A = predicted_values[6]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1250:
                        # A = A23
                        A = predicted_values[7]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1250:
                        # A = A24
                        A = predicted_values[8]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1250:
                        # A = A25
                        A = predicted_values[9]
                    else:
                        # A = A21
                        A = predicted_values[5]
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 10 ) / 4 + A32 / 2 -17.5 - 260
                            A = predicted_values[11] / 2 + (规格宽cm * 10) / 4 - 17.5 - 260
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 10 ) / 4 + A33 / 2 -17.5 - 263
                            A = predicted_values[12] / 2 + (规格宽cm * 10) / 4 - 17.5 - 263
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 10 ) / 4 + A34 / 2 -17.5 - 270
                            A = predicted_values[13] / 2 + (规格宽cm * 10) / 4 - 17.5 - 270
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 10 ) / 4 + A35 / 2 -17.5 - 275
                            A = predicted_values[14] / 2 + (规格宽cm * 10) / 4 - 17.5 - 275
                        else:
                            # A = A31
                            A = predicted_values[10]
                elif 封板结构 == "中间封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2120:
                            # A = A42
                            A = predicted_values[16]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2120:
                            # A = A43
                            A = predicted_values[17]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2120:
                            # A = A44
                            A = predicted_values[18]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2120:
                            # A = A45
                            A = predicted_values[19]
                        else:
                            # A = A41
                            A = predicted_values[15]
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
                            # A = A56
                            A = predicted_values[25]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58
                            A = predicted_values[27]
                        else:
                            # A = A54
                            A = predicted_values[23]
                elif 封板结构 == "两边封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        else:
                            # A = A61
                            A = predicted_values[28]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66
                            A = predicted_values[33]
                        else:
                            # A = A62
                            A = predicted_values[29]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        else:
                            # A = A63
                            A = predicted_values[30]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        else:
                            # A = A64
                            A = predicted_values[31]
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
                        # A = A56
                        A = predicted_values[25]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A57
                        A = predicted_values[26]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A58
                        A = predicted_values[27]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 门体结构 == "四开子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[32]
                    else:
                        # A = A61
                        A = predicted_values[28]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A66
                        A = predicted_values[33]
                    else:
                        # A = A62
                        A = predicted_values[29]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A67
                        A = predicted_values[34]
                    else:
                        # A = A63
                        A = predicted_values[30]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A68
                        A = predicted_values[35]
                    else:
                        # A = A64
                        A = predicted_values[31]
        elif type_input == '子' and plate_input == '后板' and (
                opening_direction == '内左' or opening_direction == '内右'):
            if 门体结构 == "子母门":
                子门款式宽度 = inputs['子门款式宽度']
                板材宽 = inputs['板材宽cm']
                花纹宽度 = inputs['花纹宽度']
                if 门面款式宽度 == 520:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A22  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[6] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A22  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[6] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A23  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[7] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A23  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[7] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A24  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[8] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A24  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[8] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A25  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[9] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A0 = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A0 = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A25  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[9] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
                            A = 板材宽 - 花纹宽度 - A0
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                花纹宽度 = inputs['花纹宽度']
                板材宽 = inputs['板材宽cm']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 10 - 116) / 4 + A32 - 260  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[11] + (规格宽cm * 10 - 116) / 4 - 260
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 665 + A31  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[10] + (规格宽cm * 10 - 100) / 2 - 665
                            A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 10 - 116) / 4 + A33 - 263  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[12] + (规格宽cm * 10 - 116) / 4 - 263
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 671 + A31  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[10] + (规格宽cm * 10 - 116) / 2 - 671
                            A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 10 - 116) / 4 + A34 - 270  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[13] + (规格宽cm * 10 - 116) / 4 - 270
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 685 + A31  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[10] + (规格宽cm * 10 - 116) / 2 - 685
                            A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 10 - 116) / 4 + A35 - 275  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[14] + (规格宽cm * 10 - 116) / 4 - 275
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 695 + A31  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[10] + (规格宽cm * 10 - 116) / 2 - 695
                            A = 板材宽 - 花纹宽度 - A0
                elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                    子门款式宽度 = inputs['子门款式宽度']
                    if 门面款式宽度 == 520:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2130:
                                # A = A55  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[24]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A51  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[20]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2130:
                                # A = A55  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[24]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 1930:
                                # A = 107 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 107 - 965 + 规格宽cm * 5 + predicted_values[36]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A51  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[20]
                                A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 526:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2130:
                                # A = A56  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[25]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A52  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[21]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2130:
                                # A = A56  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[25]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 1930:
                                # A = 101 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 101 - 965 + 规格宽cm * 5 + predicted_values[36]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A52  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[21]
                                A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 540:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2130:
                                # A = A57  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[26]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A53  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[22]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2130:
                                # A = A57  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[26]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 1930:
                                # A = 87 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 87 - 965 + 规格宽cm * 5 + predicted_values[36]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A53  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[22]
                                A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 550:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2130:
                                # A = A58  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[27]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A54  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[23]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2130:
                                # A = A58  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[27]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 1930:
                                # A = 77 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 77 - 965 + 规格宽cm * 5 + predicted_values[36]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A54  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[23]
                                A = 板材宽 - 花纹宽度 - A0
                elif 封板结构 == "两边封板":
                    子门款式宽度 = inputs['子门款式宽度']
                    if 门面款式宽度 == 520:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A65  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[32]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A61  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[28]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A65  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[32]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 2200:
                                # A = 95 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 95 - 1100 + 规格宽cm * 5 + predicted_values[37]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A61  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[28]
                                A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 526:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A66  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[33]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A62  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[29]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A66  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[33]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 2200:
                                # A = 89 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 89 - 1100 + 规格宽cm * 5 + predicted_values[37]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A62  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[29]
                                A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 540:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A67  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[34]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A63  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[30]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A67  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[34]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 2200:
                                # A = 75 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 75 - 1100 + 规格宽cm * 5 + predicted_values[37]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A63  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[30]
                                A = 板材宽 - 花纹宽度 - A0
                    elif 门面款式宽度 == 550:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A68  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[35]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A64  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[31]
                                A = 板材宽 - 花纹宽度 - A0
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A68  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[35]
                                A = 板材宽 - 花纹宽度 - A0
                            elif 规格宽cm * 10 < 2200:
                                # A = 65 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                                A0 = 65 - 1100 + 规格宽cm * 5 + predicted_values[37]
                                A = 板材宽 - 花纹宽度 - A0
                            else:
                                # A = A64  A = 板材宽 - 花纹宽度 - A0
                                A0 = predicted_values[31]
                                A = 板材宽 - 花纹宽度 - A0
            elif 门体结构 == "三开子母门":
                子门款式宽度 = inputs['子门款式宽度']
                花纹宽度 = inputs['花纹宽度']
                板材宽 = inputs['板材宽cm']
                if 门面款式宽度 == 520:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[24]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A51  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[20]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A55  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[24]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 1930:
                            # A = 107 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 107 - 965 + 规格宽cm * 5 + predicted_values[36]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A51  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[20]
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A56  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[25]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A52  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[21]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A56  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[25]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 1930:
                            # A = 101 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 101 - 965 + 规格宽cm * 5 + predicted_values[36]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A52  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[21]
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[26]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A53 A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[22]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[26]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 1930:
                            # A = 87 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 87 - 965 + 规格宽cm * 5 + predicted_values[36]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A53  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[22]
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[27]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A54  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[23]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[27]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 1930:
                            # A = 77 - 965 + 规格宽cm * 5 + A5补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 77 - 965 + 规格宽cm * 5 + predicted_values[36]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A54  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[23]
                            A = 板材宽 - 花纹宽度 - A0
            elif 门体结构 == "四开子母门":
                子门款式宽度 = inputs['子门款式宽度']
                花纹宽度 = inputs['花纹宽度']
                板材宽 = inputs['板材宽cm']
                if 门面款式宽度 == 520:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[32]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A61  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[28]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[32]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 2200:
                            # A = 95 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 95 - 1100 + 规格宽cm * 5 + predicted_values[37]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A61  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[28]
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[33]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A62  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[29]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[33]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 2200:
                            # A = 89 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 89 - 1100 + 规格宽cm * 5 + predicted_values[37]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A62  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[29]
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[34]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A63  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[30]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[34]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 2200:
                            # A = 75 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 75 - 1100 + 规格宽cm * 5 + predicted_values[37]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A63  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[30]
                            A = 板材宽 - 花纹宽度 - A0
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[35]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A64  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[31]
                            A = 板材宽 - 花纹宽度 - A0
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[35]
                            A = 板材宽 - 花纹宽度 - A0
                        elif 规格宽cm * 10 < 2200:
                            # A = 65 - 1100 + 规格宽cm * 5 + A6补  A = 板材宽 - 花纹宽度 - A0
                            A0 = 65 - 1100 + 规格宽cm * 5 + predicted_values[37]
                            A = 板材宽 - 花纹宽度 - A0
                        else:
                            # A = A64  A = 板材宽 - 花纹宽度 - A0
                            A0 = predicted_values[31]
                            A = 板材宽 - 花纹宽度 - A0
        return A

    A_result = calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction,
                           predicted_values)

    def calculate_B(板材长cm, type_input, plate_input, opening_direction, predicted_values):
        花纹长度 = inputs['花纹长度']
        if type_input == '子' and (
                opening_direction == '外左' or opening_direction == '外右') and plate_input == '前板':
            B = (板材长cm - predicted_values[38] - 花纹长度) / 2
        elif type_input == '子' and (
                opening_direction == '内左' or opening_direction == '内右') and plate_input == '后板':
            B = (板材长cm - predicted_values[38] - 花纹长度) / 2
        elif type_input == '母' and (
                opening_direction == '外左' or opening_direction == '外右') and plate_input == '前板':
            B = (板材长cm - predicted_values[38] - 花纹长度) / 2
        elif type_input == '母' and (
                opening_direction == '内左' or opening_direction == '内右') and plate_input == '后板':
            B = (板材长cm - predicted_values[38] - 花纹长度) / 2
        else:
            B = predicted_values[38]
        return B

    B_result = calculate_B(板材长cm, type_input, plate_input, opening_direction, predicted_values)

    return A_result, B_result
def 挂钩孔(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/挂钩孔.xlsx'):
    # 读取数据集挂钩孔
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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "挂钩孔"
    门体结构 = inputs['门体结构']
    规格宽cm = inputs['规格宽cm']
    板材长cm = inputs['板材长cm']
    板材宽cm = inputs['板材宽cm']

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

    opening_direction = inputs['开向']

    def calculate_A1_A2(门体结构, 规格宽cm, 板材宽cm, type_input, plate_input, predicted_values):
        A1 = 0
        A2 = 0
        if 板材宽cm >= 801 and 板材宽cm < 1142:
            if type_input == '子' and plate_input == '后板':
                if 门体结构 == "对开门":
                    封板结构 = inputs['封板结构']
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
                    封板结构 = inputs['封板结构']
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

    A1_result, A2_result = calculate_A1_A2(门体结构, 规格宽cm, 板材宽cm, type_input, plate_input, predicted_values)

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

    B_result = calculate_B(门体结构, 板材长cm, type_input, plate_input, opening_direction, predicted_values)

    return A1_result, A2_result, B_result
def 折弯外形(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/折弯外形.xlsx'):
    # 读取数据集
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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "折弯外形"
    门体结构 = inputs['门体结构']
    规格宽cm = inputs['规格宽cm']
    规格高cm = inputs['规格高cm']

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
                子门款式宽度 = inputs['子门款式宽度']
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
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    # A = (规格宽cm * 10 - 100) / 2 + A3
                    A = predicted_values[5] + (规格宽cm * 10 - 100) / 2
                elif 封板结构 == "中间封板":
                    子门款式 = inputs['子门款式']
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
                    子门款式 = inputs['子门款式']
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
                    子门款式 = inputs['子门款式']
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
                子门款式 = inputs['子门款式']
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
                子门款式 = inputs['子门款式']
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
                A = 规格宽cm * 10 - predicted_values[0]
            elif 门体结构 == "子母门":
                子门款式宽度 = inputs['子门款式宽度']
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
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    # A = (规格宽cm * 10 - 100) / 2 + A3
                    A = predicted_values[5] + (规格宽cm * 10 - 100) / 2
                elif 封板结构 == "中间封板":
                    子门款式 = inputs['子门款式']
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
                    子门款式 = inputs['子门款式']
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
                    子门款式 = inputs['子门款式']
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
                子门款式 = inputs['子门款式']
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
                子门款式 = inputs['子门款式']
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
                子门款式宽度 = inputs['子门款式宽度']
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
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    # A = (规格宽cm * 10 - 100) / 2 + A3
                    A = predicted_values[5] + (规格宽cm * 10 - 100) / 2
                elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                    子门款式 = inputs['子门款式']
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
                    子门款式 = inputs['子门款式']
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
                子门款式 = inputs['子门款式']
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
                子门款式 = inputs['子门款式']
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

    return A_result, B_result
def 折弯偏花(inputs, file_path='C:/Users/朱许杨/Desktop/毕业项目/折弯偏花.xlsx'):
    # 读取数据集
    data = pd.read_excel(file_path)
    # 特征编码
    label_encoders = {}
    for column in ['子/母', '前板/后板', '孔型']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # 特征和目标变量
    features = ['厚度', '子/母', '前板/后板', '孔型']
    targets = ['A11', 'A12', 'A13', 'A14', 'A15', 'A21', 'A22', 'A23', 'A24', 'A25', 'A31', 'A32', 'A33', 'A34', 'A35',
               'A41', 'A42', 'A43', 'A44', 'A45', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58',
               'A61', 'A62', 'A63', 'A64', 'A65', 'A66', 'A67', 'A68', 'B']

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
    model.fit(X_train_np, y_train_np, epochs=10000, verbose=0)

    # 用测试集评估模型
    model.evaluate(X_test_np, y_test_np, verbose=0)

    thickness_input = inputs['厚度']
    type_input = inputs['子/母板']
    plate_input = inputs['前板/后板']
    hole_type_input = "折弯偏花"
    门体结构 = inputs['门体结构']
    门面款式宽度 = inputs['门面款式宽度']
    规格宽cm = inputs['规格宽cm']
    板材长cm = inputs['板材长cm']

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
    # print(f"预测的A56值: {predicted_values[25]}")
    # print(f"预测的A57值: {predicted_values[26]}")
    # print(f"预测的A58值: {predicted_values[27]}")
    # print(f'预测的A61值：{predicted_values[28]}')
    # print(f"预测的A62值: {predicted_values[29]}")
    # print(f"预测的A63值: {predicted_values[30]}")
    # print(f"预测的A64值: {predicted_values[31]}")
    # print(f"预测的A65值: {predicted_values[32]}")
    # print(f"预测的A66值: {predicted_values[33]}")
    # print(f"预测的A67值: {predicted_values[34]}")
    # print(f"预测的A68值: {predicted_values[35]}")
    # print(f"预测的B值: {predicted_values[36]}")

    opening_direction = inputs['开向']

    def calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction, predicted_values):
        A = 0
        if type_input == '子' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '外右'):
            if 门体结构 == "子母门":
                子门款式宽度 = inputs['子门款式宽度']
                花纹宽度 = inputs['花纹宽度']
                if 门面款式宽度 == 520:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A22
                            A = predicted_values[6] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A22
                            A = predicted_values[6] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A23
                            A = predicted_values[7] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A23
                            A = predicted_values[7] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A24
                            A = predicted_values[8] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A24
                            A = predicted_values[8] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 882 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 882 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 982 - 花纹宽度) / 2 + A25
                            A = predicted_values[9] + (规格宽cm * 10 - 982 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 872 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 872 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 972 - 花纹宽度) / 2 + A25
                            A = predicted_values[9] + (规格宽cm * 10 - 972 - 花纹宽度) / 2
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 5 - 570) / 2 + A32
                            A = predicted_values[11] + (规格宽cm * 5 - 570) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 5 - 576) / 2 + A33
                            A = predicted_values[12] + (规格宽cm * 5 - 576) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 5 - 590) / 2 + A34
                            A = predicted_values[13] + (规格宽cm * 5 - 590) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 5 - 600) / 2 + A35
                            A = predicted_values[14] + (规格宽cm * 5 - 600) / 2
                        else:
                            # A = A31
                            A = predicted_values[10]
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
                            # A = A56
                            A = predicted_values[25]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58
                            A = predicted_values[27]
                        else:
                            # A = A54
                            A = predicted_values[23]
                elif 封板结构 == "两边封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        else:
                            # A = A61
                            A = predicted_values[28]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[33]
                        else:
                            # A = A62
                            A = predicted_values[29]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        else:
                            # A = A63
                            A = predicted_values[30]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        else:
                            # A = A64
                            A = predicted_values[31]
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
                        # A = A56
                        A = predicted_values[25]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A57
                        A = predicted_values[26]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A58
                        A = predicted_values[27]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 门体结构 == "四开子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[32]
                    else:
                        # A = A61
                        A = predicted_values[28]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A66
                        A = predicted_values[33]
                    else:
                        # A = A62
                        A = predicted_values[29]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A67
                        A = predicted_values[34]
                    else:
                        # A = A63
                        A = predicted_values[30]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A68
                        A = predicted_values[35]
                    else:
                        # A = A64
                        A = predicted_values[31]
        elif type_input == '母' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '外右'):
            if 门体结构 == "单门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 880:
                        # A = (规格宽cm * 10 + A12 -520) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[1] - 520) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A13 -526) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[2] - 526) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A14 -540) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[3] - 540) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 910:
                        # A = (规格宽cm * 10 + A15 -550) / 2 + 12
                        A = (规格宽cm * 10 + predicted_values[4] - 550) / 2 + 12
                    else:
                        # A = A11
                        A = predicted_values[0]
            elif 门体结构 == "子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1250:
                        # A = A22
                        A = predicted_values[6]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1250:
                        # A = A23
                        A = predicted_values[7]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1250:
                        # A = A24
                        A = predicted_values[8]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1250:
                        # A = A25
                        A = predicted_values[9]
                    else:
                        # A = A21
                        A = predicted_values[5]
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 5 + A32 - 520) / 2
                            A = (规格宽cm * 5 + predicted_values[11] - 520) / 4
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 5 + A33 - 526) / 2
                            A = (规格宽cm * 5 + predicted_values[12] - 526) / 4
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 5 + A34 - 540) / 2
                            A = (规格宽cm * 5 + predicted_values[13] - 540) / 4
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 5 + A35 - 550) / 2
                            A = (规格宽cm * 5 + predicted_values[14] - 550) / 4
                        else:
                            # A = A31
                            A = predicted_values[10]
                elif 封板结构 == "中间封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2120:
                            # A = A42
                            A = predicted_values[16]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2120:
                            # A = A43
                            A = predicted_values[17]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2120:
                            # A = A44
                            A = predicted_values[18]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2120:
                            # A = A45
                            A = predicted_values[19]
                        else:
                            # A = A41
                            A = predicted_values[15]
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
                            # A = A56
                            A = predicted_values[25]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2130:
                            # A = A59
                            A = predicted_values[27]
                        else:
                            # A = A54
                            A = predicted_values[23]
                elif 封板结构 == "两边封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        else:
                            # A = A61
                            A = predicted_values[28]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66
                            A = predicted_values[33]
                        else:
                            # A = A62
                            A = predicted_values[29]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        else:
                            # A = A63
                            A = predicted_values[30]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        else:
                            # A = A64
                            A = predicted_values[31]
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
                        # A = A56
                        A = predicted_values[25]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A57
                        A = predicted_values[26]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A59
                        A = predicted_values[27]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 门体结构 == "四开子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[32]
                    else:
                        # A = A61
                        A = predicted_values[28]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A66
                        A = predicted_values[33]
                    else:
                        # A = A62
                        A = predicted_values[29]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A67
                        A = predicted_values[34]
                    else:
                        # A = A63
                        A = predicted_values[30]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A68
                        A = predicted_values[35]
                    else:
                        # A = A64
                        A = predicted_values[31]
        elif type_input == '母' and plate_input == '后板' and (
                opening_direction == '内左' or opening_direction == '内右'):
            if 门体结构 == "单门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 880:
                        # A = (规格宽cm * 10 + A12 - 520) / 2 - 12.5
                        A = (规格宽cm * 10 + predicted_values[1] - 520) / 2 - 12.5
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A13 - 526) / 2 - 12.5
                        A = (规格宽cm * 10 + predicted_values[2] - 526) / 2 - 12.5
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 900:
                        # A = (规格宽cm * 10 + A14 - 540) / 2 - 12.5
                        A = (规格宽cm * 10 + predicted_values[3] - 540) / 2 - 12.5
                    else:
                        # A = A11
                        A = predicted_values[0]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 910:
                        # A = (规格宽cm * 10 + A15 - 550) / 2 - 12.5
                        A = (规格宽cm * 10 + predicted_values[4] - 550) / 2 - 12.5
                    else:
                        # A = A11
                        A = predicted_values[0]
            elif 门体结构 == "子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 1250:
                        # A = A22
                        A = predicted_values[6]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 1250:
                        # A = A23
                        A = predicted_values[7]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 1250:
                        # A = A24
                        A = predicted_values[8]
                    else:
                        # A = A21
                        A = predicted_values[5]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 1250:
                        # A = A25
                        A = predicted_values[9]
                    else:
                        # A = A21
                        A = predicted_values[5]
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 10 ) / 4 + A32 / 2 -12.5 - 260
                            A = predicted_values[11] / 2 + (规格宽cm * 10) / 4 - 12.5 - 260
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 10 ) / 4 + A33 / 2 -12.5 - 263
                            A = predicted_values[12] / 2 + (规格宽cm * 10) / 4 - 12.5 - 263
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 10 ) / 4 + A34 / 2 -12.5 - 270
                            A = predicted_values[13] / 2 + (规格宽cm * 10) / 4 - 12.5 - 270
                        else:
                            # A = A31
                            A = predicted_values[10]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 10 ) / 4 + A35 / 2 -12.5 - 275
                            A = predicted_values[14] / 2 + (规格宽cm * 10) / 4 - 12.5 - 275
                        else:
                            # A = A31
                            A = predicted_values[10]
                elif 封板结构 == "中间封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2120:
                            # A = A42
                            A = predicted_values[16]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2120:
                            # A = A43
                            A = predicted_values[17]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2120:
                            # A = A44
                            A = predicted_values[18]
                        else:
                            # A = A41
                            A = predicted_values[15]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2120:
                            # A = A45
                            A = predicted_values[19]
                        else:
                            # A = A41
                            A = predicted_values[15]
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
                            # A = A56
                            A = predicted_values[25]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58
                            A = predicted_values[27]
                        else:
                            # A = A54
                            A = predicted_values[23]
                elif 封板结构 == "两边封板":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        else:
                            # A = A61
                            A = predicted_values[28]
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66
                            A = predicted_values[33]
                        else:
                            # A = A62
                            A = predicted_values[29]
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        else:
                            # A = A63
                            A = predicted_values[30]
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        else:
                            # A = A64
                            A = predicted_values[31]
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
                        # A = A56
                        A = predicted_values[25]
                    else:
                        # A = A52
                        A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2130:
                        # A = A57
                        A = predicted_values[26]
                    else:
                        # A = A53
                        A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2130:
                        # A = A58
                        A = predicted_values[27]
                    else:
                        # A = A54
                        A = predicted_values[23]
            elif 门体结构 == "四开子母门":
                if 门面款式宽度 == 520:
                    if 规格宽cm * 10 >= 2400:
                        # A = A65
                        A = predicted_values[32]
                    else:
                        # A = A61
                        A = predicted_values[28]
                elif 门面款式宽度 == 526:
                    if 规格宽cm * 10 >= 2400:
                        # A = A66
                        A = predicted_values[33]
                    else:
                        # A = A62
                        A = predicted_values[29]
                elif 门面款式宽度 == 540:
                    if 规格宽cm * 10 >= 2400:
                        # A = A67
                        A = predicted_values[34]
                    else:
                        # A = A63
                        A = predicted_values[30]
                elif 门面款式宽度 == 550:
                    if 规格宽cm * 10 >= 2400:
                        # A = A68
                        A = predicted_values[35]
                    else:
                        # A = A64
                        A = predicted_values[31]
        elif type_input == '子' and plate_input == '后板' and (
                opening_direction == '内左' or opening_direction == '内右'):
            if 门体结构 == "子母门":
                子门款式宽度 = inputs['子门款式宽度']
                花纹宽度 = inputs['花纹宽度']
                if 门面款式宽度 == 520:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A22
                            A = predicted_values[6] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A22
                            A = predicted_values[6] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A23
                            A = predicted_values[7] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A23
                            A = predicted_values[7] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A24
                            A = predicted_values[8] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A24
                            A = predicted_values[8] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 <= 230:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 860 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 860 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 960 - 花纹宽度) / 2 + A25
                            A = predicted_values[9] + (规格宽cm * 10 - 960 - 花纹宽度) / 2
                    else:
                        if 规格宽cm * 10 < 1250:
                            # A = (规格宽cm * 10 - 850 - 花纹宽度) / 2 + A21
                            A = predicted_values[5] + (规格宽cm * 10 - 850 - 花纹宽度) / 2
                        else:
                            # A = (规格宽cm * 10 - 950 - 花纹宽度) / 2 + A25
                            A = predicted_values[9] + (规格宽cm * 10 - 950 - 花纹宽度) / 2
            elif 门体结构 == "对开门":
                封板结构 = inputs['封板结构']
                if 封板结构 == "无":
                    if 门面款式宽度 == 520:
                        if 规格宽cm * 10 >= 1720:
                            # A = (规格宽cm * 10 - 116) / 4 + A32 - 260
                            A = predicted_values[11] + (规格宽cm * 10 - 116) / 4 - 260
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 665 + A31
                            A = predicted_values[10] + (规格宽cm * 10 - 100) / 2 - 665
                    elif 门面款式宽度 == 526:
                        if 规格宽cm * 10 >= 1740:
                            # A = (规格宽cm * 10 - 116) / 4 + A33 - 263
                            A = predicted_values[12] + (规格宽cm * 10 - 116) / 4 - 263
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 671 + A31
                            A = predicted_values[10] + (规格宽cm * 10 - 116) / 2 - 671
                    elif 门面款式宽度 == 540:
                        if 规格宽cm * 10 >= 1760:
                            # A = (规格宽cm * 10 - 116) / 4 + A34 - 270
                            A = predicted_values[13] + (规格宽cm * 10 - 116) / 4 - 270
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 685 + A31
                            A = predicted_values[10] + (规格宽cm * 10 - 116) / 2 - 685
                    elif 门面款式宽度 == 550:
                        if 规格宽cm * 10 >= 1780:
                            # A = (规格宽cm * 10 - 116) / 4 + A35 - 275
                            A = predicted_values[14] + (规格宽cm * 10 - 116) / 4 - 275
                        else:
                            # A = (规格宽cm * 10 - 100) / 2 - 695 + A31
                            A = predicted_values[10] + (规格宽cm * 10 - 116) / 2 - 695
                elif 封板结构 == "左边封板" or 封板结构 == "右边封板":
                    子门款式宽度 = inputs['子门款式宽度']
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
                                # A = 107 - 965 + 规格宽cm * 5
                                A = 107 - 965 + 规格宽cm * 5
                            else:
                                # A = A51
                                A = predicted_values[20]
                    elif 门面款式宽度 == 526:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2130:
                                # A = A56
                                A = predicted_values[25]
                            else:
                                # A = A52
                                A = predicted_values[21]
                        else:
                            if 规格宽cm * 10 >= 2130:
                                # A = A56
                                A = predicted_values[25]
                            elif 规格宽cm * 10 < 1930:
                                # A = 101 - 965 + 规格宽cm * 5
                                A = 101 - 965 + 规格宽cm * 5
                            else:
                                # A = A52
                                A = predicted_values[21]
                    elif 门面款式宽度 == 540:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2130:
                                # A = A57
                                A = predicted_values[26]
                            else:
                                # A = A53
                                A = predicted_values[22]
                        else:
                            if 规格宽cm * 10 >= 2130:
                                # A = A57
                                A = predicted_values[26]
                            elif 规格宽cm * 10 < 1930:
                                # A = 87 - 965 + 规格宽cm * 5
                                A = 87 - 965 + 规格宽cm * 5
                            else:
                                # A = A53
                                A = predicted_values[22]
                    elif 门面款式宽度 == 550:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2130:
                                # A = A58
                                A = predicted_values[27]
                            else:
                                # A = A54
                                A = predicted_values[23]
                        else:
                            if 规格宽cm * 10 >= 2130:
                                # A = A58
                                A = predicted_values[27]
                            elif 规格宽cm * 10 < 1930:
                                # A = 77 - 965 + 规格宽cm * 5 + A5补
                                A = 77 - 965 + 规格宽cm * 5
                            else:
                                # A = A54
                                A = predicted_values[23]
                elif 封板结构 == "两边封板":
                    子门款式宽度 = inputs['子门款式宽度']
                    if 门面款式宽度 == 520:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A65
                                A = predicted_values[32]
                            else:
                                # A = A61
                                A = predicted_values[28]
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A65
                                A = predicted_values[32]
                            elif 规格宽cm * 10 < 2200:
                                # A = 95 - 1100 + 规格宽cm * 5
                                A = 95 - 1100 + 规格宽cm * 5
                            else:
                                # A = A61
                                A = predicted_values[28]
                    elif 门面款式宽度 == 526:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A66
                                A = predicted_values[33]
                            else:
                                # A = A62
                                A = predicted_values[29]
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A66
                                A = predicted_values[33]
                            elif 规格宽cm * 10 < 2200:
                                # A = 89 - 1100 + 规格宽cm * 5
                                A = 89 - 1100 + 规格宽cm * 5
                            else:
                                # A = A62
                                A = predicted_values[29]
                    elif 门面款式宽度 == 540:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A67
                                A = predicted_values[34]
                            else:
                                # A = A63
                                A = predicted_values[30]
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A67
                                A = predicted_values[34]
                            elif 规格宽cm * 10 < 2200:
                                # A = 75 - 1100 + 规格宽cm * 5
                                A = 75 - 1100 + 规格宽cm * 5
                            else:
                                # A = A63
                                A = predicted_values[30]
                    elif 门面款式宽度 == 550:
                        if 子门款式宽度 < 220:
                            if 规格宽cm * 10 >= 2400:
                                # A = A68
                                A = predicted_values[35]
                            else:
                                # A = A64
                                A = predicted_values[31]
                        else:
                            if 规格宽cm * 10 >= 2400:
                                # A = A68
                                A = predicted_values[35]
                            elif 规格宽cm * 10 < 2200:
                                # A = 65 - 1100 + 规格宽cm * 5
                                A = 65 - 1100 + 规格宽cm * 5
                            else:
                                # A = A64
                                A = predicted_values[31]
            elif 门体结构 == "三开子母门":
                子门款式宽度 = inputs['子门款式宽度']
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
                            # A = 107 - 965 + 规格宽cm * 5
                            A = 107 - 965 + 规格宽cm * 5
                        else:
                            # A = A51
                            A = predicted_values[20]
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A56
                            A = predicted_values[25]
                        else:
                            # A = A52
                            A = predicted_values[21]
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A56
                            A = predicted_values[25]
                        elif 规格宽cm * 10 < 1930:
                            # A = 101 - 965 + 规格宽cm * 5
                            A = 101 - 965 + 规格宽cm * 5
                        else:
                            # A = A52
                            A = predicted_values[21]
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        else:
                            # A = A53
                            A = predicted_values[22]
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A57
                            A = predicted_values[26]
                        elif 规格宽cm * 10 < 1930:
                            # A = 87 - 965 + 规格宽cm * 5
                            A = 87 - 965 + 规格宽cm * 5
                        else:
                            # A = A53
                            A = predicted_values[22]
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58
                            A = predicted_values[27]
                        else:
                            # A = A54
                            A = predicted_values[23]
                    else:
                        if 规格宽cm * 10 >= 2130:
                            # A = A58
                            A = predicted_values[27]
                        elif 规格宽cm * 10 < 1930:
                            # A = 77 - 965 + 规格宽cm * 5 + A5补
                            A = 77 - 965 + 规格宽cm * 5
                        else:
                            # A = A54
                            A = predicted_values[23]
            elif 门体结构 == "四开子母门":
                子门款式宽度 = inputs['子门款式宽度']
                if 门面款式宽度 == 520:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        else:
                            # A = A61
                            A = predicted_values[28]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A65
                            A = predicted_values[32]
                        elif 规格宽cm * 10 < 2200:
                            # A = 95 - 1100 + 规格宽cm * 5
                            A = 95 - 1100 + 规格宽cm * 5
                        else:
                            # A = A61
                            A = predicted_values[28]
                elif 门面款式宽度 == 526:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66
                            A = predicted_values[33]
                        else:
                            # A = A62
                            A = predicted_values[29]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A66
                            A = predicted_values[33]
                        elif 规格宽cm * 10 < 2200:
                            # A = 89 - 1100 + 规格宽cm * 5
                            A = 89 - 1100 + 规格宽cm * 5
                        else:
                            # A = A62
                            A = predicted_values[29]
                elif 门面款式宽度 == 540:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        else:
                            # A = A63
                            A = predicted_values[30]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A67
                            A = predicted_values[34]
                        elif 规格宽cm * 10 < 2200:
                            # A = 75 - 1100 + 规格宽cm * 5
                            A = 75 - 1100 + 规格宽cm * 5
                        else:
                            # A = A63
                            A = predicted_values[30]
                elif 门面款式宽度 == 550:
                    if 子门款式宽度 < 220:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        else:
                            # A = A64
                            A = predicted_values[31]
                    else:
                        if 规格宽cm * 10 >= 2400:
                            # A = A68
                            A = predicted_values[35]
                        elif 规格宽cm * 10 < 2200:
                            # A = 65 - 1100 + 规格宽cm * 5
                            A = 65 - 1100 + 规格宽cm * 5
                        else:
                            # A = A64
                            A = predicted_values[31]
        return A

    A_result = calculate_A(门面款式宽度, 门体结构, 规格宽cm, type_input, plate_input, opening_direction,
                           predicted_values)

    def calculate_B(板材长cm, predicted_values):
        if type_input == '母' and plate_input == '前板' and (
                opening_direction == '外左' or opening_direction == '外右'):
            花纹长度 = inputs['花纹长度']
            B = (板材长cm - predicted_values[38] - 花纹长度) / 2
        else:
            B = predicted_values[36]
        return B

    B_result = calculate_B(板材长cm, predicted_values)
    return A_result, B_result

def main():
    inputs = collect_inputs()  # 收集所有的输入数据
    # 调用上下插销孔函数
    A_result, B_result = 上下插销孔(inputs)
    print(f"上下插销孔A的值为: {A_result}")
    print(f"上下插销孔B的值为: {B_result}")
    # 调用中控插销孔函数
    A_result, B_result = 中控插销孔(inputs)
    print(f"中控插销孔A的值为: {A_result}")
    print(f"中控插销孔B的值为: {B_result}")
    # 调用主锁孔函数
    A_result, B_result = 主锁孔(inputs)
    print(f"主锁孔A的值为: {A_result}")
    print(f"主锁孔B的值为: {B_result}")
    # 调用副锁孔函数
    A_result, B1_result, B2_result = 副锁孔(inputs)
    print(f'副锁孔A的值为：{A_result}')
    print(f'副锁孔B1的值为：{B1_result}')
    print(f'副锁孔B2的值为：{B2_result}')
    # 调用拉手孔函数
    A_result, B_result = 拉手孔(inputs)
    print(f"拉手孔A的值为: {A_result}")
    print(f"拉手孔B的值为: {B_result}")
    # 调用暗铰链孔函数
    A_result, B_result, C_result, D_result, E_result = 暗铰链孔(inputs)
    print(f"暗铰链孔A的值为: {A_result}")
    print(f"暗铰链孔B的值为: {B_result}")
    print(f"暗铰链孔C的值为: {C_result}")
    print(f"暗铰链孔D的值为: {D_result}")
    print(f"暗铰链孔E的值为: {E_result}")
    # 调用明铰链孔函数
    A_result, B_result, C_result, D_result, E_result = 明铰链孔(inputs)
    print(f"明铰链孔A的值为: {A_result}")
    print(f"明铰链孔B的值为: {B_result}")
    print(f"明铰链孔C的值为: {C_result}")
    print(f"明铰链孔D的值为: {D_result}")
    print(f"明铰链孔E的值为: {E_result}")
    # 调用硬标函数
    A_result, B_result = 硬标(inputs)
    print(f"硬标A的值为: {A_result}")
    print(f"硬标B的值为: {B_result}")
    # 调用猫眼孔函数
    A_result, B_result = 猫眼孔(inputs)
    print(f"猫眼孔A的值为: {A_result}")
    print(f"猫眼孔B的值为: {B_result}")
    # 调用商标函数
    A_result, B_result = 商标(inputs)
    print(f"商标A的值为: {A_result}")
    print(f"商标B的值为: {B_result}")
    # 调用压纹函数
    A_result, B_result = 压纹(inputs)
    print(f"压纹A的值为: {A_result}")
    print(f"压纹B的值为: {B_result}")
    # 调用挂钩孔函数
    A1_result, A2_result, B_result = 挂钩孔(inputs)
    print(f'挂钩孔A1的值为：{A1_result}')
    print(f'挂钩孔A2的值为：{A2_result}')
    print(f'挂钩孔B的值为：{B_result}')
    # 调用折弯外形函数
    A_result, B_result = 折弯外形(inputs)
    print(f"折弯外形A的值为: {A_result}")
    print(f"折弯外形B的值为: {B_result}")
    # 调用折弯偏花函数
    A_result, B_result = 折弯偏花(inputs)
    print(f"折弯偏花A的值为: {A_result}")
    print(f"折弯偏花B的值为: {B_result}")

if __name__ == "__main__":
    main()
