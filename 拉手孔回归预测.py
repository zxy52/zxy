import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

# 禁用特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
# 读取 Excel 文件
file_path = 'C:/Users/朱许杨/Desktop/毕业项目/拉手孔.xlsx'
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
    tf.keras.layers.Dense(100, activation='relu', input_shape=(5,)),  # 输入特征数为 5，隐藏层神经元个数为 100
    tf.keras.layers.Dense(2)  # 输出 A 和 B 两个值
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练模型
num_epochs = 10000
model.fit(X_train, y_train, epochs=num_epochs, batch_size=16, verbose=1)

# 用测试集评估模型
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}')

# 进行预测
def predict_A_and_B(thickness, type_input, plate_input, hole_type, lock_type, label_encoders):
    if lock_type in ['5001', '5003', '霸王锁', '插芯锁', '五舌锁']:
        lock_type = '插芯锁'
    if lock_type in ['5010', '5020', '特能锁']:
        lock_type= '特能锁'
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

# 用户输入进行预测
thickness = float(input("请输入厚度："))
type_input = input("请输入子/母：")
plate_input = input("请输入前板/后板：")
hole_type = input("请输入孔型：")
lock_type = input("请输入锁型：")
板材长cm = float(input("请输入板材长cm: "))
opening_direction = input("请输入开向：")


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

print(f"拉手孔A的值为: {A_result}")
print(f"拉手孔B的值为: {B_result}")
