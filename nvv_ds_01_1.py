"""
充电时间预测模型训练脚本
使用线性回归 + 独热编码
数据集：nvv2t.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# 获取当前脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'time_model.pkl')

# 分类变量
CATEGORICAL_COLUMNS = ['weekday', 'platform', 'facilityType']


def load_and_prepare_data():
    """加载并准备数据"""
    csv_path = os.path.join(DATA_DIR, 'nvv2t.csv')
    hdfs_path = '/Car/nvv2t.csv'

    # 尝试本地读取
    if os.path.exists(csv_path):
        print(f"[nvv_ds_01_1] 从本地读取: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[nvv_ds_01_1] 从 HDFS 读取: {hdfs_path}")
        from hdfs import InsecureClient
        client = InsecureClient('http://localhost:9870', user='root')
        with client.read(hdfs_path, encoding='utf-8') as reader:
            df = pd.read_csv(reader)

    print(f"[nvv_ds_01_1] 原始数据行数: {len(df)}")

    # 数据清洗
    before = len(df)
    df = df.dropna()
    df = df.drop(columns=['sessionId', 'created', 'ended', 'userId', 'stationId', 'locationId'], errors='ignore')
    after = len(df)
    print(f"[nvv_ds_01_1] 清洗后数据行数: {after} (剔除 {before - after} 条)")

    # 独热编码
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # 特征列（除了目标列）
    target_col = 'chargeTimeHrs'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print(f"[nvv_ds_01_1] 特征维度: {X.shape}")
    print(f"[nvv_ds_01_1] 特征列: {feature_cols}")

    return X, y, feature_cols


def train_model(X, y):
    """训练线性回归模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[nvv_ds_01_1] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[nvv_ds_01_1] 测试集 MSE: {mse:.6f}")
    print(f"[nvv_ds_01_1] 测试集 RMSE: {np.sqrt(mse):.6f}")

    return model


def save_model(model, feature_cols):
    """保存模型"""
    os.makedirs(DATA_DIR, exist_ok=True)
    model_package = {
        'model': model,
        'feature_cols': feature_cols,
        'categorical_columns': CATEGORICAL_COLUMNS
    }
    joblib.dump(model_package, MODEL_PATH)
    print(f"[nvv_ds_01_1] 模型已保存至: {MODEL_PATH}")


def predict_charge_time(data_point):
    """
    充电时间推理函数
    参数: data_point - dict 包含特征数据
    返回: 预测的充电时间（小时）
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    feature_cols = model_package['feature_cols']
    categorical_columns = model_package['categorical_columns']

    # 转换为DataFrame
    df = pd.DataFrame([data_point])

    # 独热编码
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # 填充缺失的独热编码列
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # 确保特征顺序一致
    X = df[feature_cols]

    # 预测
    charge_time = model.predict(X)[0]
    return float(max(0, charge_time))  # 确保非负


if __name__ == '__main__':
    print("=" * 60)
    print("充电时间预测模型训练")
    print("=" * 60)

    # 加载数据
    X, y, feature_cols = load_and_prepare_data()

    # 训练模型
    model = train_model(X, y)

    # 保存模型
    save_model(model, feature_cols)

    # 测试预测
    test_data = {
        'kwhTotal': 8.5,
        'charging_fees': 0,
        'startTime': 15,
        'endTime': 17,
        'managerVehicle': 0,
        'weekday': 'Tue',
        'platform': 'android',
        'facilityType': 3
    }
    predicted_time = predict_charge_time(test_data)
    print(f"[nvv_ds_01_1] 测试预测 - 输入: {test_data}")
    print(f"[nvv_ds_01_1] 测试预测 - 输出: {predicted_time:.2f} 小时")

    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
