"""
平台选择预测模型训练脚本
使用随机森林分类器 + StandardScaler标准化
目标列：platform (0-iOS, 1-Android)
数据集：nvv2t.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 获取当前脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'platform_model.pkl')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler.pkl')

# 分类变量
CATEGORICAL_COLUMNS = ['weekday', 'facilityType']


def load_and_prepare_data():
    """加载并准备数据"""
    csv_path = os.path.join(DATA_DIR, 'nvv2t.csv')
    hdfs_path = '/Car/nvv2t.csv'

    # 尝试本地读取
    if os.path.exists(csv_path):
        print(f"[nvv_ds_03_1] 从本地读取: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[nvv_ds_03_1] 从 HDFS 读取: {hdfs_path}")
        from hdfs import InsecureClient
        client = InsecureClient('http://localhost:9870', user='root')
        with client.read(hdfs_path, encoding='utf-8') as reader:
            df = pd.read_csv(reader)

    print(f"[nvv_ds_03_1] 原始数据行数: {len(df)}")

    # 数据清洗
    before = len(df)
    df = df.dropna()
    df = df.drop(columns=['sessionId', 'created', 'ended'], errors='ignore')
    after = len(df)
    print(f"[nvv_ds_03_1] 清洗后数据行数: {after} (剔除 {before - after} 条)")

    # 平台映射：0-iOS, 1-Android
    # 注意：根据实际数据调整映射
    if 'platform' in df.columns:
        # 尝试字符串映射
        if df['platform'].dtype == 'object':
            df['platform'] = df['platform'].map({'android': 1, 'ios': 0, 'IOS': 0})
        else:
            # 如果已经是数值，保持原样或按需求转换
            df['platform'] = df['platform'].apply(lambda x: 1 if x in [1, 'android', 'Android'] else 0)

    # 独热编码
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # 特征列（除了目标列）
    target_col = 'platform'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print(f"[nvv_ds_03_1] 特征维度: {X.shape}")
    print(f"[nvv_ds_03_1] 特征列: {feature_cols}")
    print(f"[nvv_ds_03_1] 目标分布:\n{y.value_counts()}")

    return X, y, feature_cols


def train_model(X, y):
    """训练随机森林分类模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[nvv_ds_03_1] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[nvv_ds_03_1] 数据标准化完成")

    # 训练随机森林分类器
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    print("[nvv_ds_03_1] 模型训练完成")

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"[nvv_ds_03_1] 准确率: {accuracy:.4f}")
    print(f"[nvv_ds_03_1] 分类报告:\n{report}")

    return model, scaler, accuracy


def save_model(model, scaler, feature_cols):
    """保存模型和标准化器"""
    os.makedirs(DATA_DIR, exist_ok=True)

    # 保存模型
    model_package = {
        'model': model,
        'feature_cols': feature_cols,
        'categorical_columns': CATEGORICAL_COLUMNS
    }
    joblib.dump(model_package, MODEL_PATH)
    print(f"[nvv_ds_03_1] 模型已保存至: {MODEL_PATH}")

    # 保存标准化器
    joblib.dump(scaler, SCALER_PATH)
    print(f"[nvv_ds_03_1] 标准化器已保存至: {SCALER_PATH}")


def predict_platform(data_point):
    """
    平台选择推理函数
    参数: data_point - dict 包含特征数据
    返回: 预测的平台 (0-iOS, 1-Android)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"标准化器文件不存在: {SCALER_PATH}")

    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    feature_cols = model_package['feature_cols']
    categorical_columns = model_package['categorical_columns']

    scaler = joblib.load(SCALER_PATH)

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

    # 标准化
    X_scaled = scaler.transform(X)

    # 预测
    platform = model.predict(X_scaled)[0]

    return int(platform)


if __name__ == '__main__':
    print("=" * 60)
    print("平台选择预测模型训练 (随机森林分类)")
    print("=" * 60)

    # 加载数据
    X, y, feature_cols = load_and_prepare_data()

    # 训练模型
    model, scaler, accuracy = train_model(X, y)

    # 保存模型
    save_model(model, scaler, feature_cols)

    # 测试预测
    test_data = {
        'kwhTotal': 8.5,
        'charging_fees': 0,
        'chargeTimeHrs': 2.5,
        'weekday': 'Mon',
        'facilityType': 3,
        'managerVehicle': 0,
        'userId': 12345,
        'stationId': 1,
        'locationId': 1,
        'Mon': 1, 'Tues': 0, 'Wed': 0, 'Thurs': 0, 'Fri': 0, 'Sat': 0, 'Sun': 0
    }
    predicted_platform = predict_platform(test_data)
    platform_name = "Android" if predicted_platform == 1 else "iOS"
    print(f"\n[nvv_ds_03_1] 测试预测 - 输入: {test_data}")
    print(f"[nvv_ds_03_1] 测试预测 - 输出: {predicted_platform} ({platform_name})")

    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
