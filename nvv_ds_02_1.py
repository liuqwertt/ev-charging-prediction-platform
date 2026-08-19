"""
充电费用预测模型训练脚本
使用LightGBM回归，对比默认参数与调参后的MSE差异
数据集：nvv2t.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import joblib


# 获取当前脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'fee_model.pkl')

# 分类变量
CATEGORICAL_COLUMNS = ['weekday', 'platform', 'facilityType', 'managerVehicle']


def load_and_prepare_data():
    """加载并准备数据"""
    csv_path = os.path.join(DATA_DIR, 'nvv2t.csv')
    hdfs_path = '/Car/nvv2t.csv'

    # 尝试本地读取
    if os.path.exists(csv_path):
        print(f"[nvv_ds_02_1] 从本地读取: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[nvv_ds_02_1] 从 HDFS 读取: {hdfs_path}")
        from hdfs import InsecureClient
        client = InsecureClient('http://localhost:9870', user='root')
        with client.read(hdfs_path, encoding='utf-8') as reader:
            df = pd.read_csv(reader)

    print(f"[nvv_ds_02_1] 原始数据行数: {len(df)}")

    # 数据清洗
    before = len(df)
    df = df.dropna()
    df = df.drop(columns=['sessionId', 'created', 'ended', 'userId', 'stationId', 'locationId'], errors='ignore')
    after = len(df)
    print(f"[nvv_ds_02_1] 清洗后数据行数: {after} (剔除 {before - after} 条)")

    # 独热编码
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # 特征列（除了目标列）
    target_col = 'charging_fees'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print(f"[nvv_ds_02_1] 特征维度: {X.shape}")
    print(f"[nvv_ds_02_1] 特征列: {feature_cols}")

    return X, y, feature_cols


def train_default_model(X_train, X_test, y_train, y_test):
    """训练默认参数的LightGBM模型"""
    print("\n" + "=" * 50)
    print("训练默认参数 LightGBM 模型")
    print("=" * 50)

    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[nvv_ds_02_1] 默认参数 MSE: {mse:.6f}")
    print(f"[nvv_ds_02_1] 默认参数 RMSE: {np.sqrt(mse):.6f}")

    return model, mse


def train_tuned_model(X_train, X_test, y_train, y_test):
    """训练调参后的LightGBM模型"""
    print("\n" + "=" * 50)
    print("训练调参后 LightGBM 模型")
    print("=" * 50)

    # 调参参数
    params = {
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': 10,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }

    print(f"[nvv_ds_02_1] 调参参数: {params}")

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"[nvv_ds_02_1] 调参后 MSE: {mse:.6f}")
    print(f"[nvv_ds_02_1] 调参后 RMSE: {np.sqrt(mse):.6f}")

    return model, mse


def compare_models(default_mse, tuned_mse):
    """对比默认参数与调参后的MSE差异"""
    print("\n" + "=" * 50)
    print("模型对比")
    print("=" * 50)
    improvement = (default_mse - tuned_mse) / default_mse * 100
    print(f"[nvv_ds_02_1] 默认MSE: {default_mse:.6f}")
    print(f"[nvv_ds_02_1] 调参MSE: {tuned_mse:.6f}")
    print(f"[nvv_ds_02_1] MSE改进: {improvement:.2f}%")

    if tuned_mse < default_mse:
        print("[nvv_ds_02_1] 调参后模型性能更优!")
    else:
        print("[nvv_ds_02_1] 默认参数模型性能更优!")


def save_model(model, feature_cols, best_mse):
    """保存模型"""
    os.makedirs(DATA_DIR, exist_ok=True)
    model_package = {
        'model': model,
        'feature_cols': feature_cols,
        'categorical_columns': CATEGORICAL_COLUMNS,
        'best_mse': best_mse
    }
    joblib.dump(model_package, MODEL_PATH)
    print(f"[nvv_ds_02_1] 模型已保存至: {MODEL_PATH}")


def predict_charge_fee(data_point):
    """
    充电费用推理函数
    参数: data_point - dict 包含特征数据
    返回: 预测的充电费用（元）
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
    fee = model.predict(X)[0]
    return float(max(0, fee))  # 确保非负


if __name__ == '__main__':
    print("=" * 60)
    print("充电费用预测模型训练 (LightGBM)")
    print("=" * 60)

    # 加载数据
    X, y, feature_cols = load_and_prepare_data()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[nvv_ds_02_1] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 训练默认参数模型
    default_model, default_mse = train_default_model(X_train, X_test, y_train, y_test)

    # 训练调参后模型
    tuned_model, tuned_mse = train_tuned_model(X_train, X_test, y_train, y_test)

    # 对比模型
    compare_models(default_mse, tuned_mse)

    # 保存调参后模型（性能更好）
    save_model(tuned_model, feature_cols, tuned_mse)

    # 测试预测
    test_data = {
        'kwhTotal': 18.5,
        'chargeTimeHrs': 2.5,
        'startTime': 15,
        'endTime': 17,
        'managerVehicle': 1,
        'weekday': 'Tue',
        'platform': 'android',
        'facilityType': 3
    }
    predicted_fee = predict_charge_fee(test_data)
    print(f"\n[nvv_ds_02_1] 测试预测 - 输入: {test_data}")
    print(f"[nvv_ds_02_1] 测试预测 - 输出: {predicted_fee:.2f} 元")

    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
