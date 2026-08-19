"""
SOC预测模型训练脚本
使用随机森林回归 + GridSearchCV超参数优化
数据集：dsv13r2.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# 获取当前脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_PATH = os.path.join(DATA_DIR, 'soc_model.pkl')


def load_and_prepare_data():
    """加载并准备数据"""
    csv_path = os.path.join(DATA_DIR, 'dsv13r2.csv')
    hdfs_path = '/Car/dsv13r2.csv'

    # 尝试本地读取
    if os.path.exists(csv_path):
        print(f"[ds_battery72_1] 从本地读取: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[ds_battery72_1] 从 HDFS 读取: {hdfs_path}")
        from hdfs import InsecureClient
        client = InsecureClient('http://localhost:9870', user='root')
        with client.read(hdfs_path, encoding='utf-8') as reader:
            df = pd.read_csv(reader)

    print(f"[ds_battery72_1] 原始数据行数: {len(df)}")

    # 数据清洗：剔除SOC为空或不在[0,100]的记录
    before = len(df)
    df = df.dropna(subset=['soc'])
    df = df[(df['soc'] >= 0) & (df['soc'] <= 100)]
    after = len(df)
    print(f"[ds_battery72_1] 清洗后数据行数: {after} (剔除 {before - after} 条异常记录)")

    # 特征列
    feature_cols = [
        'pack_voltage (V)',
        'charge_current (A)',
        'max_cell_voltage (V)',
        'min_cell_voltage (V)',
        'max_temperature (℃)',
        'min_temperature (℃)',
        'available_energy (kw)',
        'available_capacity (Ah)'
    ]

    # 确保所有特征列存在
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"缺少特征列: {col}")

    # 将record_time转为纳秒数值
    if 'record_time' in df.columns:
        df['record_time'] = pd.to_datetime(df['record_time'], format='%Y%m%d%H%M%S')
        df['record_time_ns'] = df['record_time'].astype('int64')
        feature_cols = ['record_time_ns'] + feature_cols

    X = df[feature_cols]
    y = df['soc']

    print(f"[ds_battery72_1] 特征列: {feature_cols}")
    print(f"[ds_battery72_1] 特征维度: {X.shape}")

    return X, y, feature_cols


def train_model(X, y):
    """训练随机森林回归模型"""
    # 划分训练集和验证集
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"[ds_battery72_1] 训练集: {len(X_train)}, 验证集: {len(X_valid)}")

    # 超参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10]
    }

    print("[ds_battery72_1] 开始 GridSearchCV 超参数优化...")
    print(f"[ds_battery72_1] 参数网格: {param_grid}")

    # 初始化随机森林模型
    model = RandomForestRegressor(random_state=0, n_jobs=-1)

    # 网格搜索
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 最佳参数
    print(f"[ds_battery72_1] 最佳参数: {grid_search.best_params_}")
    print(f"[ds_battery72_1] 最佳交叉验证 MAE: {-grid_search.best_score_:.4f}")

    # 使用最佳参数的模型
    best_model = grid_search.best_estimator_

    # 验证集评估
    y_pred = best_model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    mse = mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    print(f"[ds_battery72_1] 验证集评估:")
    print(f"  MAE: {mae:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  R2: {r2:.4f}")

    return best_model, grid_search.best_params_


def save_model(model, feature_cols, best_params):
    """保存模型"""
    os.makedirs(DATA_DIR, exist_ok=True)
    model_package = {
        'model': model,
        'feature_cols': feature_cols,
        'best_params': best_params
    }
    joblib.dump(model_package, MODEL_PATH)
    print(f"[ds_battery72_1] 模型已保存至: {MODEL_PATH}")


def predict_soc(data_point):
    """
    SOC推理函数
    参数: data_point - dict 包含特征数据
    返回: 预测的SOC值
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    model_package = joblib.load(MODEL_PATH)
    model = model_package['model']
    feature_cols = model_package['feature_cols']

    # 转换为DataFrame
    df = pd.DataFrame([data_point])

    # 确保特征顺序一致（此时 record_time_ns 已由调用方传入，无需再转换）
    X = df[feature_cols]

    # 预测
    soc = model.predict(X)[0]
    return float(soc)


if __name__ == '__main__':
    print("=" * 60)
    print("SOC预测模型训练")
    print("=" * 60)

    # 加载数据
    X, y, feature_cols = load_and_prepare_data()

    # 训练模型
    model, best_params = train_model(X, y)

    # 保存模型
    save_model(model, feature_cols, best_params)

    print("=" * 60)
    print("训练完成!")
    print("=" * 60)
