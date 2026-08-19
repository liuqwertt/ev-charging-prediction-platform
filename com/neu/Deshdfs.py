"""
数据读取与清洗模块
支持本地 CSV 读取和 HDFS 读取
"""
import pandas as pd
import os


def read_data_local(csv_path: str) -> pd.DataFrame:
    """读取本地 CSV 文件"""
    df = pd.read_csv(csv_path)
    return df


def read_data_hdfs(hdfs_path: str) -> pd.DataFrame:
    """读取 HDFS 上的 CSV 文件"""
    from hdfs import InsecureClient
    client = InsecureClient('http://localhost:9870', user='root')
    with client.read(hdfs_path, encoding='utf-8') as reader:
        df = pd.read_csv(reader)
    return df


def read_data(csv_path: str = None, hdfs_path: str = None) -> pd.DataFrame:
    """
    读取数据：优先本地 CSV，若不存在则尝试 HDFS
    """
    if csv_path and os.path.exists(csv_path):
        print(f"[Deshdfs] 从本地读取: {csv_path}")
        return read_data_local(csv_path)
    if hdfs_path:
        print(f"[Deshdfs] 从 HDFS 读取: {hdfs_path}")
        return read_data_hdfs(hdfs_path)
    raise FileNotFoundError(f"本地文件不存在且未提供有效的 HDFS 路径")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗:
    1. 将 record_time 转为 datetime，提取 date 和 hour 列
    2. 剔除 charge_current 为空的数据
    3. 统计充电事件（charge_current < 0）的总条数
    """
    df = df.copy()
    df['record_time'] = pd.to_datetime(df['record_time'], format='%Y%m%d%H%M%S')
    df['date'] = df['record_time'].dt.date
    df['hour'] = df['record_time'].dt.hour

    before = len(df)
    df = df.dropna(subset=['charge_current (A)'])
    after = len(df)
    print(f"[Deshdfs] 剔除 charge_current 为空的行: {before - after} 条")

    charging_mask = df['charge_current (A)'] < 0
    charging_count = charging_mask.sum()
    print(f"[Deshdfs] 充电事件（charge_current < 0）总条数: {charging_count}")

    return df


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '../..', 'data')
    csv_path = os.path.join(data_dir, 'dsv13r2.csv')
    hdfs_path = '/Car/dsv13r2.csv'

    df = read_data(csv_path=csv_path, hdfs_path=hdfs_path)
    df_cleaned = clean_data(df)

    pkl_path = os.path.join(data_dir, 'df_cleaned.pkl')
    os.makedirs(data_dir, exist_ok=True)
    df_cleaned.to_pickle(pkl_path)
    print(f"[Deshdfs] 清洗后数据已保存至: {pkl_path}")
    print(f"[Deshdfs] 总行数: {len(df_cleaned)}")
