"""
充电桩数据读取与清洗模块
支持本地 CSV 读取和 HDFS 读取
用于充电时间、费用、平台预测
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
        print(f"[Nvv2thdfs] 从本地读取: {csv_path}")
        return read_data_local(csv_path)
    if hdfs_path:
        print(f"[Nvv2thdfs] 从 HDFS 读取: {hdfs_path}")
        return read_data_hdfs(hdfs_path)
    raise FileNotFoundError(f"本地文件不存在且未提供有效的 HDFS 路径")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗:
    1. 将 created, ended 转为 datetime
    2. 提取 startTime, endTime (小时)
    3. 分类变量映射：platform, weekday
    4. 处理缺失值
    """
    df = df.copy()

    # 转换时间字段
    if 'created' in df.columns:
        df['created'] = pd.to_datetime(df['created'])
    if 'ended' in df.columns:
        df['ended'] = pd.to_datetime(df['ended'])

    # 提取小时特征（如果列存在且未提取）
    if 'created' in df.columns and 'startTime' not in df.columns:
        df['startTime'] = df['created'].dt.hour
    if 'ended' in df.columns and 'endTime' not in df.columns:
        df['endTime'] = df['ended'].dt.hour

    # 分类变量映射
    if 'platform' in df.columns:
        df['platform'] = df['platform'].map({'android': 1, 'ios': 0, 'ios': 0})

    # 删除不需要的列
    cols_to_drop = ['sessionId', 'created', 'ended']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 处理缺失值
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[Nvv2thdfs] 剔除缺失值行: {before - after} 条")

    return df


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), '../..', 'data')
    csv_path = os.path.join(data_dir, 'nvv2t.csv')
    hdfs_path = '/Car/nvv2t.csv'

    df = read_data(csv_path=csv_path, hdfs_path=hdfs_path)
    df_cleaned = clean_data(df)

    pkl_path = os.path.join(data_dir, 'nvv_cleaned.pkl')
    os.makedirs(data_dir, exist_ok=True)
    df_cleaned.to_pickle(pkl_path)
    print(f"[Nvv2thdfs] 清洗后数据已保存至: {pkl_path}")
    print(f"[Nvv2thdfs] 总行数: {len(df_cleaned)}")
    print(f"[Nvv2thdfs] 列名: {list(df_cleaned.columns)}")
