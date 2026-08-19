"""
充电频数统计模块
统计每日/每月充电次数，充电事件定义为 charge_current < 0
"""
import pandas as pd
import os


def load_cleaned_data(pkl_path: str = None) -> pd.DataFrame:
    """加载清洗后的 pkl 数据"""
    if pkl_path is None:
        pkl_path = os.path.join(os.path.dirname(__file__), 'data', 'df_cleaned.pkl')
    return pd.read_pickle(pkl_path)


def get_daily_count(df: pd.DataFrame) -> pd.Series:
    """
    统计每日充电次数
    返回: Series，索引为日期（date），值为充电次数
    """
    charging = df[df['charge_current (A)'] < 0]
    daily_count = charging.groupby('date').size()
    daily_count.index = pd.to_datetime(daily_count.index)
    daily_count.name = 'daily_charging_count'
    return daily_count


def get_monthly_count(df: pd.DataFrame) -> pd.Series:
    """
    统计每月充电次数
    返回: Series，索引为月份（YYYY-MM），值为充电次数
    """
    charging = df[df['charge_current (A)'] < 0]
    charging = charging.copy()
    charging['month'] = charging['record_time'].dt.to_period('M')
    monthly_count = charging.groupby('month').size()
    monthly_count.index = monthly_count.index.astype(str)
    monthly_count.name = 'monthly_charging_count'
    return monthly_count


if __name__ == '__main__':
    df = load_cleaned_data()
    daily = get_daily_count(df)
    monthly = get_monthly_count(df)
    print("=== 每日充电次数 ===")
    print(daily)
    print("\n=== 每月充电次数 ===")
    print(monthly)
