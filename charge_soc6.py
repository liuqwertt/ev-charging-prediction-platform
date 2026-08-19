"""
充电速率分析模块
使用 diff() 计算 SOC 变化率（每分钟），标出速率最高峰时间点
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO


def charge_rate_soc(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每分钟 SOC 变化率
    :param df: 已清洗的 DataFrame（含 record_time、soc 列）
    :return: DataFrame，含 record_time、hour、charge_rate_soc_per_min
    """
    df_sorted = df.copy()
    df_sorted.sort_values(by='record_time', inplace=True)

    # 计算 SOC 变化率：SOC 差分 / 时间差（分钟）
    df_sorted['charge_rate_soc_per_min'] = (
        df_sorted['soc'].diff()
        / (df_sorted['record_time'].diff().dt.seconds / 60)
    )
    df_sorted.dropna(subset=['charge_rate_soc_per_min'], inplace=True)
    df_sorted['hour'] = df_sorted['record_time'].dt.hour

    return df_sorted


def hourly_avg_charge_rate(rate_df: pd.DataFrame) -> pd.Series:
    """按小时计算平均充电速率"""
    return rate_df.groupby('hour')['charge_rate_soc_per_min'].mean()


def peak_charge_time(rate_df: pd.DataFrame):
    """
    找出充电速率最高峰对应的小时和速率值
    :return: (peak_hour, peak_rate)
    """
    hourly = hourly_avg_charge_rate(rate_df)
    peak_hour = hourly.idxmax()
    peak_rate = hourly.max()
    return peak_hour, peak_rate


def _charging_rate_fig(rate_df: pd.DataFrame, peak_hour: int, peak_rate: float,
                       font_prop=None) -> str:
    """
    绘制充电速率折线图，标注最高峰，返回 base64 编码字符串
    """
    sns.set_style("whitegrid")
    hourly = hourly_avg_charge_rate(rate_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly.index, hourly.values,
            linewidth=2, color='#1890FF', marker='o', markersize=5)

    # 标注最高峰
    ax.scatter([peak_hour], [peak_rate], color='red', s=120, zorder=5, label='Peak')
    ax.annotate(
        f'Peak: Hour {peak_hour}\n{peak_rate:.4f}',
        xy=(peak_hour, peak_rate),
        xytext=(peak_hour + 1, peak_rate + 0.02),
        fontproperties=font_prop,
        fontsize=11,
        color='red',
        arrowprops=dict(arrowstyle='->', color='red')
    )

    ax.set_title('充电速率曲线（SOC/min）', fontproperties=font_prop, fontsize=18)
    ax.set_xlabel('小时 (Hour of Day)', fontproperties=font_prop, fontsize=14)
    ax.set_ylabel('平均充电速率 (SOC/min)', fontproperties=font_prop, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    from battery_data_analysis2 import load_cleaned_data
    df = load_cleaned_data()
    rate_df = charge_rate_soc(df)
    h, d, m = peak_charge_time(rate_df)
    print(f"=== 充电速率峰值: Hour={h}, Rate={d:.6f} ===")
