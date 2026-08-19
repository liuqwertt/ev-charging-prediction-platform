"""
Flask Web应用 - 新能源充电桩数据可视化与预测平台
包含数据分析图表和机器学习预测功能
"""
import base64
import json
from io import BytesIO
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from flask import Flask, render_template, request, jsonify
from flask_caching import Cache

app = Flask(__name__)

# ==================== 缓存配置 ====================
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})

# ==================== 全局模型加载 ====================
_MODELS = {}  # 全局模型字典，存储所有加载的模型


def load_all_models():
    """Flask启动时一次性加载所有.pkl模型文件，避免每次请求读盘"""
    import os
    import joblib
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    _MODELS['soc'] = joblib.load(os.path.join(data_dir, 'soc_model.pkl'))
    _MODELS['time'] = joblib.load(os.path.join(data_dir, 'time_model.pkl'))
    _MODELS['fee'] = joblib.load(os.path.join(data_dir, 'fee_model.pkl'))
    _MODELS['platform'] = joblib.load(os.path.join(data_dir, 'platform_model.pkl'))
    _MODELS['scaler'] = joblib.load(os.path.join(data_dir, 'scaler.pkl'))
    print(f"[app] load_all_models() 完成，已加载: {list(_MODELS.keys())}")


# ── Ubuntu 兼容中文配置 ──
_FONT_CANDIDATES = [
    'WenQuanYi Micro Hei',
    'Noto Sans CJK SC',
    'Noto Sans CJK JP',
    'SimHei',
    'Microsoft YaHei',
]


def _find_chinese_font():
    """查找系统可用的中文字体，返回 FontProperties 对象"""
    fonts = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in _FONT_CANDIDATES:
        if name in fonts:
            print(f"[app] 使用字体: {name} -> {fonts[name]}")
            return fm.FontProperties(fname=fonts[name])
    fallback_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for p in fallback_paths:
        if __import__('os').path.exists(p):
            print(f"[app] 使用后备字体: {p}")
            return fm.FontProperties(fname=p)
    print("[app] 警告：未找到中文字体，使用默认字体")
    return None


_font_prop = _find_chinese_font()


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ==================== 数据分析路由 ====================

@app.route('/')
def index():
    return render_template('base.html')


@app.route('/bda1')
@cache.cached(timeout=300)
def bda1():
    from battery_data_analysis2 import load_cleaned_data, get_daily_count, get_monthly_count
    try:
        df = load_cleaned_data()
    except FileNotFoundError:
        return render_template('page1.html', plot_url='', plot_url2='',
                               error='未找到清洗后的数据，请先运行 com/neu/Deshdfs.py')

    daily = get_daily_count(df)
    monthly = get_monthly_count(df)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(daily.index, daily.values, linewidth=2, color='blue', marker='o', markersize=3)
    ax1.set_title('设备的每日充电频数', fontproperties=_font_prop, fontsize=18)
    ax1.set_xlabel('日期', fontproperties=_font_prop, fontsize=14)
    ax1.set_ylabel('充电次数', fontproperties=_font_prop, fontsize=14)
    ax1.tick_params(labelsize=10)
    ax1.grid(True, alpha=0.3)
    fig1.autofmt_xdate()
    plot_url = _fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(monthly.index, monthly.values, linewidth=2, color='red', marker='s', markersize=5)
    ax2.set_title('设备每月充电频数', fontproperties=_font_prop, fontsize=18)
    ax2.set_xlabel('月份', fontproperties=_font_prop, fontsize=14)
    ax2.set_ylabel('充电次数', fontproperties=_font_prop, fontsize=14)
    ax2.tick_params(labelsize=10)
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plot_url2 = _fig_to_base64(fig2)

    return render_template('page1.html', plot_url=plot_url, plot_url2=plot_url2, error=None)


@app.route('/bda2')
@cache.cached(timeout=300)
def bda2():
    from battery_soc3 import hourly_socf, _hourly_soc_line_fig
    from battery_soc4 import hourly_soc_pivotf, _heatmap_fig
    from battery_data_analysis2 import load_cleaned_data
    try:
        df = load_cleaned_data()
    except FileNotFoundError:
        return render_template('page2.html', plot_url='', plot_url2='',
                               error='未找到清洗后的数据，请先运行 com/neu/Deshdfs.py')

    hourly_soc = hourly_socf(df)
    plot_url = _hourly_soc_line_fig(hourly_soc, font_prop=_font_prop)

    pivot = hourly_soc_pivotf(df)
    plot_url2 = _heatmap_fig(pivot, font_prop=_font_prop)

    return render_template('page2.html', plot_url=plot_url, plot_url2=plot_url2, error=None)


@app.route('/bda3')
@cache.cached(timeout=300)
def bda3():
    from charge_current5 import charge_time_by_hour_day_month, _charging_time_fig
    from battery_data_analysis2 import load_cleaned_data
    try:
        df = load_cleaned_data()
    except FileNotFoundError:
        return render_template('page3.html', plot_url='',
                               error='未找到清洗后的数据，请先运行 com/neu/Deshdfs.py')

    hourly, daily, monthly = charge_time_by_hour_day_month(df)
    plot_url = _charging_time_fig(hourly, daily, monthly, font_prop=_font_prop)

    return render_template('page3.html', plot_url=plot_url, error=None)


@app.route('/bda4')
@cache.cached(timeout=300)
def bda4():
    from charge_soc6 import charge_rate_soc, _charging_rate_fig, peak_charge_time
    from battery_data_analysis2 import load_cleaned_data
    try:
        df = load_cleaned_data()
    except FileNotFoundError:
        return render_template('page4.html', plot_url='', peak_info='',
                               error='未找到清洗后的数据，请先运行 com/neu/Deshdfs.py')

    rate_df = charge_rate_soc(df)
    peak_hour, peak_rate = peak_charge_time(rate_df)
    plot_url = _charging_rate_fig(rate_df, peak_hour, peak_rate, font_prop=_font_prop)
    peak_info = f'充电速率峰值出现在第 {peak_hour} 时，平均充电速率为 {peak_rate:.6f} SOC/min'

    return render_template('page4.html', plot_url=plot_url, peak_info=peak_info, error=None)


@app.route('/bda5')
@cache.cached(timeout=300)
def bda5():
    from battery_failure5 import build_failure_model, _classification_bar_fig
    from battery_data_analysis2 import load_cleaned_data
    try:
        df = load_cleaned_data()
    except FileNotFoundError:
        return render_template('page5.html', plot_url='',
                              error='未找到清洗后的数据，请先运行 com/neu/Deshdfs.py')

    X_test, y_test, y_pred = build_failure_model(df)
    plot_url = _classification_bar_fig(y_test, y_pred, font_prop=_font_prop)
    return render_template('page5.html', plot_url=plot_url, error=None)


# ==================== 预测功能路由 ====================

@app.route('/predict_soc')
def page_predict_soc():
    """SOC预测页面"""
    return render_template('page4_soc.html')


@app.route('/predict_soc_api', methods=['POST'])
def api_predict_soc():
    """SOC预测API"""
    try:
        from ds_battery72_1 import predict_soc

        data = request.get_json()
        print(f"[api] SOC预测请求: {data}")

        # 构建特征数据
        record_time = data.get('record_time')
        pack_voltage = float(data.get('pack_voltage (V)', 0))
        charge_current = float(data.get('charge_current (A)', 0))
        max_cell_voltage = float(data.get('max_cell_voltage (V)', 0))
        min_cell_voltage = float(data.get('min_cell_voltage (V)', 0))
        max_temperature = float(data.get('max_temperature (℃)', 0))
        min_temperature = float(data.get('min_temperature (℃)', 0))
        available_energy = float(data.get('available_energy (kw)', 0))
        available_capacity = float(data.get('available_capacity (Ah)', 0))

        # 转换时间戳为纳秒（与模型训练时一致）
        if isinstance(record_time, str):
            for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
                try:
                    dt = datetime.strptime(record_time, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"无法解析时间格式: {record_time}")
        elif isinstance(record_time, (int, float)):
            dt = datetime.fromtimestamp(record_time / 1000)
        else:
            dt = datetime.now()

        # 转纳秒（与训练时 ds_battery72_1.py 第 64-66 行一致）
        record_time_ns = int(dt.timestamp() * 1e9)

        data_point = {
            'record_time_ns': record_time_ns,
            'pack_voltage (V)': pack_voltage,
            'charge_current (A)': charge_current,
            'max_cell_voltage (V)': max_cell_voltage,
            'min_cell_voltage (V)': min_cell_voltage,
            'max_temperature (℃)': max_temperature,
            'min_temperature (℃)': min_temperature,
            'available_energy (kw)': available_energy,
            'available_capacity (Ah)': available_capacity
        }

        soc = predict_soc(data_point)
        return jsonify({'soc': round(soc, 2)})

    except Exception as e:
        print(f"[api] SOC预测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_time')
def page_predict_time():
    """充电时间预测页面"""
    return render_template('page4_time.html')


@app.route('/predict_time_api', methods=['POST'])
def api_predict_time():
    """充电时间预测API"""
    try:
        from nvv_ds_01_1 import predict_charge_time

        data = request.get_json()
        print(f"[api] 充电时间预测请求: {data}")

        # 构建特征数据
        data_point = {
            'kwhTotal': float(data.get('kwhTotal', 0)),
            'charging_fees': float(data.get('charging_fees', 0)),
            'startTime': int(data.get('startTime', 0)),
            'endTime': int(data.get('endTime', 0)),
            'managerVehicle': int(data.get('managerVehicle', 0)),
            'Mon': int(data.get('Mon', 0)),
            'Tues': int(data.get('Tues', 0)),
            'Wed': int(data.get('Wed', 0)),
            'Thurs': int(data.get('Thurs', 0)),
            'Fri': int(data.get('Fri', 0)),
            'Sat': int(data.get('Sat', 0)),
            'Sun': int(data.get('Sun', 0)),
            'weekday': data.get('weekday', 'Mon'),
            'platform': data.get('platform', 'android'),
            'facilityType': int(data.get('facilityType', 0))
        }

        charge_time = predict_charge_time(data_point)
        return jsonify({'charge_time': round(charge_time, 2)})

    except Exception as e:
        print(f"[api] 充电时间预测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_fee')
def page_predict_fee():
    """充电费用预测页面"""
    return render_template('page4_fee.html')


@app.route('/predict_fee_api', methods=['POST'])
def api_predict_fee():
    """充电费用预测API"""
    try:
        from nvv_ds_02_1 import predict_charge_fee

        data = request.get_json()
        print(f"[api] 充电费用预测请求: {data}")

        # 构建特征数据
        data_point = {
            'kwhTotal': float(data.get('kwhTotal', 0)),
            'chargeTimeHrs': float(data.get('chargeTimeHrs', 0)),
            'startTime': int(data.get('startTime', 0)),
            'endTime': int(data.get('endTime', 0)),
            'managerVehicle': int(data.get('managerVehicle', 0)),
            'Mon': int(data.get('Mon', 0)),
            'Tues': int(data.get('Tues', 0)),
            'Wed': int(data.get('Wed', 0)),
            'Thurs': int(data.get('Thurs', 0)),
            'Fri': int(data.get('Fri', 0)),
            'Sat': int(data.get('Sat', 0)),
            'Sun': int(data.get('Sun', 0)),
            'weekday': data.get('weekday', 'Mon'),
            'platform': data.get('platform', 'android'),
            'facilityType': int(data.get('facilityType', 0))
        }

        fee = predict_charge_fee(data_point)
        return jsonify({'fee': round(fee, 2)})

    except Exception as e:
        print(f"[api] 充电费用预测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_platform')
def page_predict_platform():
    """平台选择预测页面"""
    return render_template('page4_platform.html')


@app.route('/predict_platform_api', methods=['POST'])
def api_predict_platform():
    """平台选择预测API"""
    try:
        from nvv_ds_03_1 import predict_platform

        data = request.get_json()
        print(f"[api] 平台预测请求: {data}")

        # 构建特征数据
        data_point = {
            'kwhTotal': float(data.get('kwhTotal', 0)),
            'charging_fees': float(data.get('charging_fees', 0)),
            'chargeTimeHrs': float(data.get('chargeTimeHrs', 0)),
            'userId': int(data.get('userId', 0)),
            'stationId': int(data.get('stationId', 0)),
            'locationId': int(data.get('locationId', 0)),
            'managerVehicle': int(data.get('managerVehicle', 0)),
            'Mon': int(data.get('Mon', 0)),
            'Tues': int(data.get('Tues', 0)),
            'Wed': int(data.get('Wed', 0)),
            'Thurs': int(data.get('Thurs', 0)),
            'Fri': int(data.get('Fri', 0)),
            'Sat': int(data.get('Sat', 0)),
            'Sun': int(data.get('Sun', 0)),
            'weekday': data.get('weekday', 'Mon'),
            'facilityType': int(data.get('facilityType', 0))
        }

        platform = predict_platform(data_point)
        platform_name = "Android" if platform == 1 else "iOS"
        return jsonify({'platform': platform, 'platform_name': platform_name})

    except Exception as e:
        print(f"[api] 平台预测错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== 0630 新路由：统一预测接口 ====================

@app.route('/predict_page4', methods=['POST'])
def api_predict_page4_fee():
    """充电费用预测（LightGBM）- POST"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': '请求体必须是有效JSON'}), 400

    try:
        from nvv_ds_02_1 import predict_charge_fee
        data_point = {
            'kwhTotal': float(data['kwhTotal']),
            'chargeTimeHrs': float(data['chargeTimeHrs']),
            'startTime': int(data['startTime']),
            'endTime': int(data['endTime']),
            'managerVehicle': int(data['managerVehicle']),
            'Mon': int(data.get('Mon', 0)),
            'Tues': int(data.get('Tues', 0)),
            'Wed': int(data.get('Wed', 0)),
            'Thurs': int(data.get('Thurs', 0)),
            'Fri': int(data.get('Fri', 0)),
            'Sat': int(data.get('Sat', 0)),
            'Sun': int(data.get('Sun', 0)),
            'weekday': data.get('weekday', 'Mon'),
            'platform': data.get('platform', 'android'),
            'facilityType': int(data.get('facilityType', 0))
        }
        fee = predict_charge_fee(data_point)
        return jsonify({'fee': round(fee, 2)})
    except KeyError as e:
        return jsonify({'error': f'缺少必要参数: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_page4ds1', methods=['POST'])
def api_predict_page4ds1_soc():
    """SOC预测（随机森林回归）- POST"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': '请求体必须是有效JSON'}), 400

    try:
        from ds_battery72_1 import predict_soc
        record_time = data.get('record_time')
        if isinstance(record_time, str):
            for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
                try:
                    dt = datetime.strptime(record_time, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"无法解析时间格式: {record_time}")
        elif isinstance(record_time, (int, float)):
            dt = datetime.fromtimestamp(record_time / 1000)
        else:
            dt = datetime.now()
        record_time_ns = int(dt.timestamp() * 1e9)

        data_point = {
            'record_time_ns': record_time_ns,
            'pack_voltage (V)': float(data['pack_voltage (V)']),
            'charge_current (A)': float(data['charge_current (A)']),
            'max_cell_voltage (V)': float(data['max_cell_voltage (V)']),
            'min_cell_voltage (V)': float(data['min_cell_voltage (V)']),
            'max_temperature (℃)': float(data['max_temperature (℃)']),
            'min_temperature (℃)': float(data['min_temperature (℃)']),
            'available_energy (kw)': float(data['available_energy (kw)']),
            'available_capacity (Ah)': float(data['available_capacity (Ah)'])
        }
        soc = predict_soc(data_point)
        return jsonify({'soc': round(soc, 2)})
    except KeyError as e:
        return jsonify({'error': f'缺少必要参数: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_page4ds2', methods=['POST'])
def api_predict_page4ds2_time():
    """充电时间预测（线性回归）- POST"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': '请求体必须是有效JSON'}), 400

    try:
        from nvv_ds_01_1 import predict_charge_time
        data_point = {
            'kwhTotal': float(data['kwhTotal']),
            'charging_fees': float(data['charging_fees']),
            'startTime': int(data['startTime']),
            'endTime': int(data['endTime']),
            'managerVehicle': int(data['managerVehicle']),
            'Mon': int(data.get('Mon', 0)),
            'Tues': int(data.get('Tues', 0)),
            'Wed': int(data.get('Wed', 0)),
            'Thurs': int(data.get('Thurs', 0)),
            'Fri': int(data.get('Fri', 0)),
            'Sat': int(data.get('Sat', 0)),
            'Sun': int(data.get('Sun', 0)),
            'weekday': data.get('weekday', 'Mon'),
            'platform': data.get('platform', 'android'),
            'facilityType': int(data.get('facilityType', 0))
        }
        charge_time = predict_charge_time(data_point)
        return jsonify({'charge_time': round(charge_time, 2)})
    except KeyError as e:
        return jsonify({'error': f'缺少必要参数: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_page4ds3', methods=['POST'])
def api_predict_page4ds3_platform():
    """平台分类（随机森林分类）- POST"""
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': '请求体必须是有效JSON'}), 400

    try:
        from nvv_ds_03_1 import predict_platform
        data_point = {
            'kwhTotal': float(data['kwhTotal']),
            'charging_fees': float(data['charging_fees']),
            'chargeTimeHrs': float(data['chargeTimeHrs']),
            'userId': int(data.get('userId', 0)),
            'stationId': int(data.get('stationId', 0)),
            'locationId': int(data.get('locationId', 0)),
            'managerVehicle': int(data.get('managerVehicle', 0)),
            'Mon': int(data.get('Mon', 0)),
            'Tues': int(data.get('Tues', 0)),
            'Wed': int(data.get('Wed', 0)),
            'Thurs': int(data.get('Thurs', 0)),
            'Fri': int(data.get('Fri', 0)),
            'Sat': int(data.get('Sat', 0)),
            'Sun': int(data.get('Sun', 0)),
            'weekday': data.get('weekday', 'Mon'),
            'facilityType': int(data.get('facilityType', 0))
        }
        platform = predict_platform(data_point)
        platform_name = "Android" if platform == 1 else "iOS"
        return jsonify({'platform': platform, 'platform_name': platform_name})
    except KeyError as e:
        return jsonify({'error': f'缺少必要参数: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== 废弃路由（兼容） ====================

@app.route('/charging_frequency')
def charging_frequency():
    return render_template('base.html')


@app.route('/soc_trajectory')
def soc_trajectory():
    return render_template('base.html')


@app.route('/charging_time')
def charging_time():
    return render_template('base.html')


@app.route('/charging_speed')
def charging_speed():
    return render_template('base.html')


if __name__ == '__main__':
    load_all_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
