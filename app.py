# ===== Fix for Streamlit Cloud =====
import matplotlib
matplotlib.use('Agg')
# ==================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import io
warnings.filterwarnings('ignore')

# ===== 中文字体配置 =====
from matplotlib import font_manager

chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'STHeiti', 'Heiti SC']
font_set = False
for font_name in chinese_fonts:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        font_set = True
        break
    except:
        continue

if not font_set:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# ===== 黑色主题图表配置 =====
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#0d1117'
# ============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import plotly.graph_objects as go
import shap

# Page config
st.set_page_config(
    page_title="污水处理智能分析平台",
    page_icon="💧",
    layout="wide"
)

# ============ 自定义CSS ============
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #58a6ff;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
        letter-spacing: 2px;
    }
    .sub-header {
        font-size: 1rem;
        color: #8b949e;
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 1px solid #30363d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #161b22;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
        border-left: 4px solid #58a6ff;
        text-align: center;
        margin: 0 4px;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f0f6fc;
        margin: 4px 0;
    }
    .metric-card .sub {
        font-size: 0.7rem;
        color: #8b949e;
    }
    .result-card {
        background: #161b22;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
        text-align: center;
        border-top: 4px solid #58a6ff;
        height: 100%;
    }
    .result-card .label {
        font-size: 0.8rem;
        color: #8b949e;
        font-weight: 500;
    }
    .result-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f6fc;
        margin: 6px 0;
    }
    .stButton button {
        background: #238636;
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    .stButton button:hover {
        background: #2ea043;
        box-shadow: 0 4px 16px rgba(35, 134, 54, 0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #161b22;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background: #238636;
        color: white;
    }
    .status-normal { color: #3fb950; font-weight: 700; }
    .status-warning { color: #d29922; font-weight: 700; }
    .status-danger { color: #f85149; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💧 污水处理智能分析平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于进水参数的污泥指标预测与SRT优化系统</div>', unsafe_allow_html=True)

# ============ 初始化session_state ============
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'default'
if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'pred_values' not in st.session_state:
    st.session_state.pred_values = {}
if 'input_values' not in st.session_state:
    st.session_state.input_values = {}

# ============ 加载默认数据 ============
@st.cache_data
def load_default_data():
    try:
        df = pd.read_excel('随机森林归一化.xlsx', sheet_name='Sheet1')
        return df
    except:
        try:
            df = pd.read_excel('data/随机森林归一化.xlsx', sheet_name='Sheet1')
            return df
        except:
            return None

def load_data():
    if st.session_state.data_source == 'uploaded' and st.session_state.df_loaded is not None:
        return st.session_state.df_loaded
    else:
        return load_default_data()

# ============ 数据准备 ============
X_columns = ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)', 'SS(mg/l)', 
             'NH3-N(mg/l)', 'TP(mg/l)', 'TN(mg/l)', 'Tin℃']
y_columns = ['F/M(%)', 'SVI', 'SRT']

x_names_cn = {
    'Qoutm3/d': '进水流量',
    'BOD5 (mg/l)': '进水BOD5',
    'CODcr(mg/l)': '进水CODcr',
    'SS(mg/l)': '进水SS',
    'NH3-N(mg/l)': '进水NH3-N',
    'TP(mg/l)': '进水TP',
    'TN(mg/l)': '进水TN',
    'Tin℃': '进水水温'
}
y_names_cn = {
    'F/M(%)': '有机质占比',
    'SVI': 'SVI (污泥体积指数)',
    'SRT': 'SRT (污泥龄)'
}
x_names_en = {
    'Qoutm3/d': 'Flow Rate',
    'BOD5 (mg/l)': 'BOD5',
    'CODcr(mg/l)': 'CODcr',
    'SS(mg/l)': 'SS',
    'NH3-N(mg/l)': 'NH3-N',
    'TP(mg/l)': 'TP',
    'TN(mg/l)': 'TN',
    'Tin℃': 'Temp'
}
y_names_en = {
    'F/M(%)': 'F/M Ratio',
    'SVI': 'SVI',
    'SRT': 'SRT'
}

# ============ 加载数据 ============
df = load_data()
if df is None:
    st.error("❌ 找不到数据文件！请上传Excel文件或确保默认数据存在。")
    st.stop()

available_X = [col for col in X_columns if col in df.columns]
available_y = [col for col in y_columns if col in df.columns]

X_data = df[available_X].copy()
y_data = df[available_y].copy()

date_col = None
if '日期' in df.columns:
    date_col = '日期'
    df['日期'] = pd.to_datetime(df['日期'])

combined = pd.concat([X_data, y_data], axis=1).dropna()
X_data = combined[available_X]
y_data = combined[available_y]

if date_col:
    date_data = df.loc[combined.index, date_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

# ============ 训练模型 ============
@st.cache_resource
def train_models(X_data, y_data):
    X_scaled = scaler.fit_transform(X_data)
    models = {}
    results = {}
    
    for y_col in y_data.columns:
        y_target = y_data[y_col].values
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_target, test_size=0.2, random_state=42
        )
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        lasso = Lasso(alpha=0.1, random_state=42, max_iter=1000)
        lasso.fit(X_train, y_train)
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=50, learning_rate=0.1, max_depth=4,
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        
        models[y_col] = {
            'lr': lr, 'lasso': lasso, 'rf': rf, 'xgb': xgb_model,
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
        
        results[y_col] = {}
        for name, model in [('lr', lr), ('lasso', lasso), ('rf', rf), ('xgb', xgb_model)]:
            y_pred = model.predict(X_test)
            results[y_col][name] = {
                'r2': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred)
            }
    
    return models, results

models, results = train_models(X_data, y_data)

def predict_value(input_dict, model):
    input_array = np.array([input_dict[col] for col in available_X]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    return model.predict(input_scaled)[0]

# ============ 侧边栏 ============
with st.sidebar:
    st.markdown("## 📊 进水参数输入")
    st.markdown("---")
    
    input_values = {}
    for idx, col in enumerate(available_X):
        min_val = float(X_data[col].min())
        max_val = float(X_data[col].max())
        default_val = float(X_data[col].mean())
        input_values[col] = st.number_input(
            f"{x_names_cn.get(col, col)}",
            min_value=min_val, max_value=max_val,
            value=default_val,
            step=(max_val - min_val) / 100,
            format="%.2f"
        )
    
    st.markdown("---")
    if st.button("🚀 开始预测", use_container_width=True):
        st.session_state.predicted = True
        st.session_state.pred_values = {}
        st.session_state.input_values = input_values.copy()
        for y_col in available_y:
            model = models[y_col]['xgb']
            pred_val = predict_value(input_values, model)
            st.session_state.pred_values[y_col] = pred_val
        st.rerun()
    
    # ===== Excel导入功能 =====
    st.markdown("---")
    st.markdown("## 📁 导入数据")
    st.markdown("上传Excel文件替换默认数据")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件",
        type=['xlsx', 'xls'],
        help="上传包含'日期'、'Qoutm3/d'、'BOD5 (mg/l)'等列的Excel文件"
    )
    
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_excel(uploaded_file, sheet_name=0)
            required_cols = ['日期'] + X_columns
            missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
            if missing_cols:
                st.warning(f"⚠️ 缺少列: {missing_cols[:3]}... 请检查文件格式")
            else:
                st.session_state.df_loaded = uploaded_df
                st.session_state.data_source = 'uploaded'
                st.success(f"✅ 成功导入 {len(uploaded_df)} 行数据！")
                st.info("🔄 请点击'应用新数据'按钮更新模型")
                if st.button("🔄 应用新数据"):
                    st.rerun()
        except Exception as e:
            st.error(f"❌ 文件读取失败: {str(e)}")
    
    if st.session_state.data_source == 'uploaded':
        st.info("📌 当前使用: 上传的数据")
    else:
        st.info("📌 当前使用: 默认数据")

# ============ 自定义正常范围 ============
FM_MIN, FM_MAX = 20.0, 40.0
SVI_MIN, SVI_MAX = 50.0, 150.0
SRT_MIN, SRT_MAX = 5.0, 15.0

# ============ 主区域 ============
if 'predicted' in st.session_state and st.session_state.predicted:
    pred_fm = st.session_state.pred_values.get('F/M(%)', 0)
    pred_svi = st.session_state.pred_values.get('SVI', 0)
    pred_srt = st.session_state.pred_values.get('SRT', 0)
    input_vals = st.session_state.input_values
    
    def get_status(val, min_val, max_val):
        if val < min_val:
            return "偏低", "status-warning"
        elif val > max_val:
            return "偏高", "status-danger"
        else:
            return "正常", "status-normal"
    
    fm_status, fm_class = get_status(pred_fm, FM_MIN, FM_MAX)
    svi_status, svi_class = get_status(pred_svi, SVI_MIN, SVI_MAX)
    srt_status, srt_class = get_status(pred_srt, SRT_MIN, SRT_MAX)
    
    raw_opt_srt = (pred_fm / 15.0) * 12.0
    opt_srt = max(SRT_MIN, min(SRT_MAX, raw_opt_srt))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">🧪 预测有机质占比</div>
            <div class="value">{pred_fm:.2f}%</div>
            <div class="sub"><span class="{fm_class}">{fm_status}</span></div>
            <div style="font-size:0.65rem;color:#8b949e;">正常: {FM_MIN}% ~ {FM_MAX}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#f0883e;">
            <div class="label">📊 预测SVI</div>
            <div class="value">{pred_svi:.2f}</div>
            <div class="sub"><span class="{svi_class}">{svi_status}</span></div>
            <div style="font-size:0.65rem;color:#8b949e;">正常: {SVI_MIN} ~ {SVI_MAX}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#3fb950;">
            <div class="label">⏳ 模型预测SRT</div>
            <div class="value">{pred_srt:.2f}<span style="font-size:0.9rem;color:#8b949e;"> 天</span></div>
            <div class="sub"><span class="{srt_class}">{srt_status}</span></div>
            <div style="font-size:0.65rem;color:#8b949e;">正常: {SRT_MIN} ~ {SRT_MAX} 天</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color:#d29922;">
            <div class="label">🌟 推荐最优污泥龄</div>
            <div class="value" style="color:#d29922;">{opt_srt:.2f}<span style="font-size:0.9rem;color:#8b949e;"> 天</span></div>
            <div class="sub">基于F/M优化 (5~15天)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== 数据导出 =====
    st.markdown("---")
    st.markdown("### 💾 导出预测结果")
    
    export_data = {
        '输入参数': [],
        '数值': []
    }
    for col, val in input_vals.items():
        export_data['输入参数'].append(x_names_cn.get(col, col))
        export_data['数值'].append(val)
    
    export_data['输入参数'].extend(['预测有机质占比(F/M)', '预测SVI', '预测SRT', '推荐最优污泥龄'])
    export_data['数值'].extend([f"{pred_fm:.2f}%", f"{pred_svi:.2f}", f"{pred_srt:.2f}天", f"{opt_srt:.2f}天"])
    
    export_data['输入参数'].extend(['有机质占比状态', 'SVI状态', 'SRT状态'])
    export_data['数值'].extend([fm_status, svi_status, srt_status])
    
    export_df = pd.DataFrame(export_data)
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='预测结果', index=False)
        df_preview = df.head(20)
        df_preview.to_excel(writer, sheet_name='原始数据预览', index=False)
    
    st.download_button(
        label="📥 下载预测结果 (Excel)",
        data=output.getvalue(),
        file_name=f"预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    
else:
    col1, col2, col3, col4 = st.columns(4)
    for col in [col1, col2, col3, col4]:
        with col:
            st.markdown("""
            <div class="metric-card" style="opacity:0.5;">
                <div class="label">等待预测...</div>
                <div class="value" style="font-size:1rem;color:#8b949e;">点击"开始预测"</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

# ============ Tabs ============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 预测分析", "📈 时间序列", "📊 特征重要性",
    "📉 模型评价", "🔍 SHAP解释", "📋 数据总览"
])

# ===== Tab 1: 预测分析 =====
with tab1:
    st.markdown("### 🎯 预测结果详情")
    if 'predicted' in st.session_state and st.session_state.predicted:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="result-card" style="border-top-color:#58a6ff;">
                <div class="label">🧪 有机质占比 (F/M)</div>
                <div class="value">{pred_fm:.2f}%</div>
                <div><span class="{fm_class}">{fm_status}</span></div>
                <div style="font-size:0.75rem;color:#8b949e;margin-top:8px;">
                    正常范围: {FM_MIN}% ~ {FM_MAX}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card" style="border-top-color:#f0883e;">
                <div class="label">📊 SVI (污泥体积指数)</div>
                <div class="value">{pred_svi:.2f}</div>
                <div><span class="{svi_class}">{svi_status}</span></div>
                <div style="font-size:0.75rem;color:#8b949e;margin-top:8px;">
                    正常范围: {SVI_MIN} ~ {SVI_MAX}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-card" style="border-top-color:#3fb950;">
                <div class="label">⏳ 污泥龄 (SRT)</div>
                <div class="value">{pred_srt:.2f} 天</div>
                <div><span class="{srt_class}">{srt_status}</span></div>
                <div style="font-size:0.75rem;color:#8b949e;margin-top:8px;">
                    正常范围: {SRT_MIN} ~ {SRT_MAX} 天
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📍 SRT vs F/M 关系图")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_data['SRT'], y=y_data['F/M(%)'],
            mode='markers', name='历史数据',
            marker=dict(size=10, color='#58a6ff', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[pred_srt], y=[pred_fm],
            mode='markers', name='预测值',
            marker=dict(size=22, color='#f85149', symbol='star', line=dict(width=2, color='white'))
        ))
        fig.update_layout(
            title='SRT vs F/M 关系图',
            xaxis_title='SRT (天)',
            yaxis_title='F/M (%)',
            height=400,
            hovermode='closest',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 优化建议")
        
        if pred_fm > FM_MAX:
            st.warning(f"⚠️ 有机质占比偏高 ({pred_fm:.2f}%)，建议：减少进水量或增加MLSS浓度")
        elif pred_fm < FM_MIN:
            st.warning(f"⚠️ 有机质占比偏低 ({pred_fm:.2f}%)，建议：增加进水量或减少MLSS浓度")
        else:
            st.success(f"✅ 有机质占比正常 ({pred_fm:.2f}%)")
        
        if pred_srt > SRT_MAX:
            st.warning(f"⚠️ SRT偏高 ({pred_srt:.2f}天)，建议：减少污泥回流量，适当排泥")
        elif pred_srt < SRT_MIN:
            st.warning(f"⚠️ SRT偏低 ({pred_srt:.2f}天)，建议：增加污泥回流量")
        else:
            st.success(f"✅ SRT正常 ({pred_srt:.2f}天)")
        
        st.info(f"🌟 推荐最优污泥龄: **{opt_srt:.2f}天** (基于F/M={pred_fm:.1f}%优化)")
    else:
        st.info("💡 请先在左侧侧边栏输入参数，然后点击 '开始预测' 按钮")

# ===== Tab 2: 时间序列 =====
with tab2:
    st.markdown("### 📈 历史趋势分析")
    if date_col:
        time_target = st.selectbox(
            "选择指标查看时间序列",
            available_y + ['Qoutm3/d', 'BOD5 (mg/l)', 'CODcr(mg/l)'],
            format_func=lambda x: y_names_cn.get(x, x) if x in y_names_cn else x_names_cn.get(x, x),
            key="time_series"
        )
        if time_target:
            if time_target in y_data.columns:
                values = y_data[time_target]
                title = y_names_cn.get(time_target, time_target)
                color = '#58a6ff'
            else:
                values = X_data[time_target]
                title = x_names_cn.get(time_target, time_target)
                color = '#f0883e'
            
            ma_window = st.slider("移动平均窗口", min_value=1, max_value=10, value=3, key="ma_window")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=date_data, y=values,
                mode='lines+markers', name='原始数据',
                line=dict(color=color, width=2), marker=dict(size=5, color=color)
            ))
            if ma_window > 1:
                ma_values = values.rolling(window=ma_window).mean()
                fig.add_trace(go.Scatter(
                    x=date_data, y=ma_values,
                    mode='lines', name=f'{ma_window}日移动平均',
                    line=dict(color='#f85149', width=3, dash='dash')
                ))
            fig.update_layout(
                title=f'{title} 时间序列趋势',
                xaxis_title='日期',
                yaxis_title=title,
                height=400,
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("当前值", f"{values.iloc[-1]:.2f}")
            with col2: st.metric("平均值", f"{values.mean():.2f}")
            with col3: st.metric("变化率", f"{((values.iloc[-1] - values.iloc[0]) / values.iloc[0] * 100):.2f}%")
        
        st.markdown("---")
        st.markdown("### 📊 多指标对比")
        selected_multi = st.multiselect(
            "选择多个指标对比",
            available_y,
            default=available_y[:2],
            format_func=lambda x: y_names_cn.get(x, x),
            key="multi_series"
        )
        if selected_multi:
            fig2 = go.Figure()
            colors = ['#58a6ff', '#f85149', '#3fb950', '#d29922', '#f0883e']
            for idx, col in enumerate(selected_multi):
                values = y_data[col]
                normalized = (values - values.min()) / (values.max() - values.min())
                fig2.add_trace(go.Scatter(
                    x=date_data, y=normalized,
                    mode='lines',
                    name=y_names_cn.get(col, col),
                    line=dict(color=colors[idx % len(colors)], width=2.5)
                ))
            fig2.update_layout(
                title='多指标归一化对比',
                xaxis_title='日期',
                yaxis_title='归一化值',
                height=300,
                hovermode='x unified',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("⚠️ 数据中未找到日期列")

# ===== Tab 3: 特征重要性 + 热力图 =====
with tab3:
    st.markdown("### 📊 特征重要性柱状图")
        model_type = st.radio("选择模型", ['XGBoost', 'Random Forest', 'Lasso'], horizontal=True, key="importance")
    target = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key="importance_target")
    
    if target:
        model_key = {'XGBoost': 'xgb', 'Random Forest': 'rf', 'Lasso': 'lasso'}[model_type]
        if model_key == 'lasso':
            importance = np.abs(models[target]['lasso'].coef_)
        else:
            importance = models[target][model_key].feature_importances_
        
        sorted_idx = np.argsort(importance)[::-1]
        sorted_names = [x_names_en.get(available_X[i], available_X[i]) for i in sorted_idx]
        sorted_values = importance[sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(sorted_names, sorted_values, color='#58a6ff')
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold', color='white')
        ax.set_title(f'{model_type} - {y_names_en.get(target, target)} Feature Importance', fontsize=14, fontweight='bold', color='white')
        ax.invert_yaxis()
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')
        for i, v in enumerate(sorted_values):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center', color='white', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("### 🔥 特征相关性热力图")
    st.markdown("展示所有自变量与因变量之间的相关性")
    
    corr_data = pd.concat([X_data, y_data], axis=1)
    corr_matrix = corr_data.corr()
    rename_map = {**x_names_en, **y_names_en}
    corr_matrix = corr_matrix.rename(columns=rename_map, index=rename_map)
    
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
               fmt='.2f', square=True, linewidths=0.5, ax=ax,
               cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', color='white')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')
    plt.tight_layout()
    st.pyplot(fig)

# ===== Tab 4: 模型评价 =====
with tab4:
    st.markdown("### 📉 真实值 vs 预测值散点图")
    target_eval = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key='eval')
    
    if target_eval:
        model = models[target_eval]['xgb']
        y_test = models[target_eval]['y_test']
        y_pred = model.predict(models[target_eval]['X_test'])
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.6, color='#58a6ff', s=60)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
        ax.set_xlabel('True Value', fontsize=12, fontweight='bold', color='white')
        ax.set_ylabel('Predicted Value', fontsize=12, fontweight='bold', color='white')
        ax.set_title(f'{y_names_en.get(target_eval, target_eval)} - R² = {r2:.4f}', fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper left', facecolor='#0d1117', edgecolor='#30363d', labelcolor='white')
        ax.set_facecolor('#0d1117')
        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        st.pyplot(fig)
        
        # ===== 模型评价指标 =====
        st.markdown("---")
        st.markdown("### 📊 模型评价指标")
        st.markdown("R²、MSE、RMSE、MAE 综合评价模型性能")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² 分数", f"{r2:.4f}", help="越接近1越好")
        with col2:
            st.metric("MSE", f"{mse:.4f}", help="均方误差，越小越好")
        with col3:
            st.metric("RMSE", f"{rmse:.4f}", help="均方根误差，越小越好")
        with col4:
            st.metric("MAE", f"{mae:.4f}", help="平均绝对误差，越小越好")
        
        # ===== 各模型对比 =====
        st.markdown("---")
        st.markdown("### 📊 各模型性能对比")
        
        model_names = ['Linear', 'Lasso', 'RF', 'XGB']
        r2_values = [results[target_eval]['lr']['r2'], 
                     results[target_eval]['lasso']['r2'],
                     results[target_eval]['rf']['r2'], 
                     results[target_eval]['xgb']['r2']]
        rmse_values = [results[target_eval]['lr']['rmse'], 
                       results[target_eval]['lasso']['rmse'],
                       results[target_eval]['rf']['rmse'], 
                       results[target_eval]['xgb']['rmse']]
        
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R²对比
        bars1 = ax1.bar(model_names, r2_values, color=['#58a6ff', '#f0883e', '#3fb950', '#f85149'])
        ax1.set_ylabel('R² Score', fontsize=12, color='white')
        ax1.set_title('R² Comparison', fontsize=14, fontweight='bold', color='white')
        ax1.set_ylim(0, 1.05)
        ax1.set_facecolor('#0d1117')
        fig2.patch.set_facecolor('#0d1117')
        for bar, val in zip(bars1, r2_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)
        
        # RMSE对比
        bars2 = ax2.bar(model_names, rmse_values, color=['#58a6ff', '#f0883e', '#3fb950', '#f85149'])
        ax2.set_ylabel('RMSE', fontsize=12, color='white')
        ax2.set_title('RMSE Comparison', fontsize=14, fontweight='bold', color='white')
        ax2.set_facecolor('#0d1117')
        for bar, val in zip(bars2, rmse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig2)

# ===== Tab 5: SHAP解释（优化颜色版） =====
with tab5:
    st.markdown("### 🔍 SHAP 模型解释")
    st.markdown("SHAP值解释每个特征对预测结果的贡献")
    
    shap_target = st.selectbox("选择目标变量", available_y, format_func=lambda x: y_names_cn.get(x, x), key='shap')
    
    if st.button("生成 SHAP 解释", key="shap_btn"):
        with st.spinner("⏳ 计算SHAP值中..."):
            try:
                model = models[shap_target]['xgb']
                X_train = models[shap_target]['X_train']
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                feature_names = [x_names_en.get(col, col) for col in available_X]
                
                # ===== SHAP蜂群图 - 优化颜色 =====
                st.markdown("#### 📊 SHAP 蜂群图")
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # 设置背景为深色
                ax.set_facecolor('#0d1117')
                fig.patch.set_facecolor('#0d1117')
                
                # 生成SHAP图，使用高对比度颜色
                shap.summary_plot(
                    shap_values, 
                    X_train, 
                    feature_names=feature_names,
                    show=False,
                    color_bar=True,
                    cmap=plt.get_cmap('coolwarm')
                )
                
                # 手动调整轴标签颜色为白色
                ax = plt.gca()
                ax.tick_params(colors='white', labelsize=10)
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
                
                # 确保所有文本为白色
                for text in ax.texts:
                    text.set_color('white')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # ===== SHAP特征重要性条形图 =====
                st.markdown("#### 📊 SHAP 特征重要性")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                
                ax2.set_facecolor('#0d1117')
                fig2.patch.set_facecolor('#0d1117')
                
                shap.summary_plot(
                    shap_values, 
                    X_train, 
                    feature_names=feature_names, 
                    plot_type="bar",
                    show=False,
                    color='#58a6ff'
                )
                
                ax2 = plt.gca()
                ax2.tick_params(colors='white', labelsize=10)
                ax2.xaxis.label.set_color('white')
                ax2.yaxis.label.set_color('white')
                ax2.title.set_color('white')
                
                for patch in ax2.patches:
                    patch.set_color('#58a6ff')
                
                plt.tight_layout()
                st.pyplot(fig2)
                
                # ===== 单个预测解释 =====
                st.markdown("---")
                st.markdown("#### 🎯 当前输入的SHAP解释")
                
                input_array = np.array([st.session_state.input_values.get(col, 0) for col in available_X]).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                
                single_shap = explainer.shap_values(input_scaled)
                
                contrib_data = []
                for i, name in enumerate(feature_names):
                    shap_val = single_shap[0][i]
                    contrib_data.append({
                        '特征': name,
                        'SHAP值': f"{shap_val:.3f}",
                        '影响方向': "⬆️ 正向" if shap_val > 0 else "⬇️ 负向"
                    })
                
                contrib_df = pd.DataFrame(contrib_data)
                st.dataframe(contrib_df, use_container_width=True)
                
                pred_val = model.predict(input_scaled)[0]
                base_val = explainer.expected_value
                
                st.markdown(f"""
                <div style="background:#161b22;padding:1rem;border-radius:10px;border:1px solid #30363d;margin-top:1rem;">
                    <b style="color:#58a6ff;">预测 {y_names_cn.get(shap_target, shap_target)}:</b> 
                    <span style="color:#f0f6fc;font-size:1.2rem;font-weight:bold;">{pred_val:.3f}</span>
                    <br>
                    <b style="color:#8b949e;">基准值:</b> 
                    <span style="color:#f0f6fc;">{base_val:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"SHAP 计算失败: {e}")

# ===== Tab 6: 数据总览 =====
with tab6:
    st.markdown("### 📋 数据总览")
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("---")
    st.markdown("### 数据统计")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 自变量统计")
        st.dataframe(X_data.describe())
    with col2:
        st.markdown("#### 因变量统计")
        st.dataframe(y_data.describe())

st.markdown("---")
st.markdown("💧 **污水处理智能分析平台 v5.0** | 基于机器学习的多模型预测系统")
