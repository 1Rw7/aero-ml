import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import AeroNet

st.set_page_config(page_title="参数敏感性", page_icon="📈", layout="wide")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "aero_net.pth"
)

st.title("📈 参数敏感性分析")
st.caption("固定两个参数，扫描第三个，观察气动系数的变化趋势")

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ 模型未训练")
    st.stop()


@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = AeroNet(hidden=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stats"]


model, stats = load_model()


def predict_batch(Ma_arr, alpha_arr, AR_arr):
    """批量预测"""
    X = np.stack([Ma_arr, alpha_arr, AR_arr], axis=1).astype(np.float32)
    X_norm = (X - stats["X_mean"]) / stats["X_std"]
    with torch.no_grad():
        y_norm = model(torch.tensor(X_norm)).numpy()
    y = y_norm * stats["y_std"] + stats["y_mean"]
    return y[:, 0], y[:, 1]


def theory_batch(Ma_arr, alpha_arr, AR_arr):
    """理论公式批量计算"""
    sweep = np.deg2rad(25.0)
    e = 0.80 * np.cos(sweep)
    a0 = 2 * np.pi * np.cos(sweep)
    CL_alpha_deg = (a0 / (1 + a0 / (np.pi * AR_arr * e))) * np.pi / 180
    CL = CL_alpha_deg * (alpha_arr - (-2.0))
    CL_max = np.minimum(1.4 + 0.3 * np.log10(AR_arr) * np.cos(sweep), 1.8)
    CL = np.minimum(CL, CL_max)
    CD0 = np.where(
        Ma_arr < 0.7, 0.018,
        np.where(Ma_arr < 0.85,
                 0.018 + 0.06 * (Ma_arr - 0.7) ** 2,
                 0.025 + 0.1 * (Ma_arr - 0.85))
    )
    k = 1.0 / (np.pi * AR_arr * e)
    CD = CD0 + k * CL ** 2
    return CL, CD


# ---------- 扫描选择 ----------
st.sidebar.header("🎯 扫描设置")

scan_choice = st.sidebar.radio(
    "选择扫描变量",
    ["迎角 α", "马赫数 Ma", "展弦比 AR"],
    index=0,
)

st.sidebar.divider()
st.sidebar.header("🔒 固定参数")

# 根据扫描变量，显示固定的另外两个
if scan_choice == "迎角 α":
    Ma_fixed = st.sidebar.slider("马赫数 Ma", 0.3, 0.85, 0.78, 0.01)
    AR_fixed = st.sidebar.slider("展弦比 AR", 5.0, 13.0, 9.5, 0.1)
    x_range = np.linspace(-4, 14, 60)
    x_label = "迎角 α (°)"
    Ma_arr = np.full_like(x_range, Ma_fixed)
    alpha_arr = x_range
    AR_arr = np.full_like(x_range, AR_fixed)
elif scan_choice == "马赫数 Ma":
    alpha_fixed = st.sidebar.slider("迎角 α (°)", -4.0, 14.0, 4.0, 0.5)
    AR_fixed = st.sidebar.slider("展弦比 AR", 5.0, 13.0, 9.5, 0.1)
    x_range = np.linspace(0.3, 0.85, 60)
    x_label = "马赫数 Ma"
    Ma_arr = x_range
    alpha_arr = np.full_like(x_range, alpha_fixed)
    AR_arr = np.full_like(x_range, AR_fixed)
else:  # AR
    Ma_fixed = st.sidebar.slider("马赫数 Ma", 0.3, 0.85, 0.78, 0.01)
    alpha_fixed = st.sidebar.slider("迎角 α (°)", -4.0, 14.0, 4.0, 0.5)
    x_range = np.linspace(5, 13, 60)
    x_label = "展弦比 AR"
    Ma_arr = np.full_like(x_range, Ma_fixed)
    alpha_arr = np.full_like(x_range, alpha_fixed)
    AR_arr = x_range


# ---------- 批量预测 ----------
CL_nn, CD_nn = predict_batch(Ma_arr, alpha_arr, AR_arr)
CL_th, CD_th = theory_batch(Ma_arr, alpha_arr, AR_arr)
LD_nn = CL_nn / CD_nn
LD_th = CL_th / CD_th

# ---------- 图表 1：CL 对比 ----------
st.markdown(f"### 📊 升力系数 CL 随 {scan_choice} 变化")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=x_range, y=CL_nn, mode="lines",
                          name="神经网络",
                          line=dict(color="#1f77b4", width=3)))
fig1.add_trace(go.Scatter(x=x_range, y=CL_th, mode="lines",
                          name="理论公式",
                          line=dict(color="#1f77b4", dash="dash", width=2)))
fig1.update_layout(
    xaxis_title=x_label, yaxis_title="升力系数 CL",
    height=400, template="plotly_white", hovermode="x unified",
)
st.plotly_chart(fig1, width='stretch')

# ---------- 图表 2：CD 对比 ----------
st.markdown(f"### 📉 阻力系数 CD 随 {scan_choice} 变化")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x_range, y=CD_nn, mode="lines",
                          name="神经网络",
                          line=dict(color="#d62728", width=3)))
fig2.add_trace(go.Scatter(x=x_range, y=CD_th, mode="lines",
                          name="理论公式",
                          line=dict(color="#d62728", dash="dash", width=2)))
fig2.update_layout(
    xaxis_title=x_label, yaxis_title="阻力系数 CD",
    height=400, template="plotly_white", hovermode="x unified",
)
st.plotly_chart(fig2, width='stretch')

# ---------- 图表 3：L/D 对比 ----------
st.markdown(f"### 🚀 升阻比 L/D 随 {scan_choice} 变化")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x_range, y=LD_nn, mode="lines",
                          name="神经网络",
                          line=dict(color="#2ca02c", width=3)))
fig3.add_trace(go.Scatter(x=x_range, y=LD_th, mode="lines",
                          name="理论公式",
                          line=dict(color="#2ca02c", dash="dash", width=2)))
fig3.update_layout(
    xaxis_title=x_label, yaxis_title="升阻比 L/D",
    height=400, template="plotly_white", hovermode="x unified",
)
st.plotly_chart(fig3, width='stretch')

# ---------- 关键指标 ----------
st.divider()
st.markdown("### 🎯 关键观察点")

col1, col2, col3, col4 = st.columns(4)

if scan_choice == "迎角 α":
    CL_max_idx = np.argmax(CL_nn)
    CL_max_val = CL_nn[CL_max_idx]
    alpha_stall = x_range[CL_max_idx]
    LD_max_idx = np.argmax(LD_nn)
    LD_max_val = LD_nn[LD_max_idx]
    alpha_best = x_range[LD_max_idx]

    col1.metric("最大 CL", f"{CL_max_val:.3f}")
    col2.metric("失速迎角", f"{alpha_stall:.1f}°")
    col3.metric("最大 L/D", f"{LD_max_val:.2f}")
    col4.metric("最佳迎角", f"{alpha_best:.1f}°")
elif scan_choice == "马赫数 Ma":
    CD0_min_idx = np.argmin(CD_nn)
    LD_max_idx = np.argmax(LD_nn)
    col1.metric("最小 CD", f"{CD_nn[CD0_min_idx]:.5f}")
    col2.metric("最省油 Ma", f"{x_range[CD0_min_idx]:.3f}")
    col3.metric("最大 L/D", f"{LD_nn[LD_max_idx]:.2f}")
    col4.metric("最佳巡航 Ma", f"{x_range[LD_max_idx]:.3f}")
else:  # AR
    LD_max_idx = np.argmax(LD_nn)
    col1.metric("最大 L/D", f"{LD_nn[LD_max_idx]:.2f}")
    col2.metric("最佳展弦比", f"{x_range[LD_max_idx]:.1f}")
    col3.metric("当前 AR 范围", f"5 ~ 13")
    col4.metric("L/D 提升", f"{LD_nn[-1] - LD_nn[0]:.2f}")

st.info("""
💡 **工程含义**：

- **迎角 α**：升力随 α 线性增长，超过失速迎角后下降
- **马赫数 Ma**：跨音速区（Ma > 0.7）阻力急剧增加 → 阻力发散
- **展弦比 AR**：AR 越大，诱导阻力越小，升阻比越大（但结构重、翼展大）
""")

st.caption("✈️ 参数敏感性分析 · 神经网络 vs 理论公式")