import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import os

from core.model import AeroNet

st.set_page_config(
    page_title="气动系数预测神经网络",
    page_icon="✈️",
    layout="wide",
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "aero_net.pth"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5em; font-weight: 800;
        background: linear-gradient(135deg, #1a3a5c 0%, #2c7ba8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">✈️ 气动系数预测神经网络</div>', unsafe_allow_html=True)
st.caption("基于飞行器控制与信息工程 —— 用神经网络拟合气动特性")

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ 模型未训练")
    st.info("请先在终端运行：`python -m core.train`")
    st.stop()


@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = AeroNet(hidden=64)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stats"], ckpt["history"]


model, stats, history = load_model()

st.sidebar.header("🛠️ 输入参数")

Ma = st.sidebar.slider("马赫数 Ma", 0.3, 0.85, 0.78, 0.01)
alpha = st.sidebar.slider("迎角 α (°)", -4.0, 14.0, 4.0, 0.5)
AR = st.sidebar.slider("展弦比 AR", 5.0, 13.0, 9.5, 0.1)

x_raw = np.array([[Ma, alpha, AR]], dtype=np.float32)
x_norm = (x_raw - stats["X_mean"]) / stats["X_std"]
x_tensor = torch.tensor(x_norm)

with torch.no_grad():
    y_pred_norm = model(x_tensor).numpy()

y_pred = y_pred_norm * stats["y_std"] + stats["y_mean"]
CL_pred, CD_pred = float(y_pred[0, 0]), float(y_pred[0, 1])

sweep = np.deg2rad(25.0)
e = 0.80 * np.cos(sweep)
a0 = 2 * np.pi * np.cos(sweep)
CL_alpha_deg = (a0 / (1 + a0 / (np.pi * AR * e))) * np.pi / 180
CL_theory = CL_alpha_deg * (alpha - (-2.0))
CL_max = min(1.4 + 0.3 * np.log10(AR) * np.cos(sweep), 1.8)
CL_theory = min(CL_theory, CL_max)

CD0 = (0.018 if Ma < 0.7 else
       0.018 + 0.06 * (Ma - 0.7) ** 2 if Ma < 0.85 else
       0.025 + 0.1 * (Ma - 0.85))
k = 1.0 / (np.pi * AR * e)
CD_theory = CD0 + k * CL_theory ** 2

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🤖 神经网络预测")
    c1, c2 = st.columns(2)
    c1.metric("升力系数 CL", f"{CL_pred:.4f}")
    c2.metric("阻力系数 CD", f"{CD_pred:.5f}")
    LD = CL_pred / CD_pred if CD_pred > 0 else 0
    st.metric("升阻比 L/D", f"{LD:.2f}")

with col2:
    st.markdown("### 📐 理论公式")
    c1, c2 = st.columns(2)
    c1.metric("升力系数 CL", f"{CL_theory:.4f}")
    c2.metric("阻力系数 CD", f"{CD_theory:.5f}")
    LD_t = CL_theory / CD_theory if CD_theory > 0 else 0
    st.metric("升阻比 L/D", f"{LD_t:.2f}")

st.divider()
st.markdown("### 📊 预测 vs 理论")

c1, c2, c3 = st.columns(3)
c1.metric("CL 误差", f"{CL_pred - CL_theory:+.4f}",
          f"{(CL_pred - CL_theory) / CL_theory * 100:+.2f}%")
c2.metric("CD 误差", f"{CD_pred - CD_theory:+.5f}",
          f"{(CD_pred - CD_theory) / CD_theory * 100:+.2f}%")
c3.metric("L/D 误差", f"{LD - LD_t:+.2f}",
          f"{(LD - LD_t) / LD_t * 100:+.2f}%")

st.divider()
st.markdown("### 📈 随迎角变化曲线")

alpha_range = np.linspace(-4, 14, 50)
x_scan = np.stack([
    np.full_like(alpha_range, Ma),
    alpha_range,
    np.full_like(alpha_range, AR),
], axis=1).astype(np.float32)
x_scan_norm = (x_scan - stats["X_mean"]) / stats["X_std"]
with torch.no_grad():
    y_scan_norm = model(torch.tensor(x_scan_norm)).numpy()
y_scan = y_scan_norm * stats["y_std"] + stats["y_mean"]

CL_scan = CL_alpha_deg * (alpha_range - (-2.0))
CL_scan = np.minimum(CL_scan, CL_max)
CD_scan = CD0 + k * CL_scan ** 2

fig = go.Figure()
fig.add_trace(go.Scatter(x=alpha_range, y=y_scan[:, 0],
                         name="CL 神经网络", line=dict(color="#1f77b4", width=3)))
fig.add_trace(go.Scatter(x=alpha_range, y=CL_scan,
                         name="CL 理论", line=dict(color="#1f77b4", dash="dash", width=2)))
fig.add_trace(go.Scatter(x=alpha_range, y=y_scan[:, 1] * 20,
                         name="CD×20 神经网络", line=dict(color="#d62728", width=3)))
fig.add_trace(go.Scatter(x=alpha_range, y=CD_scan * 20,
                         name="CD×20 理论", line=dict(color="#d62728", dash="dash", width=2)))
fig.update_layout(
    xaxis_title="迎角 α (°)",
    yaxis_title="系数",
    height=450, template="plotly_white",
    hovermode="x unified",
)
st.plotly_chart(fig, width='stretch')

st.divider()
with st.expander("📉 训练过程", expanded=False):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=history["train_loss"], name="训练损失",
                              line=dict(color="#1f77b4", width=2)))
    fig2.add_trace(go.Scatter(y=history["val_loss"], name="验证损失",
                              line=dict(color="#d62728", width=2)))
    fig2.update_layout(
        xaxis_title="Epoch", yaxis_title="MSE Loss (标准化)",
        height=350, template="plotly_white",
    )
    st.plotly_chart(fig2, width='stretch')

st.divider()
st.caption("✈️ 神经网络气动系数预测 · PyTorch + Streamlit")