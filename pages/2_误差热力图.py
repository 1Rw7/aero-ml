import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import AeroNet

st.set_page_config(page_title="误差热力图", page_icon="🔥", layout="wide")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "aero_net.pth"
)

st.title("🔥 误差热力图")
st.caption("看神经网络在哪些区域误差大 —— 评估模型适用域")

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
    X = np.stack([Ma_arr, alpha_arr, AR_arr], axis=1).astype(np.float32)
    X_norm = (X - stats["X_mean"]) / stats["X_std"]
    with torch.no_grad():
        y_norm = model(torch.tensor(X_norm)).numpy()
    y = y_norm * stats["y_std"] + stats["y_mean"]
    return y[:, 0], y[:, 1]


def theory_batch(Ma_arr, alpha_arr, AR_arr):
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


# ---------- 侧边栏 ----------
st.sidebar.header("🔥 热力图设置")

AR_fixed = st.sidebar.slider("固定展弦比 AR", 5.0, 13.0, 9.5, 0.1)
n_grid = st.sidebar.slider("网格密度", 20, 60, 40, 5)
metric = st.sidebar.radio("显示指标", ["CL 误差", "CD 误差"])

# ---------- 扫描网格 ----------
Ma_grid = np.linspace(0.3, 0.85, n_grid)
alpha_grid = np.linspace(-4, 14, n_grid)
Ma_mesh, alpha_mesh = np.meshgrid(Ma_grid, alpha_grid)
AR_mesh = np.full_like(Ma_mesh, AR_fixed)

Ma_flat = Ma_mesh.flatten()
alpha_flat = alpha_mesh.flatten()
AR_flat = AR_mesh.flatten()

CL_nn, CD_nn = predict_batch(Ma_flat, alpha_flat, AR_flat)
CL_th, CD_th = theory_batch(Ma_flat, alpha_flat, AR_flat)

CL_err = np.abs(CL_nn - CL_th).reshape(Ma_mesh.shape)
CD_err = np.abs(CD_nn - CD_th).reshape(Ma_mesh.shape)

# ---------- 画热力图 ----------
if metric == "CL 误差":
    err_data = CL_err
    colorbar_title = "|ΔCL|"
    title = f"升力系数误差热力图（AR = {AR_fixed}）"
else:
    err_data = CD_err
    colorbar_title = "|ΔCD|"
    title = f"阻力系数误差热力图（AR = {AR_fixed}）"

fig = go.Figure(data=go.Heatmap(
    x=Ma_grid, y=alpha_grid, z=err_data,
    colorscale="Hot_r",
    colorbar=dict(title=colorbar_title),
))

fig.update_layout(
    title=title,
    xaxis_title="马赫数 Ma",
    yaxis_title="迎角 α (°)",
    height=600, template="plotly_white",
)
st.plotly_chart(fig, width='stretch')

# ---------- 统计信息 ----------
st.divider()
st.markdown("### 📊 误差统计")

col1, col2, col3, col4 = st.columns(4)
col1.metric("平均误差", f"{err_data.mean():.6f}")
col2.metric("最大误差", f"{err_data.max():.6f}")
col3.metric("最小误差", f"{err_data.min():.6f}")
col4.metric("误差标准差", f"{err_data.std():.6f}")

# 最大误差位置
max_idx = np.unravel_index(np.argmax(err_data), err_data.shape)
max_Ma = Ma_grid[max_idx[1]]
max_alpha = alpha_grid[max_idx[0]]

st.info(f"""
🎯 **最大误差出现在**：

- 马赫数 Ma = **{max_Ma:.3f}**
- 迎角 α = **{max_alpha:.2f}°**
- 误差值 = **{err_data.max():.6f}**

**工程意义**：

- 深色区域 = 误差大 = 模型不太可靠的区域
- 浅色区域 = 误差小 = 模型精度高的区域
- 这个"误差地图"告诉你：**在实际使用时，哪些工况下可以放心用神经网络，哪些不行**
""")

# ---------- 3D 表面图 ----------
st.divider()
st.markdown("### 🌐 3D 误差表面")

fig3d = go.Figure(data=[go.Surface(
    x=Ma_grid, y=alpha_grid, z=err_data,
    colorscale="Viridis",
    colorbar=dict(title=colorbar_title),
)])
fig3d.update_layout(
    scene=dict(
        xaxis_title="马赫数 Ma",
        yaxis_title="迎角 α (°)",
        zaxis_title=colorbar_title,
    ),
    height=600,
)
st.plotly_chart(fig3d, width='stretch')

st.divider()
st.markdown("""
### 💡 观察要点

- **浅色区（误差小）**：模型学得好的区域（通常是训练数据的密集区）
- **深色区（误差大）**：模型学得不好的区域（可能是训练数据稀疏或边界效应）
- **最大误差点**：显示具体位置和数值

**在飞控应用中**：

- 如果飞行器的实际工况落**浅色区** → 放心用神经网络
- 落在**深色区** → 需要额外验证，或回退到物理模型
""")

st.caption("✈️ 误差热力图 · 评估神经网络模型适用域")