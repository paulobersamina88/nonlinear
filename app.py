import io
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

G = 9.80665

st.set_page_config(
    page_title="MDOF RSA + Pushover Reconciliation PRO",
    layout="wide"
)

st.title("MDOF Response Spectrum + Nonlinear Pushover Reconciliation PRO")

# ---------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------

def default_mass(n):
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Weight_kN": np.full(n, 2500.0)
    })

def default_k(n):
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Storey_stiffness_kN_per_m": np.linspace(40000, 20000, n)
    })

def default_capacity(n):
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Storey_height_m": np.full(n, 3.0),
        "Frames": np.full(n, 3),
        "Column_Mp_kNm": np.linspace(1500, 900, n),
        "Beam_Mp_kNm": np.linspace(900, 600, n),
    })

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------

def shear_building_K(k):
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            K[i, i] += k[i]
        else:
            K[i, i] += k[i]
            K[i-1, i-1] += k[i]
            K[i, i-1] -= k[i]
            K[i-1, i] -= k[i]
    return K

def eig_analysis(M, K):
    A = np.linalg.solve(M, K)
    w2, phi = np.linalg.eig(A)

    w2 = np.real(w2)
    phi = np.real(phi)

    idx = w2 > 1e-6
    w2 = w2[idx]
    phi = phi[:, idx]

    order = np.argsort(w2)
    w2 = w2[order]
    phi = phi[:, order]

    w = np.sqrt(w2)
    T = 2*np.pi / w

    for i in range(phi.shape[1]):
        phi[:, i] /= phi[-1, i]

    return T, phi

def compute_capacity(df):
    df = df.copy()

    df["Column_Vy_frame"] = 2 * df["Column_Mp_kNm"] / df["Storey_height_m"]
    df["Beam_Vy_frame"] = df["Beam_Mp_kNm"] / df["Storey_height_m"]

    df["Column_total"] = df["Column_Vy_frame"] * df["Frames"]
    df["Beam_total"] = df["Beam_Vy_frame"] * df["Frames"]

    df["Vy_storey"] = np.minimum(df["Column_total"], df["Beam_total"])

    return df

def plot_xy(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.grid()
    return fig

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("Controls")
n = st.sidebar.slider("Storeys", 2, 10, 5)
run = st.sidebar.button("Run Analysis")

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

st.header("1. Weight Input (kN)")
mass_df = st.data_editor(default_mass(n))

st.header("2. Stiffness Input")
k_df = st.data_editor(default_k(n))

st.header("3. Plastic Moment Input")
cap_df = st.data_editor(default_capacity(n))

# ---------------------------------------------------------
# AUTO RESPONSE SPECTRUM
# ---------------------------------------------------------

st.header("4. Auto Response Spectrum")

col1, col2 = st.columns(2)

with col1:
    Ca = st.number_input("Ca", value=0.44)

with col2:
    Cv = st.number_input("Cv", value=0.64)

T_vals = np.linspace(0.01, 4, 100)

Sa = []
for T in T_vals:
    if T <= 0.5:
        Sa.append(2.5 * Ca)
    else:
        Sa.append(Cv / T)

spec_df = pd.DataFrame({"T": T_vals, "Sa": Sa})

st.pyplot(plot_xy(T_vals, Sa, "Response Spectrum"))

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if not run:
    st.info("Click Run Analysis")
    st.stop()

# ---------------------------------------------------------
# COMPUTE
# ---------------------------------------------------------

weights = mass_df["Weight_kN"].to_numpy()
masses = weights / G
M = np.diag(masses)

k = k_df["Storey_stiffness_kN_per_m"].to_numpy()
K = shear_building_K(k)

T, phi = eig_analysis(M, K)

st.header("5. Modal Result")
st.write("Fundamental Period:", T[0])

# ---------------------------------------------------------
# CAPACITY
# ---------------------------------------------------------

cap = compute_capacity(cap_df)

st.header("6. Yield Capacity Table")

st.dataframe(cap[[
    "Storey",
    "Column_Mp_kNm",
    "Beam_Mp_kNm",
    "Column_Vy_frame",
    "Beam_Vy_frame",
    "Column_total",
    "Beam_total",
    "Vy_storey"
]])

# ---------------------------------------------------------
# PUSHOVER
# ---------------------------------------------------------

st.header("7. Pushover Curve")

Vy = cap["Vy_storey"].to_numpy()

pattern = masses * phi[:, 0]
pattern = pattern / np.sum(pattern)

V = np.linspace(0, min(Vy)*1.5, 50)
delta = []

for vb in V:
    d = 0
    for i in range(n):
        shear = vb * np.sum(pattern[i:])
        if shear < Vy[i]:
            d += shear / k[i]
        else:
            d += Vy[i]/k[i] + (shear - Vy[i])/(0.05*k[i])
    delta.append(d)

st.pyplot(plot_xy(delta, V, "Pushover Curve"))

# ---------------------------------------------------------
# ADRS
# ---------------------------------------------------------

st.header("8. ADRS")

Sd = np.array(delta)
Sa = V / np.sum(weights)

fig, ax = plt.subplots()
ax.plot(Sd, Sa, label="Capacity")
ax.plot(T_vals, spec_df["Sa"], label="Demand")
ax.legend()
ax.grid()

st.pyplot(fig)

st.success("Analysis complete")
