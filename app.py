import io
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

G = 9.80665

st.set_page_config(page_title="MDOF RSA + Pushover Reconciliation PRO", layout="wide")

st.title("MDOF Response Spectrum + Nonlinear Pushover Reconciliation PRO")
st.caption(
    "STAAD weight/stiffness → modal RSA → static base shear comparison → "
    "first-mode pushover pattern → Mp-based storey yield capacity → pushover curve → ADRS"
)

with st.expander("Purpose and assumptions", expanded=True):
    st.markdown(
        r"""
This app is for **manual reconciliation** of MDOF modal RSA and simplified pushover analysis.

**Recommended units:** kN, m, sec. Input **floor seismic weight in kN**. The app converts weight to mass using:

\[
m = \frac{W}{g}
\]

The response spectrum uses a simplified UBC/NSCP-style shape based on **Ca** and **Cv**:

\[
S_a = 2.5C_a \quad \text{for } T \le T_s
\]

\[
S_a = \frac{C_v}{T} \quad \text{for } T > T_s
\]

where:

\[
T_s = \frac{C_v}{2.5C_a}
\]

This avoids the artificial vertical jump caused by forcing the transition at a fixed 0.5 sec.
        """
    )

# ---------------------------------------------------------
# Defaults
# ---------------------------------------------------------

def default_weight(n):
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Weight_kN": np.full(n, 2500.0)
    })


def default_k(n):
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Storey_stiffness_kN_per_m": np.linspace(40000, 25000, n)
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
# Helpers
# ---------------------------------------------------------

def shear_building_K(k_story):
    n = len(k_story)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        if i == 0:
            K[i, i] += k_story[i]
        else:
            K[i, i] += k_story[i]
            K[i - 1, i - 1] += k_story[i]
            K[i, i - 1] -= k_story[i]
            K[i - 1, i] -= k_story[i]
    return K


def parse_matrix(text, n, default_matrix):
    text = text.strip()
    if not text:
        return default_matrix, None
    try:
        rows = []
        for line in text.splitlines():
            if line.strip():
                rows.append([float(x) for x in line.replace(",", " ").split()])
        arr = np.array(rows, dtype=float)
        if arr.shape != (n, n):
            raise ValueError(f"Matrix must be {n} x {n}, but parsed shape is {arr.shape}")
        return arr, None
    except Exception as e:
        return default_matrix, str(e)


@st.cache_data(show_spinner=False)
def eig_analysis_cached(M_tuple, K_tuple):
    M = np.array(M_tuple, dtype=float)
    K = np.array(K_tuple, dtype=float)

    if M.shape != K.shape:
        raise ValueError("M and K must have the same shape.")
    if np.any(np.diag(M) <= 0):
        raise ValueError("All floor weights/masses must be positive.")

    A = np.linalg.solve(M, K)
    w2_raw, phi_raw = np.linalg.eig(A)
    w2 = np.real_if_close(w2_raw, tol=1000).astype(float)
    phi = np.real_if_close(phi_raw, tol=1000).astype(float)

    keep = np.isfinite(w2) & (w2 > 1e-9)
    w2 = w2[keep]
    phi = phi[:, keep]
    if len(w2) == 0:
        raise ValueError("No positive eigenvalues found. Check mass and stiffness inputs.")

    order = np.argsort(w2)
    w2 = w2[order]
    phi = phi[:, order]
    w = np.sqrt(w2)
    T = 2 * np.pi / w

    for j in range(phi.shape[1]):
        if phi[-1, j] < 0:
            phi[:, j] *= -1
        if abs(phi[-1, j]) > 1e-12:
            phi[:, j] /= phi[-1, j]

    return T, w, phi


def modal_props(M, phi):
    ones = np.ones(M.shape[0])
    total_mass = float(ones @ M @ ones)
    rows = []
    for j in range(phi.shape[1]):
        p = phi[:, j]
        mn = float(p @ M @ p)
        ln = float(p @ M @ ones)
        gamma = ln / mn
        meff = ln**2 / mn
        rows.append({
            "Mode": j + 1,
            "Gamma": gamma,
            "Generalized_mass": mn,
            "Effective_modal_mass": meff,
            "Mass_participation_%": 100 * meff / total_mass,
        })
    df = pd.DataFrame(rows)
    df["Cumulative_mass_%"] = df["Mass_participation_%"].cumsum()
    return df, total_mass


def build_ca_cv_spectrum(Ca, Cv, T_max=4.0, npts=300):
    T = np.linspace(0.01, T_max, npts)
    Ts = Cv / max(2.5 * Ca, 1e-12)
    Sa = np.where(T <= Ts, 2.5 * Ca, Cv / T)
    return pd.DataFrame({"T_sec": T, "Sa_g": Sa}), Ts


def interp_sa(T, spec_df):
    x = spec_df["T_sec"].to_numpy(float)
    y = spec_df["Sa_g"].to_numpy(float)
    order = np.argsort(x)
    return np.interp(T, x[order], y[order], left=y[order][0], right=y[order][-1])


def plot_xy(x, y, xlabel, ylabel, title, marker=True):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, y, marker="o" if marker else None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def storey_capacity(cap_df):
    df = cap_df.copy()
    df["Column_Vy_per_frame_kN"] = 2.0 * df["Column_Mp_kNm"] / df["Storey_height_m"]
    df["Beam_Vy_per_frame_kN"] = df["Beam_Mp_kNm"] / df["Storey_height_m"]
    df["Column_Vy_total_kN"] = df["Column_Vy_per_frame_kN"] * df["Frames"]
    df["Beam_Vy_total_kN"] = df["Beam_Vy_per_frame_kN"] * df["Frames"]
    df["Governing_mechanism"] = np.where(
        df["Column_Vy_total_kN"] <= df["Beam_Vy_total_kN"],
        "Column mechanism controls",
        "Beam mechanism controls"
    )
    df["Storey_yield_shear_kN"] = np.minimum(df["Column_Vy_total_kN"], df["Beam_Vy_total_kN"])
    return df


@st.cache_data(show_spinner=False)
def pushover_curve_cached(pattern_tuple, k_story_tuple, Vy_story_tuple, overstrength, post_ratio, nsteps):
    pattern = np.array(pattern_tuple, dtype=float)
    k_story = np.array(k_story_tuple, dtype=float)
    Vy_story = np.array(Vy_story_tuple, dtype=float)
    pattern = pattern / np.sum(pattern)

    n = len(k_story)
    shear_factor = np.array([pattern[i:].sum() for i in range(n)])
    V_y_base_by_story = Vy_story / np.maximum(shear_factor, 1e-12)
    V_global_y = float(np.min(V_y_base_by_story))
    critical = int(np.argmin(V_y_base_by_story)) + 1

    V = np.linspace(0, overstrength * V_global_y, nsteps)
    roof_delta = []
    yielded_count = []
    yielded_storeys_text = []

    for vb in V:
        story_shear = vb * shear_factor
        story_drift = np.zeros(n)
        yielded = []
        for i in range(n):
            if story_shear[i] <= Vy_story[i]:
                story_drift[i] = story_shear[i] / k_story[i]
            else:
                yielded.append(i + 1)
                dy = Vy_story[i] / k_story[i]
                post_k = max(post_ratio * k_story[i], 1e-9)
                story_drift[i] = dy + (story_shear[i] - Vy_story[i]) / post_k
        roof_delta.append(float(np.sum(story_drift)))
        yielded_count.append(len(yielded))
        yielded_storeys_text.append(", ".join(map(str, yielded)) if yielded else "-")

    push_df = pd.DataFrame({
        "Step": np.arange(1, nsteps + 1),
        "Base_shear_kN": V,
        "Roof_displacement_m": roof_delta,
        "Yielded_storey_count": yielded_count,
        "Yielded_storeys": yielded_storeys_text,
    })
    return push_df, V_global_y, critical, shear_factor, V_y_base_by_story


def to_cache_tuple(arr):
    return tuple(map(tuple, np.asarray(arr, dtype=float)))


def to_cache_vector(arr):
    return tuple(np.asarray(arr, dtype=float).tolist())

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.header("Model size and controls")
n = st.sidebar.slider("Number of storeys", 2, 10, 5)
num_modes = st.sidebar.slider("Number of modes for RSA", 1, n, min(n, 5))
input_method = st.sidebar.radio("Stiffness input method", ["Storey stiffness vector", "Full STAAD K matrix"])
st.sidebar.divider()
run_analysis = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------

st.header("1. STAAD-derived weight and stiffness input")
col1, col2 = st.columns(2)
with col1:
    weight_df = st.data_editor(default_weight(n), num_rows="fixed", use_container_width=True, key=f"weight_{n}")
with col2:
    k_df = st.data_editor(default_k(n), num_rows="fixed", use_container_width=True, key=f"k_{n}")

weights = weight_df["Weight_kN"].to_numpy(float)
masses = weights / G
k_story = k_df["Storey_stiffness_kN_per_m"].to_numpy(float)
M = np.diag(masses)
K_default = shear_building_K(k_story)

if input_method == "Full STAAD K matrix":
    st.markdown("Paste the condensed lateral stiffness matrix from STAAD, ordered from first floor DOF to roof DOF.")
    K_text = st.text_area("K matrix, kN/m", value="", height=160)
    K, parse_error = parse_matrix(K_text, n, K_default)
    if parse_error:
        st.warning(f"Could not parse matrix. Using default shear-building K. Details: {parse_error}")
else:
    K = K_default

with st.expander("Show assembled M and K matrices"):
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame(M), use_container_width=True)
    c2.dataframe(pd.DataFrame(K), use_container_width=True)

st.header("2. Corrected Auto Response Spectrum from Ca and Cv")
c1, c2, c3, c4 = st.columns(4)
with c1:
    Ca = st.number_input("Ca", min_value=0.01, max_value=2.00, value=0.44, step=0.01)
with c2:
    Cv = st.number_input("Cv", min_value=0.01, max_value=2.00, value=0.64, step=0.01)
with c3:
    T_max = st.number_input("Maximum period shown, sec", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
with c4:
    static_Cs = st.number_input("Static base shear coefficient Cs = V/W", min_value=0.001, max_value=2.0, value=0.12, step=0.005)

spec_df, Ts = build_ca_cv_spectrum(Ca, Cv, T_max=T_max)
st.info(f"Transition period: Ts = Cv / (2.5Ca) = {Ts:.3f} sec")
st.pyplot(plot_xy(spec_df["T_sec"], spec_df["Sa_g"], "Period T, sec", "Sa, g", "Corrected Ca-Cv Response Spectrum", marker=False))
with st.expander("Show generated spectrum table"):
    st.dataframe(spec_df.style.format(precision=4), use_container_width=True)

st.header("3. Beam/column plastic moment input and number of frames")
st.markdown("Input only the plastic moment capacities **per one frame in the analyzed axis**.")
cap_in = st.data_editor(default_capacity(n), num_rows="fixed", use_container_width=True, key=f"cap_{n}")
with st.expander("Yield capacity equations"):
    st.latex(r"V_{y,c,frame}=\frac{2M_{p,c}}{h}")
    st.latex(r"V_{y,b,frame}=\frac{M_{p,b}}{h}")
    st.latex(r"V_{y,storey}=\min(V_{y,c,total},V_{y,b,total})")

st.header("4. Pushover controls")
colA, colB, colC = st.columns(3)
with colA:
    overstrength = st.number_input("Push up to this × first-yield base shear", 1.00, 3.00, 1.50, 0.05)
with colB:
    post_ratio = st.number_input("Post-yield stiffness ratio", 0.001, 0.50, 0.05, 0.01)
with colC:
    nsteps = st.slider("Pushover steps", 30, 200, 80, 10)

pattern_source = st.selectbox("Pushover pattern source", ["First mode mass × phi", "RSA SRSS floor force shape"], index=0)

if not run_analysis:
    st.info("Set your inputs, then click **Run Analysis** in the sidebar.")
    st.stop()

try:
    with st.spinner("Running modal analysis, RSA, static comparison, pushover, and ADRS..."):
        T, w, phi = eig_analysis_cached(to_cache_tuple(M), to_cache_tuple(K))
        props, total_mass = modal_props(M, phi)
except Exception as e:
    st.error(f"Eigenvalue analysis failed. Check M and K inputs. Error: {e}")
    st.stop()

st.success("Analysis completed.")

Sa_modes = np.array([interp_sa(t, spec_df) for t in T[:num_modes]])
modal_forces = []
modal_base = []
for j in range(num_modes):
    p = phi[:, j]
    gamma = props.loc[j, "Gamma"]
    f = masses * p * gamma * Sa_modes[j] * G
    modal_forces.append(f)
    modal_base.append(np.sum(f))
modal_forces = np.array(modal_forces)
modal_base = np.array(modal_base)
rsa_floor_force_srss = np.sqrt(np.sum(modal_forces**2, axis=0))
rsa_base_shear_srss = float(np.sqrt(np.sum(modal_base**2)))

modal_table = props.copy()
modal_table.insert(1, "Period_sec", T[:len(modal_table)])

rsa_df = pd.DataFrame({
    "Mode": np.arange(1, num_modes + 1),
    "Period_sec": T[:num_modes],
    "Sa_g": Sa_modes,
    "Modal_base_shear_kN_signed": modal_base,
    "Modal_base_shear_kN_abs": np.abs(modal_base),
})

cap = storey_capacity(cap_in)
W_total = float(np.sum(weights))
V_static = static_Cs * W_total
scale_factor = V_static / rsa_base_shear_srss if rsa_base_shear_srss > 0 else np.nan

# Pattern
if pattern_source.startswith("First mode"):
    raw_pattern = masses * phi[:, 0]
    pattern_basis_text = "m_i × phi_i1"
else:
    raw_pattern = rsa_floor_force_srss
    pattern_basis_text = "RSA SRSS floor force shape"
raw_pattern = np.maximum(raw_pattern, 0)
if np.sum(raw_pattern) <= 0:
    raw_pattern = np.ones(n)
pattern = raw_pattern / np.sum(raw_pattern)

pattern_df = pd.DataFrame({
    "Storey": np.arange(1, n + 1),
    "Weight_kN": weights,
    "Mass_kNs2_per_m": masses,
    "Mode_1_phi_roof_normalized": phi[:, 0],
    "Raw_pattern_basis": raw_pattern,
    "Pattern_fraction": pattern,
    "Force_for_1kN_base_kN": pattern,
    "Force_scaled_to_static_V_kN": pattern * V_static,
    "Force_scaled_to_RSA_V_kN": pattern * rsa_base_shear_srss,
})

push_df, V_y_global, crit_storey, shear_factor, V_y_base_by_story = pushover_curve_cached(
    to_cache_vector(pattern),
    to_cache_vector(k_story),
    to_cache_vector(cap["Storey_yield_shear_kN"].to_numpy(float)),
    float(overstrength),
    float(post_ratio),
    int(nsteps)
)
H_total = float(cap["Storey_height_m"].sum())
push_df["Roof_drift_ratio"] = push_df["Roof_displacement_m"] / H_total

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1 Modal RSA",
    "2 Static vs Modal Base Shear",
    "3 First-Mode Pushover Pattern",
    "4 Yield Capacity",
    "5 Pushover Curve",
    "6 ADRS"
])

with tab1:
    st.header("Modal analysis and RSA")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(modal_table.style.format(precision=4), use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for j in range(min(num_modes, phi.shape[1])):
            ax.plot(phi[:, j], np.arange(1, n + 1), marker="o", label=f"Mode {j + 1}, T={T[j]:.3f}s")
        ax.set_xlabel("Relative displacement, roof normalized = 1")
        ax.set_ylabel("Storey")
        ax.set_title("Mode shapes")
        ax.grid(True, alpha=0.35)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Modal base shear table")
    st.dataframe(rsa_df.style.format(precision=4), use_container_width=True)
    st.metric("SRSS Modal/RSA base shear", f"{rsa_base_shear_srss:,.2f} kN")

with tab2:
    st.header("Static base shear vs modal RSA base shear")
    compare_df = pd.DataFrame({
        "Item": ["Total seismic weight W", "Static coefficient Cs", "Static base shear V = CsW", "RSA SRSS base shear", "Static/RSA scale factor"],
        "Value": [f"{W_total:,.2f} kN", f"{static_Cs:.4f}", f"{V_static:,.2f} kN", f"{rsa_base_shear_srss:,.2f} kN", f"{scale_factor:.3f}"]
    })
    st.table(compare_df)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(["Static V", "RSA SRSS V"], [V_static, rsa_base_shear_srss])
    ax.set_ylabel("Base shear, kN")
    ax.set_title("Static base shear vs modal RSA base shear")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    st.pyplot(fig)
    st.subheader("Per-mode base shear contribution")
    st.dataframe(rsa_df.style.format(precision=4), use_container_width=True)

with tab3:
    st.header("First-mode pushover force pattern")
    st.info(f"Pushover pattern basis used: {pattern_basis_text}")
    st.dataframe(pattern_df.style.format(precision=5), use_container_width=True)
    st.pyplot(plot_xy(pattern, np.arange(1, n + 1), "Lateral force fraction", "Storey", "First-mode pushover force pattern"))

with tab4:
    st.header("Yield capacity from beam/column plastic moments")
    show_cols = [
        "Storey", "Storey_height_m", "Frames", "Column_Mp_kNm", "Beam_Mp_kNm",
        "Column_Vy_per_frame_kN", "Beam_Vy_per_frame_kN",
        "Column_Vy_total_kN", "Beam_Vy_total_kN",
        "Governing_mechanism", "Storey_yield_shear_kN"
    ]
    st.dataframe(cap[show_cols].style.format(precision=3), use_container_width=True)

with tab5:
    st.header("Nonlinear pushover curve and result table")
    c1, c2, c3 = st.columns(3)
    c1.metric("First-yield base shear", f"{V_y_global:,.2f} kN")
    c2.metric("Critical first-yield storey", f"Storey {crit_storey}")
    c3.metric("RSA / first-yield ratio", f"{rsa_base_shear_srss / V_y_global:.3f}")
    st.pyplot(plot_xy(push_df["Roof_displacement_m"], push_df["Base_shear_kN"], "Roof displacement, m", "Base shear, kN", "Nonlinear pushover curve"))
    st.subheader("Pushover curve tabulation")
    st.dataframe(
        push_df.style.format({
            "Base_shear_kN": "{:,.3f}",
            "Roof_displacement_m": "{:.6f}",
            "Roof_drift_ratio": "{:.4%}",
        }),
        use_container_width=True
    )
    st.subheader("Storey yield trigger check")
    yield_trigger_df = pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Pattern_shear_factor": shear_factor,
        "Storey_yield_shear_kN": cap["Storey_yield_shear_kN"].to_numpy(float),
        "Base_shear_causing_yield_kN": V_y_base_by_story
    })
    st.dataframe(yield_trigger_df.style.format(precision=4), use_container_width=True)

with tab6:
    st.header("ADRS capacity spectrum reconciliation")
    Gamma1 = float(props.loc[0, "Gamma"])
    Mstar1 = float(props.loc[0, "Effective_modal_mass"])
    phi_roof1 = float(phi[-1, 0])
    push_df["Sd_m"] = push_df["Roof_displacement_m"] / max(Gamma1 * phi_roof1, 1e-12)
    push_df["Sa_g"] = push_df["Base_shear_kN"] / max(Mstar1 * G, 1e-12)
    demand = spec_df.copy().sort_values("T_sec")
    demand["Sd_m"] = demand["Sa_g"] * G * (demand["T_sec"] / (2 * np.pi)) ** 2

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(push_df["Sd_m"], push_df["Sa_g"], marker="o", markevery=max(1, len(push_df) // 12), label="Capacity spectrum")
    ax.plot(demand["Sd_m"], demand["Sa_g"], label="Elastic demand spectrum")
    ax.set_xlabel("Spectral displacement Sd, m")
    ax.set_ylabel("Spectral acceleration Sa, g")
    ax.set_title("ADRS reconciliation")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    if push_df["Sd_m"].max() > 0 and demand["Sd_m"].max() > 0:
        sd_grid = np.linspace(0, min(push_df["Sd_m"].max(), demand["Sd_m"].max()), 300)
        cap_sa = np.interp(sd_grid, push_df["Sd_m"], push_df["Sa_g"])
        dem_sa = np.interp(sd_grid, demand["Sd_m"], demand["Sa_g"])
        idx = int(np.argmin(np.abs(cap_sa - dem_sa)))
        pp_sd = sd_grid[idx]
        pp_sa = cap_sa[idx]
        pp_roof = pp_sd * Gamma1 * phi_roof1
        st.success(
            f"Approximate ADRS performance point: Sd = {pp_sd:.4f} m, "
            f"Sa = {pp_sa:.3f} g, roof displacement ≈ {pp_roof:.4f} m, "
            f"drift ≈ {pp_roof / H_total:.3%}"
        )

st.divider()
st.header("Download results")
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    weight_df.to_excel(writer, sheet_name="Weights", index=False)
    k_df.to_excel(writer, sheet_name="Storey stiffness", index=False)
    pd.DataFrame(K).to_excel(writer, sheet_name="K matrix", index=False)
    spec_df.to_excel(writer, sheet_name="Spectrum", index=False)
    modal_table.to_excel(writer, sheet_name="Modal", index=False)
    rsa_df.to_excel(writer, sheet_name="RSA", index=False)
    pattern_df.to_excel(writer, sheet_name="Pushover pattern", index=False)
    cap.to_excel(writer, sheet_name="Yield capacity", index=False)
    push_df.to_excel(writer, sheet_name="Pushover_ADRS", index=False)

st.download_button(
    "Download calculation workbook",
    data=out.getvalue(),
    file_name="mdof_rsa_pushover_reconciliation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.warning("Use this as a transparent reconciliation and teaching app only. For design approval, verify with code-compliant nonlinear static procedures and validated structural software.")
