
import io
import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # Streamlit Cloud-safe backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.linalg import eigh

G = 9.80665

st.set_page_config(
    page_title="MDOF RSA + Pushover Reconciliation PRO",
    layout="wide"
)

st.title("MDOF Response Spectrum + Nonlinear Pushover Reconciliation PRO")
st.caption(
    "STAAD mass/stiffness → modal RSA → first-mode pushover pattern → "
    "beam/column plastic capacity → nonlinear pushover → ADRS reconciliation"
)

with st.expander("Purpose and assumptions", expanded=True):
    st.markdown(
        """
This app is intended for **manual reconciliation** of MDOF dynamic analysis and simplified pushover analysis.

**Recommended units:** kN, m, sec. Use mass in **kN·s²/m**.  
If you have seismic weight in kN, use:

\[
m = \\frac{W}{g}
\]

**Important simplification:** storey yield capacity is estimated from plastic moment capacity using a transparent storey mechanism approximation.  
It is useful for teaching, preliminary checking, and reconciliation, not final certification.
        """
    )

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def default_mass(n):
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Mass_kNs2_per_m": np.full(n, 250.0)
    })


def default_k(n):
    vals = np.linspace(140000, 70000, n)
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Storey_stiffness_kN_per_m": vals
    })


def default_spectrum():
    return pd.DataFrame({
        "T_sec": [0.00, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00],
        "Sa_g":  [0.35, 0.80, 1.00, 1.00, 0.70, 0.47, 0.35, 0.23, 0.18],
    })


def default_capacity(n):
    return pd.DataFrame({
        "Storey": np.arange(1, n + 1),
        "Storey_height_m": np.full(n, 3.0),
        "Frames_in_axis": np.full(n, 3),
        "Sum_column_Mp_per_frame_kNm": np.linspace(1500, 900, n),
        "Column_factor": np.full(n, 2.0),
        "Sum_beam_Mp_per_frame_kNm": np.linspace(900, 600, n),
        "Beam_factor": np.full(n, 1.0),
    })


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
                row = [float(x) for x in line.replace(",", " ").split()]
                rows.append(row)

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

    w2, phi = eigh(K, M)

    keep = w2 > 1e-9
    w2 = w2[keep]
    phi = phi[:, keep]

    order = np.argsort(w2)
    w2 = w2[order]
    phi = phi[:, order]

    w = np.sqrt(w2)
    T = 2 * np.pi / w

    # Normalize roof component positive and equal to 1
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


def interp_sa(T, spec_df):
    x = spec_df["T_sec"].to_numpy(float)
    y = spec_df["Sa_g"].to_numpy(float)

    order = np.argsort(x)

    return np.interp(
        T,
        x[order],
        y[order],
        left=y[order][0],
        right=y[order][-1]
    )


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

    df["Column_component_kN"] = (
        df["Frames_in_axis"]
        * df["Column_factor"]
        * df["Sum_column_Mp_per_frame_kNm"]
        / df["Storey_height_m"]
    )

    df["Beam_component_kN"] = (
        df["Frames_in_axis"]
        * df["Beam_factor"]
        * df["Sum_beam_Mp_per_frame_kNm"]
        / df["Storey_height_m"]
    )

    df["Storey_yield_shear_kN"] = (
        df["Column_component_kN"] + df["Beam_component_kN"]
    )

    return df


@st.cache_data(show_spinner=False)
def pushover_curve_cached(
    pattern_tuple,
    k_story_tuple,
    Vy_story_tuple,
    overstrength,
    post_ratio,
    nsteps
):
    pattern = np.array(pattern_tuple, dtype=float)
    k_story = np.array(k_story_tuple, dtype=float)
    Vy_story = np.array(Vy_story_tuple, dtype=float)

    pattern = pattern / np.sum(pattern)
    n = len(k_story)

    # Storey shear demand per 1 kN base shear
    shear_factor = np.array([pattern[i:].sum() for i in range(n)])

    V_y_base_by_story = Vy_story / np.maximum(shear_factor, 1e-12)
    V_global_y = float(np.min(V_y_base_by_story))
    critical = int(np.argmin(V_y_base_by_story)) + 1

    V_max = overstrength * V_global_y
    V = np.linspace(0, V_max, nsteps)

    roof_delta = []
    yielded_count = []

    for vb in V:
        story_shear = vb * shear_factor
        story_drift = np.zeros(n)
        ycount = 0

        for i in range(n):
            if story_shear[i] <= Vy_story[i]:
                story_drift[i] = story_shear[i] / k_story[i]
            else:
                ycount += 1
                dy = Vy_story[i] / k_story[i]
                post_k = max(post_ratio * k_story[i], 1e-9)
                story_drift[i] = dy + (story_shear[i] - Vy_story[i]) / post_k

        roof_delta.append(float(np.sum(story_drift)))
        yielded_count.append(ycount)

    push_df = pd.DataFrame({
        "Roof_displacement_m": roof_delta,
        "Base_shear_kN": V,
        "Yielded_storeys": yielded_count
    })

    return push_df, V_global_y, critical, shear_factor


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
input_method = st.sidebar.radio(
    "Stiffness input method",
    ["Storey stiffness vector", "Full STAAD K matrix"]
)

st.sidebar.divider()
run_analysis = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

st.sidebar.info(
    "The app now waits for **Run Analysis** before solving eigenvalues, RSA, pushover, and ADRS. "
    "This prevents Streamlit Cloud from recomputing everything on each input change."
)

# ---------------------------------------------------------
# Input stage
# ---------------------------------------------------------

st.header("1. STAAD-derived mass and stiffness input")

col1, col2 = st.columns(2)

with col1:
    mass_df = st.data_editor(
        default_mass(n),
        num_rows="fixed",
        use_container_width=True,
        key=f"mass_{n}"
    )

with col2:
    k_df = st.data_editor(
        default_k(n),
        num_rows="fixed",
        use_container_width=True,
        key=f"k_{n}"
    )

masses = mass_df["Mass_kNs2_per_m"].to_numpy(float)
k_story = k_df["Storey_stiffness_kN_per_m"].to_numpy(float)

M = np.diag(masses)
K_default = shear_building_K(k_story)

K_text = ""

if input_method == "Full STAAD K matrix":
    st.markdown(
        "Paste the condensed lateral stiffness matrix from STAAD, ordered from first floor DOF to roof DOF."
    )
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

st.header("2. Response spectrum input")

spec_df = st.data_editor(
    default_spectrum(),
    num_rows="dynamic",
    use_container_width=True,
    key="spectrum"
)

st.header("3. Beam/column plastic moment capacity and number of frames")

st.markdown(
    """
Input plastic moment sums **per one frame in the analyzed axis**.  
The app multiplies by the number of participating frames.
    """
)

cap_in = st.data_editor(
    default_capacity(n),
    num_rows="fixed",
    use_container_width=True,
    key=f"cap_{n}"
)

with st.expander("Capacity equation used"):
    st.latex(
        r"V_{y,i}=N_{frames,i}\left(C_c\frac{\sum M_{p,c}}{h_i}+C_b\frac{\sum M_{p,b}}{h_i}\right)"
    )
    st.markdown(
        """
Default `Column_factor = 2.0` represents top and bottom column plastic hinges in a storey mechanism.  
`Beam_factor` is included so you can calibrate whether beam hinges contribute directly to the storey mechanism.
        """
    )

st.header("4. Pushover controls")

colA, colB, colC, colD = st.columns(4)

with colA:
    overstrength = st.number_input(
        "Push up to this × first-yield base shear",
        1.00, 3.00, 1.50, 0.05
    )

with colB:
    post_ratio = st.number_input(
        "Post-yield stiffness ratio",
        0.001, 0.50, 0.05, 0.01
    )

with colC:
    target_roof_drift = st.number_input(
        "Target roof drift ratio",
        0.001, 0.10, 0.02, 0.001
    )

with colD:
    nsteps = st.slider(
        "Pushover steps",
        30, 200, 90, 10
    )

pattern_source = st.selectbox(
    "Pushover pattern source",
    ["Mode 1 mass × phi", "RSA SRSS floor force shape"],
    index=0
)

# ---------------------------------------------------------
# Stop here until user runs
# ---------------------------------------------------------

if not run_analysis:
    st.info("Set your inputs, then click **Run Analysis** in the sidebar.")
    st.stop()

# ---------------------------------------------------------
# Analysis stage
# ---------------------------------------------------------

try:
    with st.spinner("Running modal analysis, RSA, pushover, and ADRS reconciliation..."):
        T, w, phi = eig_analysis_cached(to_cache_tuple(M), to_cache_tuple(K))
        props, total_mass = modal_props(M, phi)

except Exception as e:
    st.error(f"Eigenvalue analysis failed. Check M and K inputs. Error: {e}")
    st.stop()

st.success("Analysis completed.")

# ---------------------------------------------------------
# Modal analysis and RSA
# ---------------------------------------------------------

st.header("5. Modal analysis and RSA")

c1, c2 = st.columns([1, 1])

with c1:
    modal_table = props.copy()
    modal_table.insert(1, "Period_sec", T[:len(modal_table)])
    st.dataframe(modal_table.style.format(precision=4), use_container_width=True)

with c2:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for j in range(min(num_modes, phi.shape[1])):
        ax.plot(
            phi[:, j],
            np.arange(1, n + 1),
            marker="o",
            label=f"Mode {j + 1}, T={T[j]:.3f}s"
        )

    ax.set_xlabel("Relative displacement, roof normalized = 1")
    ax.set_ylabel("Storey")
    ax.set_title("Mode shapes")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

Sa_modes = np.array([interp_sa(t, spec_df) for t in T[:num_modes]])

modal_forces = []
modal_base = []

for j in range(num_modes):
    p = phi[:, j]
    gamma = props.loc[j, "Gamma"]

    # F = m * phi * Gamma * Sa*g
    f = masses * p * gamma * Sa_modes[j] * G

    modal_forces.append(f)
    modal_base.append(np.sum(f))

modal_forces = np.array(modal_forces)
modal_base = np.array(modal_base)

rsa_floor_force_srss = np.sqrt(np.sum(modal_forces**2, axis=0))

rsa_storey_shear_srss = np.array([
    np.sqrt(np.sum([modal_forces[j, i:].sum()**2 for j in range(num_modes)]))
    for i in range(n)
])

rsa_base_shear_srss = float(np.sqrt(np.sum(modal_base**2)))

c1, c2 = st.columns(2)

with c1:
    rsa_df = pd.DataFrame({
        "Mode": np.arange(1, num_modes + 1),
        "T_sec": T[:num_modes],
        "Sa_g": Sa_modes,
        "Modal_base_shear_kN": modal_base,
        "Abs_base_shear_kN": np.abs(modal_base),
    })

    st.dataframe(rsa_df.style.format(precision=4), use_container_width=True)
    st.metric("SRSS RSA base shear", f"{rsa_base_shear_srss:,.1f} kN")

with c2:
    st.pyplot(
        plot_xy(
            spec_df["T_sec"],
            spec_df["Sa_g"],
            "Period T, sec",
            "Sa, g",
            "Input response spectrum"
        )
    )

# ---------------------------------------------------------
# First-mode pushover pattern
# ---------------------------------------------------------

st.header("6. First-mode pushover pattern")

if pattern_source.startswith("Mode 1"):
    raw_pattern = masses * phi[:, 0]
else:
    raw_pattern = rsa_floor_force_srss

raw_pattern = np.maximum(raw_pattern, 0)

if np.sum(raw_pattern) <= 0:
    raw_pattern = np.ones(n)

pattern = raw_pattern / np.sum(raw_pattern)

pattern_df = pd.DataFrame({
    "Storey": np.arange(1, n + 1),
    "Pattern_fraction": pattern,
    "Pattern_force_for_1kN_base_kN": pattern
})

st.dataframe(pattern_df.style.format(precision=5), use_container_width=True)

st.pyplot(
    plot_xy(
        pattern,
        np.arange(1, n + 1),
        "Lateral force fraction",
        "Storey",
        "Pushover lateral load pattern"
    )
)

# ---------------------------------------------------------
# Capacity
# ---------------------------------------------------------

st.header("7. Storey yield capacity from beam/column plastic moments")

cap = storey_capacity(cap_in)

st.dataframe(cap.style.format(precision=3), use_container_width=True)

# ---------------------------------------------------------
# Pushover
# ---------------------------------------------------------

st.header("8. Nonlinear pushover curve")

push_df, V_y_global, crit_storey, shear_factor = pushover_curve_cached(
    to_cache_vector(pattern),
    to_cache_vector(k_story),
    to_cache_vector(cap["Storey_yield_shear_kN"].to_numpy(float)),
    float(overstrength),
    float(post_ratio),
    int(nsteps)
)

H_total = float(cap["Storey_height_m"].sum())
push_df["Roof_drift_ratio"] = push_df["Roof_displacement_m"] / H_total

m1, m2, m3 = st.columns(3)

m1.metric("Estimated first-yield base shear", f"{V_y_global:,.1f} kN")
m2.metric("Critical first-yield storey", f"Storey {crit_storey}")
m3.metric("RSA / first-yield ratio", f"{rsa_base_shear_srss / V_y_global:.2f}")

st.pyplot(
    plot_xy(
        push_df["Roof_displacement_m"],
        push_df["Base_shear_kN"],
        "Roof displacement, m",
        "Base shear, kN",
        "Nonlinear pushover curve"
    )
)

# ---------------------------------------------------------
# ADRS
# ---------------------------------------------------------

st.header("9. ADRS capacity spectrum reconciliation")

Gamma1 = float(props.loc[0, "Gamma"])
Mstar1 = float(props.loc[0, "Effective_modal_mass"])
phi_roof1 = float(phi[-1, 0])

push_df["Sd_m"] = push_df["Roof_displacement_m"] / max(Gamma1 * phi_roof1, 1e-12)
push_df["Sa_g"] = push_df["Base_shear_kN"] / max(Mstar1 * G, 1e-12)

demand = spec_df.copy().sort_values("T_sec")
demand["Sd_m"] = demand["Sa_g"] * G * (demand["T_sec"] / (2 * np.pi)) ** 2

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.plot(
    push_df["Sd_m"],
    push_df["Sa_g"],
    marker="o",
    markevery=max(1, len(push_df) // 12),
    label="Capacity spectrum"
)
ax.plot(
    demand["Sd_m"],
    demand["Sa_g"],
    marker="s",
    label="Elastic demand spectrum"
)
ax.set_xlabel("Spectral displacement Sd, m")
ax.set_ylabel("Spectral acceleration Sa, g")
ax.set_title("ADRS reconciliation")
ax.grid(True, alpha=0.35)
ax.legend()
fig.tight_layout()

st.pyplot(fig)

if push_df["Sd_m"].max() > 0 and demand["Sd_m"].max() > 0:
    sd_grid = np.linspace(
        0,
        min(push_df["Sd_m"].max(), demand["Sd_m"].max()),
        300
    )

    cap_sa = np.interp(sd_grid, push_df["Sd_m"], push_df["Sa_g"])
    dem_sa = np.interp(sd_grid, demand["Sd_m"], demand["Sa_g"])

    idx = int(np.argmin(np.abs(cap_sa - dem_sa)))

    pp_sd = sd_grid[idx]
    pp_sa = cap_sa[idx]
    pp_roof = pp_sd * Gamma1 * phi_roof1

    st.success(
        f"Approximate ADRS performance point: "
        f"Sd = {pp_sd:.4f} m, "
        f"Sa = {pp_sa:.3f} g, "
        f"roof displacement ≈ {pp_roof:.4f} m, "
        f"drift ≈ {pp_roof / H_total:.3%}"
    )

# ---------------------------------------------------------
# Reconciliation summary
# ---------------------------------------------------------

st.header("10. RSA versus pushover reconciliation summary")

recon = pd.DataFrame({
    "Item": [
        "Total seismic mass",
        "Mode 1 period",
        "Mode 1 mass participation",
        "Cumulative participation used",
        "RSA SRSS base shear",
        "Pushover first-yield base shear",
        "RSA / yield capacity",
        "Target roof displacement"
    ],
    "Value": [
        f"{total_mass:,.2f} kN·s²/m",
        f"{T[0]:.4f} sec",
        f"{props.loc[0, 'Mass_participation_%']:.2f}%",
        f"{props.loc[num_modes - 1, 'Cumulative_mass_%']:.2f}%",
        f"{rsa_base_shear_srss:,.1f} kN",
        f"{V_y_global:,.1f} kN",
        f"{rsa_base_shear_srss / V_y_global:.2f}",
        f"{target_roof_drift * H_total:.4f} m"
    ]
})

st.table(recon)

# ---------------------------------------------------------
# Downloads
# ---------------------------------------------------------

out = io.BytesIO()

with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    mass_df.to_excel(writer, sheet_name="Mass", index=False)
    k_df.to_excel(writer, sheet_name="Storey stiffness", index=False)
    pd.DataFrame(K).to_excel(writer, sheet_name="K matrix", index=False)
    modal_table.to_excel(writer, sheet_name="Modal", index=False)
    rsa_df.to_excel(writer, sheet_name="RSA", index=False)
    pattern_df.to_excel(writer, sheet_name="Pattern", index=False)
    cap.to_excel(writer, sheet_name="Plastic capacity", index=False)
    push_df.to_excel(writer, sheet_name="Pushover_ADRS", index=False)
    recon.to_excel(writer, sheet_name="Summary", index=False)

st.download_button(
    "Download calculation workbook",
    data=out.getvalue(),
    file_name="mdof_pushover_reconciliation.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.warning(
    "Use this as a transparent reconciliation and teaching app only. "
    "For design approval, verify with code-compliant nonlinear static procedures and validated structural software."
)
