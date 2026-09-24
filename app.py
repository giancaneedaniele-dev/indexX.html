import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go

# Configurazione della pagina
st.set_page_config(
    page_title="Risolutore di Programmazione Lineare",
    page_icon="📐",
    layout="wide"
)

st.title("📐 Risolutore di Programmazione Lineare Continua (PLC)")
st.markdown("""
Questa applicazione consente di impostare e risolvere problemi di Programmazione Lineare continua 
utilizzando il modulo `scipy.optimize.linprog`.  
Se selezioni **2 variabili decisionali ($x_1$ e $x_2$)**, verrà mostrato anche il **grafico 2D interattivo** 
con regione ammissibile, rette dei vincoli, curva di livello della funzione obiettivo e punto ottimo!
""")

# ==========================================
# 1. Configurazione Generale (Sidebar)
# ==========================================
st.sidebar.header("⚙️ Impostazioni del Problema")

obj_type = st.sidebar.radio("Tipo di Ottimizzazione:", ["Massimizzazione (MAX)", "Minimizzazione (MIN)"])
num_vars = st.sidebar.number_input("Numero di variabili decisionali (n ≥ 2):", min_value=2, max_value=20, value=2, step=1)
non_negative = st.sidebar.checkbox("Vincoli di non negatività (x_i ≥ 0)", value=True)

# Inizializzazione dello stato per il conteggio dei vincoli
if 'num_constraints' not in st.session_state:
    st.session_state.num_constraints = 2

col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    if st.button("➕ Aggiungi Vincolo"):
        st.session_state.num_constraints += 1
        st.rerun()
with col_btn2:
    if st.button("➖ Rimuovi Vincolo"):
        if st.session_state.num_constraints > 1:
            st.session_state.num_constraints -= 1
            st.rerun()

st.sidebar.markdown(f"**Vincoli correnti:** {st.session_state.num_constraints}")

# ==========================================
# 2. Funzione Obiettivo
# ==========================================
st.subheader("🎯 1. Funzione Obiettivo")
st.caption("Inserisci i coefficienti $c_i$ per ciascuna variabile decisionale.")

c_coeffs = []
cols_obj = st.columns(num_vars)
for i in range(num_vars):
    with cols_obj[i]:
        val = st.number_input(f"Coeff. $x_{i+1}$", value=1.0 if i == 0 else 2.0, step=0.5, format="%.2f", key=f"c_{i}")
        c_coeffs.append(val)

# Rappresentazione simbolica FO
obj_str = " + ".join([f"({c_coeffs[i]})·x_{i+1}" for i in range(num_vars)])
st.latex(f"\\text{{{ 'Max' if 'MAX' in obj_type else 'Min' }}} \\quad Z = {obj_str}")

# ==========================================
# 3. Vincoli
# ==========================================
st.subheader("📋 2. Vincoli Tecnologici")

constraints_data = []

for row in range(st.session_state.num_constraints):
    with st.container():
        cols = st.columns([1.5] * num_vars + [1.2, 1.5])
        
        row_coeffs = []
        for i in range(num_vars):
            with cols[i]:
                default_val = 1.0 if (row == 0 and i == 0) or (row == 1 and i == 1) else 0.0
                coeff = st.number_input(f"$x_{i+1}$ (V.{row+1})", value=float(default_val), step=0.5, format="%.2f", key=f"a_{row}_{i}")
                row_coeffs.append(coeff)
        
        with cols[num_vars]:
            sense = st.selectbox("Verso", ["<=", ">=", "="], key=f"sense_{row}")
        
        with cols[num_vars + 1]:
            rhs = st.number_input(f"Termine Noto (b_{row+1})", value=4.0 if row == 0 else 6.0, step=1.0, format="%.2f", key=f"b_{row}")
        
        constraints_data.append({
            "coeffs": row_coeffs,
            "sense": sense,
            "rhs": rhs
        })

# ==========================================
# Funzione Ausiliaria: Grafico 2D (Plotly)
# ==========================================
def plot_2d_problem(c_coeffs, constraints_data, x_opt, z_opt, non_negative, obj_type):
    # Determiniamo un intervallo di visualizzazione intelligente
    x1_ref = max(abs(x_opt[0]), 5.0)
    x2_ref = max(abs(x_opt[1]), 5.0)
    max_range = max(x1_ref, x2_ref) * 1.6

    x1_min = 0.0 if non_negative else -max_range * 0.2
    x1_max = max_range
    x2_min = 0.0 if non_negative else -max_range * 0.2
    x2_max = max_range

    # Griglia per calcolare la regione ammissibile
    grid_n = 350
    X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, grid_n), np.linspace(x2_min, x2_max, grid_n))
    feasible_mask = np.ones(X1.shape, dtype=bool)

    if non_negative:
        feasible_mask &= (X1 >= -1e-6) & (X2 >= -1e-6)

    for c in constraints_data:
        a1, a2 = c["coeffs"][0], c["coeffs"][1]
        b = c["rhs"]
        lhs = a1 * X1 + a2 * X2

        if c["sense"] == "<=":
            feasible_mask &= (lhs <= b + 1e-6)
        elif c["sense"] == ">=":
            feasible_mask &= (lhs >= b - 1e-6)
        elif c["sense"] == "=":
            # Per l'uguaglianza, una fascia sottile per evidenziarla
            feasible_mask &= (np.abs(lhs - b) <= 0.05 * (np.abs(b) + 1.0))

    fig = go.Figure()

    # 1. Regione Ammissibile (Contour/Heatmap discreta)
    fig.add_trace(go.Contour(
        x=np.linspace(x1_min, x1_max, grid_n),
        y=np.linspace(x2_min, x2_max, grid_n),
        z=feasible_mask.astype(int),
        showscale=False,
        contours=dict(start=0.5, end=1.5, size=1),
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0, 200, 100, 0.28)']],
        hoverinfo='skip',
        name='Regione Ammissibile'
    ))

    # 2. Rette dei Vincoli
    x_vals = np.linspace(x1_min, x1_max, 400)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    for i, c in enumerate(constraints_data):
        a1, a2 = c["coeffs"][0], c["coeffs"][1]
        b = c["rhs"]
        color = colors[i % len(colors)]
        label = f"V{i+1}: {a1}x₁ + {a2}x₂ {c['sense']} {b}"

        if abs(a2) > 1e-6:
            y_vals = (b - a1 * x_vals) / a2
            # Filtro per mantenere i valori entro una finestra ragionevole
            mask = (y_vals >= x2_min - 2) & (y_vals <= x2_max + 2)
            fig.add_trace(go.Scatter(
                x=x_vals[mask], y=y_vals[mask],
                mode='lines',
                name=label,
                line=dict(color=color, width=2.2)
            ))
        elif abs(a1) > 1e-6:
            # Vincolo verticale (x1 = costante)
            x_line = b / a1
            fig.add_trace(go.Scatter(
                x=[x_line, x_line], y=[x2_min, x2_max],
                mode='lines',
                name=label,
                line=dict(color=color, width=2.2)
            ))

    # 3. Retta della Funzione Obiettivo passante per la soluzione ottima
    c1, c2 = c_coeffs[0], c_coeffs[1]
    if abs(c2) > 1e-6:
        y_obj = (z_opt - c1 * x_vals) / c2
        mask_obj = (y_obj >= x2_min - 2) & (y_obj <= x2_max + 2)
        fig.add_trace(go.Scatter(
            x=x_vals[mask_obj], y=y_obj[mask_obj],
            mode='lines',
            name=f"F.O. Ottima (Z* = {z_opt:.2f})",
            line=dict(color='purple', width=2.5, dash='dash')
        ))
    elif abs(c1) > 1e-6:
        x_obj = z_opt / c1
        fig.add_trace(go.Scatter(
            x=[x_obj, x_obj], y=[x2_min, x2_max],
            mode='lines',
            name=f"F.O. Ottima (Z* = {z_opt:.2f})",
            line=dict(color='purple', width=2.5, dash='dash')
        ))

    # 4. Soluzione Ottima (Punto Evidenziato)
    fig.add_trace(go.Scatter(
        x=[x_opt[0]], y=[x_opt[1]],
        mode='markers+text',
        name='Soluzione Ottima (x*)',
        text=[f" Ottimo ({x_opt[0]:.2f}, {x_opt[1]:.2f})"],
        textposition="top right",
        marker=dict(size=14, color='red', symbol='star', line=dict(color='black', width=1.5)),
        hovertemplate=f"<b>Soluzione Ottima</b><br>x₁: {x_opt[0]:.4f}<br>x₂: {x_opt[1]:.4f}<br>Z: {z_opt:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Rappresentazione Geometrica 2D</b>", x=0.5),
        xaxis=dict(title="x₁", range=[x1_min, x1_max], zeroline=True, zerolinecolor="black"),
        yaxis=dict(title="x₂", range=[x2_min, x2_max], zeroline=True, zerolinecolor="black"),
        legend=dict(x=1.02, y=1, orientation="v", bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
        width=850,
        height=620
    )
    return fig

# ==========================================
# 4. Risoluzione con SciPy
# ==========================================
st.markdown("---")
if st.button("🚀 Risolvi Problema Lineare", type="primary"):
    
    # scipy.optimize.linprog minimizza c^T * x.
    # Per massimizzare, invertiamo il segno: min (-c)^T * x
    c_vector = np.array(c_coeffs, dtype=float)
    if "MAX" in obj_type:
        c_linprog = -c_vector
    else:
        c_linprog = c_vector

    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []

    for c in constraints_data:
        coeffs = c["coeffs"]
        sense = c["sense"]
        rhs = c["rhs"]

        if sense == "<=":
            A_ub.append(coeffs)
            b_ub.append(rhs)
        elif sense == ">=":
            A_ub.append([-x for x in coeffs])
            b_ub.append(-rhs)
        elif sense == "=":
            A_eq.append(coeffs)
            b_eq.append(rhs)

    # Limiti delle variabili
    bounds = (0, None) if non_negative else (None, None)
    bounds_list = [bounds for _ in range(num_vars)]

    res = linprog(
        c=c_linprog,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds_list,
        method="highs"
    )

    # ==========================================
    # 5. Output dei Risultati
    # ==========================================
    if res.success:
        st.success("✅ **Soluzione Ottima Trovata!**")
        
        # Calcolo del valore ottimo reale di Z
        valore_ottimo = -res.fun if "MAX" in obj_type else res.fun
        x_opt = res.x

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric(label="Valore Ottimo Z*", value=f"{valore_ottimo:.4f}")
        
        # Valori Ottimi delle Variabili
        st.markdown("### 📌 Valori Ottimi delle Variabili:")
        df_vars = pd.DataFrame({
            "Variabile": [f"x_{i+1}" for i in range(num_vars)],
            "Valore Ottimo": [round(val, 4) for val in x_opt]
        })
        st.dataframe(df_vars, use_container_width=True)

        # Analisi Saturazione e Slack dei Vincoli
        st.markdown("### 📊 Analisi di Saturazione dei Vincoli")
        
        vincoli_report = []
        toll = 1e-6  # Tolleranza numerica per identificare vincoli attivi

        for idx, c in enumerate(constraints_data):
            lhs_val = np.dot(c["coeffs"], x_opt)
            rhs_val = c["rhs"]
            sense = c["sense"]
            
            # Calcolo Slack / Surplus
            if sense == "<=":
                slack = rhs_val - lhs_val
                is_active = abs(slack) < toll
                slack_str = f"{slack:.4f} (Scarto/Slack)"
            elif sense == ">=":
                surplus = lhs_val - rhs_val
                is_active = abs(surplus) < toll
                slack_str = f"{surplus:.4f} (Surplus)"
            else:  # Vincolo '='
                diff = abs(lhs_val - rhs_val)
                is_active = diff < toll
                slack_str = f"{0.0:.4f} (Uguaglianza)"

            stato = "🟢 Attivo (Saturato)" if is_active else "⚪ Non Attivo (Lassità)"

            vincoli_report.append({
                "Vincolo": f"Vincolo {idx + 1}",
                "Primo Membro (LHS)": round(lhs_val, 4),
                "Verso": sense,
                "Secondo Membro (RHS)": rhs_val,
                "Slack / Surplus": slack_str,
                "Stato": stato
            })

        df_constraints = pd.DataFrame(vincoli_report)
        st.dataframe(df_constraints, use_container_width=True)

        # ==========================================
        # 6. Grafico 2D (Solo per n = 2)
        # ==========================================
        if num_vars == 2:
            st.markdown("---")
            st.markdown("### 📈 Visualizzazione Grafica 2D")
            fig = plot_2d_problem(c_coeffs, constraints_data, x_opt, valore_ottimo, non_negative, obj_type)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Il grafico 2D viene generato automaticamente solo quando ci sono esattamente 2 variabili decisionali.")

    else:
        st.error(f"❌ Impossibile trovare una soluzione ottima: **{res.message}**")
