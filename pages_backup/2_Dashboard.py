import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Tableau de Bord — SihaIQ", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1D3557; }
        [data-testid="stSidebar"] * { color: white !important; }
        .kpi-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #1D3557;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
    padding: 20px 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin:0;'>📊 Tableau de Bord Opérationnel</h1>
        <p style='color: #A8DADC; margin:5px 0 0 0;'>
        Analyse des rejets • Données synthétiques BAF • 3,000 dossiers</p>
    </div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/sihaiq_synthetic_baf.csv")
    return df

df = load_data()

# KPI Cards
st.markdown("### 📈 Indicateurs Clés de Performance")
col1, col2, col3, col4 = st.columns(4)

total = len(df)
rejected = df['rejected'].sum()
rejection_rate = rejected / total * 100
clean_rate = 100 - rejection_rate
top_cause = df[df['rejected']==1]['rejection_cause'].value_counts().index[0]

with col1:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size:2rem; font-weight:900; color:#1D3557;'>{total:,}</div>
            <div style='color:#666; margin-top:5px;'>Total Dossiers</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size:2rem; font-weight:900; color:#E63946;'>
            {rejection_rate:.1f}%</div>
            <div style='color:#666; margin-top:5px;'>Taux de Rejet</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size:2rem; font-weight:900; color:#2A9D8F;'>
            {clean_rate:.1f}%</div>
            <div style='color:#666; margin-top:5px;'>Taux de Dossiers Propres</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-size:1.1rem; font-weight:900; color:#E63946;'>
            {top_cause.replace('_',' ')}</div>
            <div style='color:#666; margin-top:5px;'>Cause Principale de Rejet</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# Row 2: Pareto + Payer breakdown
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📉 Analyse Pareto — Causes de Rejet")
    cause_counts = df[df['rejected']==1]['rejection_cause'].value_counts().reset_index()
    cause_counts.columns = ['Cause', 'Nombre']
    cause_counts['Cause'] = cause_counts['Cause'].str.replace('_', ' ')
    cause_counts['Pourcentage'] = cause_counts['Nombre'] / cause_counts['Nombre'].sum() * 100
    cause_counts['Cumulatif'] = cause_counts['Pourcentage'].cumsum()

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=cause_counts['Cause'],
        y=cause_counts['Pourcentage'],
        name='% Rejets',
        marker_color='#E63946',
        text=cause_counts['Pourcentage'].round(1).astype(str) + '%',
        textposition='outside'
    ))
    fig_pareto.add_trace(go.Scatter(
        x=cause_counts['Cause'],
        y=cause_counts['Cumulatif'],
        name='% Cumulatif',
        yaxis='y2',
        line=dict(color='#1D3557', width=2),
        marker=dict(size=6)
    ))
    fig_pareto.update_layout(
        yaxis=dict(title='% des Rejets'),
        yaxis2=dict(title='% Cumulatif', overlaying='y', side='right', range=[0,110]),
        legend=dict(orientation='h', y=-0.2),
        plot_bgcolor='white',
        height=400,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

with col_right:
    st.markdown("#### 🏥 Taux de Rejet par Organisme Payeur")
    payer_map = {0: 'AMO', 1: 'CNOPS', 2: 'CNSS'}
    df['payer_label'] = df['payer'].map(payer_map)
    payer_stats = df.groupby('payer_label')['rejected'].mean().reset_index()
    payer_stats.columns = ['Organisme', 'Taux de Rejet']
    payer_stats['Taux de Rejet'] = payer_stats['Taux de Rejet'] * 100
    payer_stats['Couleur'] = payer_stats['Taux de Rejet'].apply(
        lambda x: '#E63946' if x > 40 else '#F4A261' if x > 35 else '#2A9D8F'
    )

    fig_payer = go.Figure(go.Bar(
        x=payer_stats['Organisme'],
        y=payer_stats['Taux de Rejet'],
        marker_color=payer_stats['Couleur'],
        text=payer_stats['Taux de Rejet'].round(1).astype(str) + '%',
        textposition='outside'
    ))
    fig_payer.update_layout(
        yaxis=dict(title='Taux de Rejet (%)', range=[0, 60]),
        plot_bgcolor='white',
        height=400,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_payer, use_container_width=True)

st.markdown("---")

# Row 3: Feature impact
st.markdown("#### ⚙️ Impact des Features sur le Taux de Rejet")

binary_features = {
    'immatriculation_valid': 'Immatriculation Valide',
    'ngap_coding_valid': 'Codage NGAP Valide',
    'cin_valid': 'CIN Valide',
    'prescription_legible': 'Ordonnance Lisible',
    'droits_active': 'Droits AMO Actifs',
    'inpe_present': 'INPE Présent',
    'pec_obtained': 'PEC Obtenue'
}

impact_data = []
for col, label in binary_features.items():
    rate_0 = df[df[col]==0]['rejected'].mean() * 100
    rate_1 = df[df[col]==1]['rejected'].mean() * 100
    impact_data.append({'Feature': label, 'Absent (0)': rate_0, 'Présent (1)': rate_1})

impact_df = pd.DataFrame(impact_data)

fig_impact = go.Figure()
fig_impact.add_trace(go.Bar(
    name='Absent / Invalide',
    x=impact_df['Feature'],
    y=impact_df['Absent (0)'],
    marker_color='#E63946',
    text=impact_df['Absent (0)'].round(1).astype(str) + '%',
    textposition='outside'
))
fig_impact.add_trace(go.Bar(
    name='Présent / Valide',
    x=impact_df['Feature'],
    y=impact_df['Présent (1)'],
    marker_color='#2A9D8F',
    text=impact_df['Présent (1)'].round(1).astype(str) + '%',
    textposition='outside'
))
fig_impact.update_layout(
    barmode='group',
    yaxis=dict(title='Taux de Rejet (%)', range=[0, 100]),
    legend=dict(orientation='h', y=-0.2),
    plot_bgcolor='white',
    height=420,
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig_impact, use_container_width=True)