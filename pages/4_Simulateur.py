import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Simulateur ROI — SihaIQ", page_icon="💰", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1D3557; }
        [data-testid="stSidebar"] * { color: white !important; }
        .result-card {
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            color: white;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
    padding: 20px 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin:0;'>💰 Simulateur de Récupération de Revenus</h1>
        <p style='color: #A8DADC; margin:5px 0 0 0;'>
        Calculez le montant récupérable avec SihaIQ — en temps réel</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### ⚙️ Paramètres de votre Établissement")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Volume & Finances**")
    nb_dossiers_mois = st.slider(
        "Nombre de dossiers soumis par mois", 
        100, 5000, 800, step=50)
    
    montant_moyen_mad = st.slider(
        "Montant moyen par dossier (MAD)", 
        500, 15000, 3200, step=100)
    
    taux_rejet_actuel = st.slider(
        "Taux de rejet actuel (%)", 
        5, 70, 38, step=1)

with col2:
    st.markdown("**Performance SihaIQ**")
    taux_rejet_cible = st.slider(
        "Taux de rejet cible avec SihaIQ (%)",
        5, 50, 18, step=1)
    
    taux_recuperation = st.slider(
        "Taux de récupération sur rejets détectés (%)",
        30, 95, 72, step=1)
    
    abonnement_mad = st.selectbox(
        "Plan SihaIQ (MAD/mois)",
        options=[2000, 5000, 8000, 12000],
        format_func=lambda x: {
            2000: "Starter — 2,000 MAD/mois",
            5000: "Professionnel — 5,000 MAD/mois",
            8000: "Clinique — 8,000 MAD/mois",
            12000: "Entreprise — 12,000 MAD/mois"
        }[x]
    )

st.markdown("---")

# Calculations
volume_mad_mois = nb_dossiers_mois * montant_moyen_mad
rejets_actuels = nb_dossiers_mois * (taux_rejet_actuel / 100)
mad_perdu_mois = rejets_actuels * montant_moyen_mad

rejets_cible = nb_dossiers_mois * (taux_rejet_cible / 100)
rejets_evites = rejets_actuels - rejets_cible
mad_recupere_mois = rejets_evites * montant_moyen_mad * (taux_recuperation / 100)

# Performance fee (4% of recovered)
performance_fee = mad_recupere_mois * 0.04
cout_total_mois = abonnement_mad + performance_fee
roi_mois = mad_recupere_mois - cout_total_mois
roi_annuel = roi_mois * 12
payback_days = (cout_total_mois / mad_recupere_mois * 30) if mad_recupere_mois > 0 else 0
st.markdown("### 📊 Résultats de la Simulation")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
        <div class='result-card' style='background: #E63946;'>
            <div style='font-size:0.85rem; opacity:0.9;'>Pertes Actuelles / Mois</div>
            <div style='font-size:1.8rem; font-weight:900; margin:8px 0;'>
            {mad_perdu_mois:,.0f} MAD</div>
            <div style='font-size:0.8rem; opacity:0.85;'>
            {rejets_actuels:.0f} dossiers rejetés</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class='result-card' style='background: #2A9D8F;'>
            <div style='font-size:0.85rem; opacity:0.9;'>Récupération / Mois</div>
            <div style='font-size:1.8rem; font-weight:900; margin:8px 0;'>
            {mad_recupere_mois:,.0f} MAD</div>
            <div style='font-size:0.8rem; opacity:0.85;'>
            {rejets_evites:.0f} rejets évités</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class='result-card' style='background: #1D3557;'>
            <div style='font-size:0.85rem; opacity:0.9;'>ROI Net Annuel</div>
            <div style='font-size:1.8rem; font-weight:900; margin:8px 0;'>
            {roi_annuel:,.0f} MAD</div>
            <div style='font-size:0.8rem; opacity:0.85;'>
            Après abonnement + commission</div>
        </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
        <div class='result-card' style='background: #457B9D;'>
            <div style='font-size:0.85rem; opacity:0.9;'>Retour sur Investissement</div>
            <div style='font-size:1.8rem; font-weight:900; margin:8px 0;'>
            {payback_days:.0f} jours</div>
            <div style='font-size:0.8rem; opacity:0.85;'>Délai de rentabilisation</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Monthly projection chart
st.markdown("#### 📈 Projection sur 12 Mois")

mois = [f"M{i}" for i in range(1, 13)]
pertes = [mad_perdu_mois] * 12
recuperation_cumul = [mad_recupere_mois * i for i in range(1, 13)]
cout_cumul = [cout_total_mois * i for i in range(1, 13)]
roi_cumul = [r - c for r, c in zip(recuperation_cumul, cout_cumul)]

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Pertes sans SihaIQ (MAD)',
    x=mois,
    y=pertes,
    marker_color='#E63946',
    opacity=0.7
))

fig.add_trace(go.Scatter(
    name='Récupération Cumulée (MAD)',
    x=mois,
    y=recuperation_cumul,
    mode='lines+markers',
    line=dict(color='#2A9D8F', width=3),
    marker=dict(size=8)
))

fig.add_trace(go.Scatter(
    name='ROI Net Cumulé (MAD)',
    x=mois,
    y=roi_cumul,
    mode='lines+markers',
    line=dict(color='#1D3557', width=3, dash='dash'),
    marker=dict(size=8)
))

fig.update_layout(
    yaxis=dict(title='Montant (MAD)'),
    legend=dict(orientation='h', y=-0.2),
    plot_bgcolor='white',
    height=400,
    margin=dict(t=20, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Cost breakdown
st.markdown("#### 💼 Détail des Coûts Mensuels SihaIQ")

col_cost, col_summary = st.columns(2)

with col_cost:
    fig_cost = go.Figure(go.Pie(
        labels=['Abonnement fixe', 'Commission performance (4%)'],
        values=[abonnement_mad, performance_fee],
        hole=0.5,
        marker_colors=['#1D3557', '#457B9D']
    ))
    fig_cost.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_cost, use_container_width=True)

with col_summary:
    st.markdown(f"""
    <div style='background:#f8f9fa; padding:25px; border-radius:12px; margin-top:20px;'>
        <h4 style='color:#1D3557; margin-top:0;'>📋 Résumé Financier</h4>
        <table width='100%'>
            <tr><td>Volume mensuel total</td>
                <td align='right'><b>{volume_mad_mois:,.0f} MAD</b></td></tr>
            <tr><td>Pertes actuelles/mois</td>
                <td align='right' style='color:#E63946;'>
                <b>-{mad_perdu_mois:,.0f} MAD</b></td></tr>
            <tr><td>Récupération estimée/mois</td>
                <td align='right' style='color:#2A9D8F;'>
                <b>+{mad_recupere_mois:,.0f} MAD</b></td></tr>
            <tr><td>Abonnement SihaIQ</td>
                <td align='right'><b>-{abonnement_mad:,.0f} MAD</b></td></tr>
            <tr><td>Commission performance</td>
                <td align='right'><b>-{performance_fee:,.0f} MAD</b></td></tr>
            <tr style='border-top:2px solid #1D3557;'>
                <td><b>ROI Net / Mois</b></td>
                <td align='right' style='color:#2A9D8F;'>
                <b>+{roi_mois:,.0f} MAD</b></td></tr>
            <tr><td><b>ROI Net Annuel</b></td>
                <td align='right' style='color:#1D3557; font-size:1.1rem;'>
                <b>+{roi_annuel:,.0f} MAD</b></td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    Simulation basée sur les données du marché marocain • 
    CNOPS/CNSS/AMO • Taux de rejet moyen secteur privé: 35-42%
</div>
""", unsafe_allow_html=True)