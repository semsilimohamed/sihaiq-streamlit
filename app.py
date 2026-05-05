import streamlit as st

st.set_page_config(
    page_title="SihaIQ — Intelligence du Cycle de Revenus",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #1D3557;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        .main-header {
            background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
            padding: 20px 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .main-header h1 {
            color: white !important;
            font-size: 2rem;
            margin: 0;
        }
        .main-header p {
            color: #A8DADC !important;
            margin: 5px 0 0 0;
            font-size: 1rem;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #1D3557;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>⚕ SihaIQ — Intelligence du Cycle de Revenus</h1>
        <p>Plateforme IA de prédiction et récupération des rejets d'assurance </p>
    </div>
""", unsafe_allow_html=True)

# Home page content
col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    ### 🎯 Prédiction de Rejet
    Scorez un dossier en temps réel avant soumission.
    Identifiez les causes de rejet avec SHAP.
    
    → **Page : Prédiction**
    """)

with col2:
    st.info("""
    ### 📊 Tableau de Bord
    KPIs opérationnels, analyse Pareto des rejets,
    performance par organisme payeur.
    
    → **Page : Tableau de Bord**
    """)

with col3:
    st.info("""
    ### 🔬 Performance Modèle
    AUC, courbe ROC, matrice de confusion,
    importance globale des features SHAP.
    
    → **Page : Performance Modèle**
    """)

st.markdown("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    SihaIQ v1.0 • Données synthétiques CNDP-conformes • 
    Soumis au CM6RI Lab d'Innovation 2026
</div>
""", unsafe_allow_html=True)