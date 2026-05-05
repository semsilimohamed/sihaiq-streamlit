import streamlit as st

st.set_page_config(
    page_title="SihaIQ — Intelligence du Cycle de Revenus",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state init
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1D3557; }
        [data-testid="stSidebar"] * { color: white !important; }
        .login-container {
            max-width: 420px;
            margin: 80px auto;
            padding: 40px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }
    </style>
""", unsafe_allow_html=True)

# LOGIN GATE
if not st.session_state.authenticated:

    # Hide sidebar when not logged in
    st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align:center; margin-top:60px;'>
            <div style='font-size:3rem;'>⚕</div>
            <h1 style='color:#1D3557; font-size:2rem; margin:10px 0 5px 0;'>SihaIQ</h1>
            <p style='color:#457B9D; font-size:1rem; margin:0;'>
            Intelligence du Cycle de Revenus</p>
            <p style='color:#888; font-size:0.85rem; margin-top:5px;'>
            Plateforme IA • CNOPS • CNSS • AMO</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔐 Connexion")

        username = st.text_input("Identifiant", placeholder="Entrez votre identifiant")
        password = st.text_input("Mot de passe", type="password", 
                                  placeholder="Entrez votre mot de passe")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Se connecter", type="primary", use_container_width=True):
            # Credentials
            USERS = {
                "admin": "sihaiq2026",
                "demo": "cm6ri2026",
                "jury": "innovation2026"
            }
            if username in USERS and password == USERS[username]:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Identifiant ou mot de passe incorrect")

        st.markdown("""
            <div style='text-align:center; margin-top:20px; color:#888; font-size:0.8rem;'>
                Accès réservé aux établissements partenaires<br>
                Contact: contact@sihaiq.ma
            </div>
        """, unsafe_allow_html=True)

    st.stop()

# AUTHENTICATED — show full app
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1D3557; }
        [data-testid="stSidebar"] * { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
    <div style='background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
    padding: 20px 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin:0;'>⚕ SihaIQ — Intelligence du Cycle de Revenus</h1>
        <p style='color: #A8DADC; margin:5px 0 0 0;'>
        Plateforme IA de prédiction et récupération des rejets d'assurance 
        • CNOPS • CNSS • AMO</p>
    </div>
""", unsafe_allow_html=True)

# Logout button
col_title, col_logout = st.columns([5, 1])
with col_logout:
    if st.button("🚪 Déconnexion"):
        st.session_state.authenticated = False
        st.rerun()

# Welcome
st.markdown(f"### Bienvenue, **{st.session_state.username}** 👋")
st.markdown("Sélectionnez un module dans la barre latérale gauche.")

st.markdown("<br>", unsafe_allow_html=True)

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

col4, col5, _ = st.columns(3)

with col4:
    st.info("""
    ### 💰 Simulateur ROI
    Calculez le montant récupérable avec SihaIQ
    selon votre volume et taux de rejet.
    → **Page : Simulateur**
    """)

with col5:
    st.info("""
    ### 📦 Scoring par Lot
    Uploadez un CSV de dossiers → SihaIQ
    score chaque dossier en masse.
    → **Page : Batch Scorer**
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    SihaIQ v1.0 • Données synthétiques CNDP-conformes • 
    Soumis au CM6RI Lab d'Innovation 2026
</div>
""", unsafe_allow_html=True)