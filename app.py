import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io

st.set_page_config(
    page_title="SihaIQ — Intelligence du Cycle de Revenus",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        [data-testid="collapsedControl"] {display: none;}
        .block-container {
            padding-top: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #1D3557;
            padding: 8px 16px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            color: white !important;
            background-color: transparent;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #457B9D !important;
            color: white !important;
        }
        .main-header {
            background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
            padding: 15px 25px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# LOGIN GATE
if not st.session_state.authenticated:
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

# HEADER
col_h, col_logout = st.columns([5, 1])
with col_h:
    st.markdown("""
        <div class='main-header'>
            <h1 style='color:white; margin:0; font-size:1.5rem;'>
            ⚕ SihaIQ — Intelligence du Cycle de Revenus</h1>
            <p style='color:#A8DADC; margin:3px 0 0 0; font-size:0.85rem;'>
            Exploitez vos données pour votre futur.</p>
        </div>
    """, unsafe_allow_html=True)
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Déconnexion", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# LOAD MODEL AND DATA
@st.cache_resource
def load_model():
    model = joblib.load("model/sihaiq_xgboost_model.pkl")
    features = joblib.load("model/sihaiq_feature_names.pkl")
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv("data/sihaiq_synthetic_baf.csv")

model, feature_names = load_model()
df = load_data()

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prédiction",
    "📊 Tableau de Bord",
    "🔬 Performance Modèle",
    "💰 Simulateur ROI",
    "📦 Scoring par Lot"
])

# ─────────────────────────────────────────
# TAB 1 — PREDICTION
# ─────────────────────────────────────────
with tab1:
    st.markdown("### 📋 Saisie du Dossier BAF")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informations Payeur**")
        payer = st.selectbox("Organisme Payeur",
            options=[0,1,2],
            format_func=lambda x: {0:"AMO",1:"CNOPS",2:"CNSS"}[x])
        service_type = st.selectbox("Type de Prestation",
            options=[0,1,2,3,4,5],
            format_func=lambda x: {0:"Biologie",1:"Chirurgie",2:"Consultation",
                                    3:"Hospitalisation",4:"Imagerie",5:"Urgences"}[x])
        ngap_code = st.selectbox("Code NGAP",
            options=[0,1,2,3,4,5,6,7,8,9],
            format_func=lambda x: {0:"B",1:"C",2:"C2",3:"Cs",4:"FORFAIT",
                                    5:"HN",6:"K",7:"P",8:"V",9:"Z"}[x])
        num_acts = st.slider("Nombre d'Actes", 1, 20, 3)
        patient_age = st.slider("Âge du Patient", 0, 100, 35)

    with col2:
        st.markdown("**Statut Patient**")
        is_ald = st.toggle("Affection Longue Durée (ALD)", value=False)
        is_ayant_droit = st.toggle("Ayant Droit", value=False)
        st.markdown("**Identito-Vigilance**")
        inpe_present = st.toggle("INPE Présent", value=True)
        immatriculation_valid = st.toggle("Immatriculation Valide", value=True)
        cin_valid = st.toggle("CIN Valide", value=True)

    with col3:
        st.markdown("**Qualité du Dossier**")
        ngap_coding_valid = st.toggle("Codage NGAP Valide", value=True)
        prescription_legible = st.toggle("Ordonnance Lisible", value=True)
        droits_active = st.toggle("Droits AMO Actifs", value=True)
        pec_required = st.toggle("PEC Requise", value=False)
        pec_obtained = st.toggle("PEC Obtenue", value=False)
        docs_completeness_ratio = st.slider("Ratio Complétude Dossier", 0.0, 1.0, 0.85)
        days_since_service = st.slider("Jours depuis Prestation", 0, 90, 15)

    if st.button("🔍 Analyser le Dossier", type="primary", use_container_width=True):
        input_data = np.array([[
            payer, service_type, ngap_code, num_acts, patient_age,
            int(is_ald), int(is_ayant_droit), int(inpe_present),
            int(immatriculation_valid), int(cin_valid), int(ngap_coding_valid),
            int(prescription_legible), int(droits_active),
            docs_completeness_ratio, days_since_service,
            int(pec_required), int(pec_obtained)
        ]])
        input_df = pd.DataFrame(input_data, columns=feature_names)
        prob = model.predict_proba(input_df)[0][1]
        risk_pct = prob * 100

        st.markdown("---")
        st.markdown("## 📊 Résultat de l'Analyse")
        col_gauge, col_detail = st.columns([1, 2])

        with col_gauge:
            if risk_pct >= 70:
                color, niveau, emoji = "#E63946", "RISQUE ÉLEVÉ", "🔴"
            elif risk_pct >= 40:
                color, niveau, emoji = "#F4A261", "RISQUE MODÉRÉ", "🟠"
            else:
                color, niveau, emoji = "#2A9D8F", "RISQUE FAIBLE", "🟢"

            st.markdown(f"""
                <div style='background:{color}; padding:30px; border-radius:15px;
                text-align:center; color:white;'>
                    <div style='font-size:3.5rem; font-weight:900;'>{risk_pct:.1f}%</div>
                    <div style='font-size:1.2rem; font-weight:700; margin-top:8px;'>
                    {emoji} {niveau}</div>
                    <div style='font-size:0.85rem; margin-top:8px; opacity:0.9;'>
                    Probabilité de rejet</div>
                </div>
            """, unsafe_allow_html=True)

        with col_detail:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            shap_series = pd.Series(shap_values[0], index=feature_names)
            top_positive = shap_series.nlargest(3)
            top_negative = shap_series.nsmallest(3)

            factor_labels = {
                'immatriculation_valid':'Immatriculation','ngap_coding_valid':'Codage NGAP',
                'pec_obtained':'PEC Obtenue','days_since_service':'Délai de soumission',
                'docs_completeness_ratio':'Complétude dossier','droits_active':'Droits AMO actifs',
                'cin_valid':'CIN Valide','inpe_present':'INPE Présent',
                'prescription_legible':'Ordonnance lisible','payer':'Organisme payeur',
                'service_type':'Type de prestation','ngap_code':'Code NGAP',
                'num_acts':"Nombre d'actes",'patient_age':'Âge patient',
                'is_ald':'ALD','is_ayant_droit':'Ayant droit','pec_required':'PEC requise'
            }

            st.markdown("#### 🔍 Facteurs de Risque Principaux")
            for feat, val in top_positive.items():
                label = factor_labels.get(feat, feat)
                st.markdown(f"""
                    <div style='background:#fff5f5; border-left:4px solid #E63946;
                    padding:10px 15px; margin:5px 0; border-radius:5px;'>
                        <strong>{label}</strong> — ↑ Augmente le risque
                        <span style='float:right; color:#E63946; font-weight:700;'>
                        +{val:.3f}</span>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("#### ✅ Facteurs Protecteurs")
            for feat, val in top_negative.items():
                label = factor_labels.get(feat, feat)
                st.markdown(f"""
                    <div style='background:#f0faf9; border-left:4px solid #2A9D8F;
                    padding:10px 15px; margin:5px 0; border-radius:5px;'>
                        <strong>{label}</strong> — Réduit le risque
                        <span style='float:right; color:#2A9D8F; font-weight:700;'>
                        {val:.3f}</span>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📉 Explication SHAP Détaillée")
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=feature_names
            ), show=False)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        if risk_pct >= 70:
            st.error("**⛔ NE PAS SOUMETTRE** — Corriger immatriculation, codage NGAP, et complétude des documents.")
        elif risk_pct >= 40:
            st.warning("**⚠️ SOUMETTRE AVEC PRÉCAUTION** — Vérifier les facteurs signalés ci-dessus.")
        else:
            st.success("**✅ DOSSIER PRÊT** — Faible probabilité de rejet. Soumettre à l'organisme payeur.")

# ─────────────────────────────────────────
# TAB 2 — DASHBOARD
# ─────────────────────────────────────────
with tab2:
    st.markdown("### 📈 Indicateurs Clés de Performance")
    total = len(df)
    rejected = df['rejected'].sum()
    rejection_rate = rejected / total * 100
    clean_rate = 100 - rejection_rate
    top_cause = df[df['rejected']==1]['rejection_cause'].value_counts().index[0]

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in zip(
        [c1,c2,c3,c4],
        [f"{total:,}", f"{rejection_rate:.1f}%", f"{clean_rate:.1f}%", top_cause.replace('_',' ')],
        ["Total Dossiers","Taux de Rejet","Dossiers Propres","Cause Principale"],
        ["#1D3557","#E63946","#2A9D8F","#E63946"]
    ):
        with col:
            st.markdown(f"""
                <div style='background:white; padding:20px; border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                border-left:4px solid {color}; text-align:center;'>
                    <div style='font-size:1.8rem; font-weight:900; color:{color};'>{val}</div>
                    <div style='color:#666; margin-top:5px;'>{label}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 📉 Analyse Pareto — Causes de Rejet")
        cause_counts = df[df['rejected']==1]['rejection_cause'].value_counts().reset_index()
        cause_counts.columns = ['Cause','Nombre']
        cause_counts['Cause'] = cause_counts['Cause'].str.replace('_',' ')
        cause_counts['Pourcentage'] = cause_counts['Nombre'] / cause_counts['Nombre'].sum() * 100
        cause_counts['Cumulatif'] = cause_counts['Pourcentage'].cumsum()

        fig_pareto = go.Figure()
        fig_pareto.add_trace(go.Bar(
            x=cause_counts['Cause'], y=cause_counts['Pourcentage'],
            name='% Rejets', marker_color='#E63946',
            text=cause_counts['Pourcentage'].round(1).astype(str)+'%',
            textposition='outside'))
        fig_pareto.add_trace(go.Scatter(
            x=cause_counts['Cause'], y=cause_counts['Cumulatif'],
            name='% Cumulatif', yaxis='y2',
            line=dict(color='#1D3557', width=2)))
        fig_pareto.update_layout(
            yaxis=dict(title='% Rejets'),
            yaxis2=dict(title='% Cumulatif', overlaying='y', side='right', range=[0,110]),
            legend=dict(orientation='h', y=-0.2),
            plot_bgcolor='white', height=350, margin=dict(t=20,b=20))
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col_right:
        st.markdown("#### 🏥 Taux de Rejet par Organisme")
        payer_map = {0:'AMO',1:'CNOPS',2:'CNSS'}
        df['payer_label'] = df['payer'].map(payer_map)
        payer_stats = df.groupby('payer_label')['rejected'].mean().reset_index()
        payer_stats.columns = ['Organisme','Taux']
        payer_stats['Taux'] = payer_stats['Taux'] * 100
        payer_stats['Couleur'] = payer_stats['Taux'].apply(
            lambda x: '#E63946' if x>40 else '#F4A261' if x>35 else '#2A9D8F')

        fig_payer = go.Figure(go.Bar(
            x=payer_stats['Organisme'], y=payer_stats['Taux'],
            marker_color=payer_stats['Couleur'],
            text=payer_stats['Taux'].round(1).astype(str)+'%',
            textposition='outside'))
        fig_payer.update_layout(
            yaxis=dict(title='Taux de Rejet (%)', range=[0,60]),
            plot_bgcolor='white', height=350, margin=dict(t=20,b=20))
        st.plotly_chart(fig_payer, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Impact des Features sur le Taux de Rejet")
    binary_features = {
        'immatriculation_valid':'Immatriculation Valide',
        'ngap_coding_valid':'Codage NGAP Valide',
        'cin_valid':'CIN Valide',
        'prescription_legible':'Ordonnance Lisible',
        'droits_active':'Droits AMO Actifs',
        'inpe_present':'INPE Présent',
        'pec_obtained':'PEC Obtenue'
    }
    impact_data = []
    for col, label in binary_features.items():
        r0 = df[df[col]==0]['rejected'].mean() * 100
        r1 = df[df[col]==1]['rejected'].mean() * 100
        impact_data.append({'Feature':label,'Absent (0)':r0,'Présent (1)':r1})
    impact_df = pd.DataFrame(impact_data)

    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(name='Absent / Invalide',
        x=impact_df['Feature'], y=impact_df['Absent (0)'],
        marker_color='#E63946',
        text=impact_df['Absent (0)'].round(1).astype(str)+'%', textposition='outside'))
    fig_impact.add_trace(go.Bar(name='Présent / Valide',
        x=impact_df['Feature'], y=impact_df['Présent (1)'],
        marker_color='#2A9D8F',
        text=impact_df['Présent (1)'].round(1).astype(str)+'%', textposition='outside'))
    fig_impact.update_layout(
        barmode='group',
        yaxis=dict(title='Taux de Rejet (%)', range=[0,100]),
        legend=dict(orientation='h', y=-0.2),
        plot_bgcolor='white', height=380, margin=dict(t=20,b=20))
    st.plotly_chart(fig_impact, use_container_width=True)

# ─────────────────────────────────────────
# TAB 3 — PERFORMANCE
# ─────────────────────────────────────────
with tab3:
    drop_cols = ['claim_id','service_date','rejection_cause',
                 'forclusion_risk','droits_verified','montant_reclame_mad']
    df_model = df.drop(columns=drop_cols)
    for col in ['payer','service_type','ngap_code']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])

    X = df_model[feature_names]
    y = df_model['rejected']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.markdown("### 🏆 Scorecard du Modèle")
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, label, value, color in zip(
        [c1,c2,c3,c4,c5],
        ["AUC","Précision","Rappel","F1-Score","Exactitude"],
        [f"{roc_auc:.3f}", f"{report['1']['precision']:.1%}",
         f"{report['1']['recall']:.1%}", f"{report['1']['f1-score']:.3f}",
         f"{report['accuracy']:.1%}"],
        ["#1D3557","#2A9D8F","#2A9D8F","#457B9D","#457B9D"]
    ):
        with col:
            st.markdown(f"""
                <div style='background:white; padding:20px; border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,0.08);
                border-left:4px solid {color}; text-align:center;'>
                    <div style='font-size:1.8rem; font-weight:900; color:{color};'>
                    {value}</div>
                    <div style='color:#666; margin-top:5px;'>{label}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("#### 📈 Courbe ROC")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
            name=f'XGBoost (AUC={roc_auc:.3f})',
            line=dict(color='#1D3557', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
            name='Aléatoire', line=dict(color='#E63946', width=2, dash='dash')))
        fig_roc.update_layout(
            xaxis=dict(title='Faux Positifs'),
            yaxis=dict(title='Vrais Positifs'),
            plot_bgcolor='white', height=350, margin=dict(t=20,b=20))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        st.markdown("#### 🔲 Matrice de Confusion")
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Prédit: Accepté','Prédit: Rejeté'],
            y=['Réel: Accepté','Réel: Rejeté'],
            colorscale=[[0,'#f8f9fa'],[1,'#1D3557']],
            text=cm, texttemplate='%{text}',
            textfont=dict(size=24, color='white'), showscale=False))
        fig_cm.update_layout(height=350, margin=dict(t=20,b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🧠 Importance Globale SHAP")

    @st.cache_data
    def compute_shap(_model, _X_test):
        explainer = shap.TreeExplainer(_model)
        return explainer.shap_values(_X_test)

    shap_values = compute_shap(model, X_test)
    french_labels = {
        'immatriculation_valid':'Immatriculation Valide',
        'ngap_coding_valid':'Codage NGAP Valide',
        'pec_obtained':'PEC Obtenue',
        'days_since_service':'Jours depuis Prestation',
        'docs_completeness_ratio':'Complétude Dossier',
        'droits_active':'Droits AMO Actifs',
        'cin_valid':'CIN Valide','inpe_present':'INPE Présent',
        'prescription_legible':'Ordonnance Lisible',
        'payer':'Organisme Payeur','service_type':'Type de Prestation',
        'ngap_code':'Code NGAP','num_acts':"Nombre d'Actes",
        'patient_age':'Âge Patient','is_ald':'ALD',
        'is_ayant_droit':'Ayant Droit','pec_required':'PEC Requise'
    }
    mean_shap = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('Importance', ascending=True)
    mean_shap['FR'] = mean_shap['Feature'].map(french_labels)

    fig_shap = go.Figure(go.Bar(
        x=mean_shap['Importance'], y=mean_shap['FR'],
        orientation='h', marker_color='#1D3557',
        text=mean_shap['Importance'].round(3), textposition='outside'))
    fig_shap.update_layout(
        xaxis=dict(title='Importance SHAP Moyenne'),
        plot_bgcolor='white', height=480,
        margin=dict(t=20,b=20,l=200))
    st.plotly_chart(fig_shap, use_container_width=True)

# ─────────────────────────────────────────
# TAB 4 — SIMULATEUR ROI
# ─────────────────────────────────────────
with tab4:
    st.markdown("### ⚙️ Paramètres de votre Établissement")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Volume & Finances**")
        nb_dossiers_mois = st.slider("Dossiers soumis par mois", 100, 5000, 800, step=50)
        montant_moyen_mad = st.slider("Montant moyen par dossier (MAD)", 500, 15000, 3200, step=100)
        taux_rejet_actuel = st.slider("Taux de rejet actuel (%)", 5, 70, 38, step=1)

    with col2:
        st.markdown("**Performance SihaIQ**")
        taux_rejet_cible = st.slider("Taux de rejet cible avec SihaIQ (%)", 5, 50, 18, step=1)
        taux_recuperation = st.slider("Taux de récupération (%)", 30, 95, 72, step=1)
        abonnement_mad = st.selectbox("Plan SihaIQ (MAD/mois)",
            options=[2000,5000,8000,12000],
            format_func=lambda x: {
                2000:"Starter — 2,000 MAD/mois",
                5000:"Professionnel — 5,000 MAD/mois",
                8000:"Clinique — 8,000 MAD/mois",
                12000:"Entreprise — 12,000 MAD/mois"}[x])

    volume_mad_mois = nb_dossiers_mois * montant_moyen_mad
    rejets_actuels = nb_dossiers_mois * (taux_rejet_actuel/100)
    mad_perdu_mois = rejets_actuels * montant_moyen_mad
    rejets_evites = rejets_actuels - nb_dossiers_mois * (taux_rejet_cible/100)
    mad_recupere_mois = rejets_evites * montant_moyen_mad * (taux_recuperation/100)
    performance_fee = mad_recupere_mois * 0.04
    cout_total_mois = abonnement_mad + performance_fee
    roi_mois = mad_recupere_mois - cout_total_mois
    roi_annuel = roi_mois * 12
    payback_days = (cout_total_mois / mad_recupere_mois * 30) if mad_recupere_mois > 0 else 0

    st.markdown("---")
    st.markdown("### 📊 Résultats")
    k1,k2,k3,k4 = st.columns(4)
    for col, val, label, color in zip(
        [k1,k2,k3,k4],
        [f"{mad_perdu_mois:,.0f} MAD", f"{mad_recupere_mois:,.0f} MAD",
         f"{roi_annuel:,.0f} MAD", f"{payback_days:.0f} jours"],
        ["Pertes Actuelles/Mois","Récupération/Mois","ROI Net Annuel","Délai Rentabilisation"],
        ["#E63946","#2A9D8F","#1D3557","#457B9D"]
    ):
        with col:
            st.markdown(f"""
                <div style='background:{color}; padding:20px; border-radius:12px;
                text-align:center; color:white;'>
                    <div style='font-size:1.5rem; font-weight:900;'>{val}</div>
                    <div style='font-size:0.8rem; margin-top:5px; opacity:0.9;'>{label}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    mois = [f"M{i}" for i in range(1,13)]
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Bar(name='Pertes sans SihaIQ',
        x=mois, y=[mad_perdu_mois]*12, marker_color='#E63946', opacity=0.7))
    fig_proj.add_trace(go.Scatter(name='Récupération Cumulée',
        x=mois, y=[mad_recupere_mois*i for i in range(1,13)],
        mode='lines+markers', line=dict(color='#2A9D8F', width=3)))
    fig_proj.add_trace(go.Scatter(name='ROI Net Cumulé',
        x=mois, y=[roi_mois*i for i in range(1,13)],
        mode='lines+markers', line=dict(color='#1D3557', width=3, dash='dash')))
    fig_proj.update_layout(
        yaxis=dict(title='MAD'),
        legend=dict(orientation='h', y=-0.2),
        plot_bgcolor='white', height=350, margin=dict(t=20,b=20))
    st.plotly_chart(fig_proj, use_container_width=True)

# ─────────────────────────────────────────
# TAB 5 — BATCH SCORER
# ─────────────────────────────────────────
with tab5:
    st.markdown("### 📤 Scoring par Lot — Analyse BAF en Masse")

    with st.expander("📋 Format requis du fichier CSV"):
        st.markdown("""
        | Colonne | Valeurs |
        |---|---|
        | `payer` | AMO, CNOPS, CNSS |
        | `service_type` | Biologie, Chirurgie, Consultation, Hospitalisation, Imagerie, Urgences |
        | `ngap_code` | B, C, C2, Cs, FORFAIT, HN, K, P, V, Z |
        | `num_acts` | Entier |
        | `patient_age` | 0–100 |
        | Colonnes binaires | 0 ou 1 |
        | `docs_completeness_ratio` | 0.0–1.0 |
        | `days_since_service` | Entier |
        """)

    sample_data = {
        'claim_id':[f'BAF-{i:04d}' for i in range(1,11)],
        'payer':['CNOPS','CNSS','AMO','CNOPS','CNSS','AMO','CNOPS','CNSS','AMO','CNOPS'],
        'service_type':['Consultation','Chirurgie','Biologie','Hospitalisation',
                        'Imagerie','Urgences','Consultation','Biologie','Chirurgie','Imagerie'],
        'ngap_code':['C','K','B','K','Z','C','Cs','B','K','Z'],
        'num_acts':[2,5,3,8,1,2,3,4,6,2],
        'patient_age':[45,62,28,71,35,19,55,40,67,30],
        'is_ald':[0,1,0,1,0,0,1,0,1,0],
        'is_ayant_droit':[0,0,1,0,0,1,0,0,0,1],
        'inpe_present':[1,1,0,1,1,1,0,1,1,1],
        'immatriculation_valid':[1,0,1,1,0,1,1,1,0,1],
        'cin_valid':[1,1,0,1,1,0,1,1,1,0],
        'ngap_coding_valid':[1,0,1,0,1,1,0,1,0,1],
        'prescription_legible':[1,1,1,0,1,1,1,0,1,1],
        'droits_active':[1,1,1,0,0,1,1,1,0,1],
        'docs_completeness_ratio':[0.95,0.60,0.85,0.40,0.70,0.90,0.55,0.88,0.45,0.92],
        'days_since_service':[5,45,12,60,30,3,55,20,48,8],
        'pec_required':[0,1,0,1,0,0,1,0,1,0],
        'pec_obtained':[0,0,0,1,0,0,0,0,0,0],
    }
    sample_df = pd.DataFrame(sample_data)
    csv_buf = io.StringIO()
    sample_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Télécharger fichier exemple (10 dossiers)",
        data=csv_buf.getvalue(), file_name="sihaiq_exemple.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Glissez votre fichier CSV ici", type=['csv'])

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(df_input)} dossiers chargés")

            df_scoring = df_input.copy()
            payer_map = {'AMO':0,'CNOPS':1,'CNSS':2}
            service_map = {'Biologie':0,'Chirurgie':1,'Consultation':2,
                          'Hospitalisation':3,'Imagerie':4,'Urgences':5}
            ngap_map = {'B':0,'C':1,'C2':2,'Cs':3,'FORFAIT':4,
                       'HN':5,'K':6,'P':7,'V':8,'Z':9}
            df_scoring['payer'] = df_scoring['payer'].map(payer_map)
            df_scoring['service_type'] = df_scoring['service_type'].map(service_map)
            df_scoring['ngap_code'] = df_scoring['ngap_code'].map(ngap_map)

            probs = model.predict_proba(df_scoring[feature_names])[:,1]
            predictions = model.predict(df_scoring[feature_names])

            results = df_input.copy()
            results['Score_Risque_%'] = (probs*100).round(1)
            results['Niveau_Risque'] = pd.cut(probs*100,
                bins=[0,40,70,100], labels=['🟢 FAIBLE','🟠 MODÉRÉ','🔴 ÉLEVÉ'])
            results['Recommandation'] = pd.Series(predictions).map(
                {0:'✅ Soumettre', 1:'⛔ Corriger avant soumission'})

            high = (probs>=0.70).sum()
            med = ((probs>=0.40)&(probs<0.70)).sum()
            low = (probs<0.40).sum()

            k1,k2,k3,k4 = st.columns(4)
            for col, val, label, color in zip(
                [k1,k2,k3,k4],
                [high, med, low, f"{probs.mean()*100:.1f}%"],
                ["🔴 Risque Élevé","🟠 Risque Modéré","🟢 Risque Faible","Score Moyen"],
                ["#E63946","#F4A261","#2A9D8F","#1D3557"]
            ):
                with col:
                    st.markdown(f"""
                        <div style='background:{color}; padding:15px; border-radius:10px;
                        text-align:center; color:white;'>
                            <div style='font-size:1.8rem; font-weight:900;'>{val}</div>
                            <div style='font-size:0.85rem;'>{label}</div>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            display_cols = ['claim_id','payer','service_type','Score_Risque_%',
                           'Niveau_Risque','Recommandation'] \
                if 'claim_id' in results.columns else \
                ['payer','service_type','Score_Risque_%','Niveau_Risque','Recommandation']

            st.dataframe(results[display_cols].sort_values('Score_Risque_%', ascending=False),
                use_container_width=True, height=380)

            out_buf = io.StringIO()
            results.to_csv(out_buf, index=False)
            st.download_button("📥 Télécharger résultats scorés",
                data=out_buf.getvalue(),
                file_name="sihaiq_resultats.csv",
                mime="text/csv", type="primary")

        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.8rem;'>
    SihaIQ v1.0 • Données synthétiques CNDP-conformes • CM6RI Lab d'Innovation 2026
</div>
""", unsafe_allow_html=True)