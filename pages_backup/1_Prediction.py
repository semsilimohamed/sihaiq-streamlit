import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(page_title="Prédiction — SihaIQ", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1D3557; }
        [data-testid="stSidebar"] * { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("model/sihaiq_xgboost_model.pkl")
    features = joblib.load("model/sihaiq_feature_names.pkl")
    return model, features

model, feature_names = load_model()

# Header
st.markdown("""
    <div style='background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
    padding: 20px 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin:0;'>🎯 Prédiction de Rejet de Dossier</h1>
        <p style='color: #A8DADC; margin:5px 0 0 0;'>
        Scorez un dossier BAF avant soumission — explication SHAP intégrée</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 📋 Saisie du Dossier")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Informations Payeur**")
    payer = st.selectbox("Organisme Payeur", 
        options=[0, 1, 2], 
        format_func=lambda x: {0: "AMO", 1: "CNOPS", 2: "CNSS"}[x])
    
    service_type = st.selectbox("Type de Prestation",
        options=[0, 1, 2, 3, 4, 5],
        format_func=lambda x: {
            0: "Biologie", 1: "Chirurgie", 2: "Consultation",
            3: "Hospitalisation", 4: "Imagerie", 5: "Urgences"
        }[x])
    
    ngap_code = st.selectbox("Code NGAP",
        options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        format_func=lambda x: {
            0: "B", 1: "C", 2: "C2", 3: "Cs", 4: "FORFAIT",
            5: "HN", 6: "K", 7: "P", 8: "V", 9: "Z"
        }[x])
    
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

st.markdown("---")

# Predict button
if st.button("🔍 Analyser le Dossier", type="primary", use_container_width=True):
    
    # Build input vector in exact feature order
    input_data = np.array([[
        payer, service_type, ngap_code, num_acts, patient_age,
        int(is_ald), int(is_ayant_droit), int(inpe_present),
        int(immatriculation_valid), int(cin_valid), int(ngap_coding_valid),
        int(prescription_legible), int(droits_active),
        docs_completeness_ratio, days_since_service,
        int(pec_required), int(pec_obtained)
    ]])
    
    input_df = pd.DataFrame(input_data, columns=feature_names)
    
    # Predict
    prob = model.predict_proba(input_df)[0][1]
    risk_pct = prob * 100
    
    # Risk gauge display
    st.markdown("## 📊 Résultat de l'Analyse")
    
    col_gauge, col_detail = st.columns([1, 2])
    
    with col_gauge:
        if risk_pct >= 70:
            color = "#E63946"
            niveau = "RISQUE ÉLEVÉ"
            emoji = "🔴"
        elif risk_pct >= 40:
            color = "#F4A261"
            niveau = "RISQUE MODÉRÉ"
            emoji = "🟠"
        else:
            color = "#2A9D8F"
            niveau = "RISQUE FAIBLE"
            emoji = "🟢"
        
        st.markdown(f"""
            <div style='background: {color}; padding: 30px; border-radius: 15px; 
            text-align: center; color: white;'>
                <div style='font-size: 3.5rem; font-weight: 900;'>{risk_pct:.1f}%</div>
                <div style='font-size: 1.2rem; font-weight: 700; margin-top: 8px;'>
                {emoji} {niveau}</div>
                <div style='font-size: 0.85rem; margin-top: 8px; opacity: 0.9;'>
                Probabilité de rejet</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col_detail:
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # Top 3 risk factors
        shap_series = pd.Series(shap_values[0], index=feature_names)
        top_positive = shap_series.nlargest(3)
        top_negative = shap_series.nsmallest(3)
        
        st.markdown("#### 🔍 Facteurs de Risque Principaux")
        
        factor_labels = {
            'immatriculation_valid': 'Immatriculation',
            'ngap_coding_valid': 'Codage NGAP',
            'pec_obtained': 'PEC Obtenue',
            'days_since_service': 'Délai de soumission',
            'docs_completeness_ratio': 'Complétude dossier',
            'droits_active': 'Droits AMO actifs',
            'cin_valid': 'CIN Valide',
            'inpe_present': 'INPE Présent',
            'prescription_legible': 'Ordonnance lisible',
            'payer': 'Organisme payeur',
            'service_type': 'Type de prestation',
            'ngap_code': 'Code NGAP',
            'num_acts': "Nombre d'actes",
            'patient_age': 'Âge patient',
            'is_ald': 'ALD',
            'is_ayant_droit': 'Ayant droit',
            'pec_required': 'PEC requise'
        }
        
        for feat, val in top_positive.items():
            label = factor_labels.get(feat, feat)
            impact = "↑ Augmente le risque de rejet"
            st.markdown(f"""
                <div style='background: #fff5f5; border-left: 4px solid #E63946; 
                padding: 10px 15px; margin: 5px 0; border-radius: 5px;'>
                    <strong>{label}</strong> — {impact}
                    <span style='float:right; color:#E63946; font-weight:700;'>
                    +{val:.3f}</span>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### ✅ Facteurs Protecteurs")
        for feat, val in top_negative.items():
            label = factor_labels.get(feat, feat)
            st.markdown(f"""
                <div style='background: #f0faf9; border-left: 4px solid #2A9D8F; 
                padding: 10px 15px; margin: 5px 0; border-radius: 5px;'>
                    <strong>{label}</strong> — Réduit le risque
                    <span style='float:right; color:#2A9D8F; font-weight:700;'>
                    {val:.3f}</span>
                </div>
            """, unsafe_allow_html=True)
    
    # SHAP Waterfall
    st.markdown("---")
    st.markdown("#### 📉 Explication SHAP Détaillée (Waterfall)")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()
    
    # Action recommendation
    st.markdown("---")
    st.markdown("#### 💡 Recommandation Agent BAF")
    
    if risk_pct >= 70:
        st.error("""
        **⛔ NE PAS SOUMETTRE** — Dossier à haut risque de rejet.
        Corriger impérativement avant soumission :
        vérifier immatriculation, codage NGAP, et complétude des documents.
        """)
    elif risk_pct >= 40:
        st.warning("""
        **⚠️ SOUMETTRE AVEC PRÉCAUTION** — Risque modéré détecté.
        Vérifier les facteurs signalés ci-dessus avant envoi.
        """)
    else:
        st.success("""
        **✅ DOSSIER PRÊT** — Faible probabilité de rejet.
        Le dossier peut être soumis à l'organisme payeur.
        """)