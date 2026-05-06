import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import io

st.set_page_config(page_title="Batch Scorer — SihaIQ", 
                   page_icon="📦", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1D3557; }
        [data-testid="stSidebar"] * { color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
    padding: 20px 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin:0;'>📦 Scoring par Lot — Analyse BAF en Masse</h1>
        <p style='color: #A8DADC; margin:5px 0 0 0;'>
        Uploadez un fichier CSV de dossiers → SihaIQ score chaque dossier en temps réel</p>
    </div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("model/sihaiq_xgboost_model.pkl")
    features = joblib.load("model/sihaiq_feature_names.pkl")
    return model, features

model, feature_names = load_model()

# Instructions
with st.expander("📋 Format requis du fichier CSV — Cliquez pour voir"):
    st.markdown("""
    Votre CSV doit contenir ces colonnes (dans n'importe quel ordre):
    
    | Colonne | Valeurs |
    |---|---|
    | `payer` | AMO, CNOPS, CNSS |
    | `service_type` | Biologie, Chirurgie, Consultation, Hospitalisation, Imagerie, Urgences |
    | `ngap_code` | B, C, C2, Cs, FORFAIT, HN, K, P, V, Z |
    | `num_acts` | Nombre entier |
    | `patient_age` | 0–100 |
    | `is_ald` | 0 ou 1 |
    | `is_ayant_droit` | 0 ou 1 |
    | `inpe_present` | 0 ou 1 |
    | `immatriculation_valid` | 0 ou 1 |
    | `cin_valid` | 0 ou 1 |
    | `ngap_coding_valid` | 0 ou 1 |
    | `prescription_legible` | 0 ou 1 |
    | `droits_active` | 0 ou 1 |
    | `docs_completeness_ratio` | 0.0 – 1.0 |
    | `days_since_service` | Nombre entier |
    | `pec_required` | 0 ou 1 |
    | `pec_obtained` | 0 ou 1 |
    """)

# Demo CSV download
st.markdown("### 📥 Télécharger un fichier exemple")

sample_data = {
    'claim_id': [f'BAF-{i:04d}' for i in range(1, 11)],
    'payer': ['CNOPS','CNSS','AMO','CNOPS','CNSS','AMO','CNOPS','CNSS','AMO','CNOPS'],
    'service_type': ['Consultation','Chirurgie','Biologie','Hospitalisation',
                     'Imagerie','Urgences','Consultation','Biologie','Chirurgie','Imagerie'],
    'ngap_code': ['C','K','B','K','Z','C','Cs','B','K','Z'],
    'num_acts': [2,5,3,8,1,2,3,4,6,2],
    'patient_age': [45,62,28,71,35,19,55,40,67,30],
    'is_ald': [0,1,0,1,0,0,1,0,1,0],
    'is_ayant_droit': [0,0,1,0,0,1,0,0,0,1],
    'inpe_present': [1,1,0,1,1,1,0,1,1,1],
    'immatriculation_valid': [1,0,1,1,0,1,1,1,0,1],
    'cin_valid': [1,1,0,1,1,0,1,1,1,0],
    'ngap_coding_valid': [1,0,1,0,1,1,0,1,0,1],
    'prescription_legible': [1,1,1,0,1,1,1,0,1,1],
    'droits_active': [1,1,1,0,0,1,1,1,0,1],
    'docs_completeness_ratio': [0.95,0.60,0.85,0.40,0.70,0.90,0.55,0.88,0.45,0.92],
    'days_since_service': [5,45,12,60,30,3,55,20,48,8],
    'pec_required': [0,1,0,1,0,0,1,0,1,0],
    'pec_obtained': [0,0,0,1,0,0,0,0,0,0],
}

sample_df = pd.DataFrame(sample_data)
csv_buffer = io.StringIO()
sample_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="⬇️ Télécharger fichier exemple (10 dossiers)",
    data=csv_buffer.getvalue(),
    file_name="sihaiq_exemple_dossiers.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("### 📤 Uploader vos Dossiers BAF")

uploaded_file = st.file_uploader(
    "Glissez votre fichier CSV ici", 
    type=['csv'],
    help="Format CSV avec les colonnes requises ci-dessus"
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.success(f"✅ {len(df_input)} dossiers chargés avec succès")
        
        with st.expander("Aperçu des données chargées"):
            st.dataframe(df_input.head())
        
        # Encode categoricals
        df_scoring = df_input.copy()
        
        payer_map = {'AMO': 0, 'CNOPS': 1, 'CNSS': 2}
        service_map = {'Biologie': 0, 'Chirurgie': 1, 'Consultation': 2,
                      'Hospitalisation': 3, 'Imagerie': 4, 'Urgences': 5}
        ngap_map = {'B': 0, 'C': 1, 'C2': 2, 'Cs': 3, 'FORFAIT': 4,
                   'HN': 5, 'K': 6, 'P': 7, 'V': 8, 'Z': 9}
        
        df_scoring['payer'] = df_scoring['payer'].map(payer_map)
        df_scoring['service_type'] = df_scoring['service_type'].map(service_map)
        df_scoring['ngap_code'] = df_scoring['ngap_code'].map(ngap_map)
        
        X = df_scoring[feature_names]
        
        # Score all claims
        probs = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)
        
        # Build results
        results = df_input.copy()
        results['Score_Risque_%'] = (probs * 100).round(1)
        results['Niveau_Risque'] = pd.cut(
            probs * 100,
            bins=[0, 40, 70, 100],
            labels=['🟢 FAIBLE', '🟠 MODÉRÉ', '🔴 ÉLEVÉ']
        )
        results['Recommandation'] = predictions
        results['Recommandation'] = results['Recommandation'].map({
            0: '✅ Soumettre',
            1: '⛔ Corriger avant soumission'
        })
        
        st.markdown("---")
        st.markdown("### 📊 Résultats du Scoring par Lot")
        
        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        
        high_risk = (probs >= 0.70).sum()
        medium_risk = ((probs >= 0.40) & (probs < 0.70)).sum()
        low_risk = (probs < 0.40).sum()
        avg_score = probs.mean() * 100
        
        with k1:
            st.markdown(f"""
                <div style='background:#E63946; padding:15px; border-radius:10px;
                text-align:center; color:white;'>
                    <div style='font-size:1.8rem; font-weight:900;'>{high_risk}</div>
                    <div style='font-size:0.85rem;'>🔴 Risque Élevé</div>
                </div>
            """, unsafe_allow_html=True)
        
        with k2:
            st.markdown(f"""
                <div style='background:#F4A261; padding:15px; border-radius:10px;
                text-align:center; color:white;'>
                    <div style='font-size:1.8rem; font-weight:900;'>{medium_risk}</div>
                    <div style='font-size:0.85rem;'>🟠 Risque Modéré</div>
                </div>
            """, unsafe_allow_html=True)
        
        with k3:
            st.markdown(f"""
                <div style='background:#2A9D8F; padding:15px; border-radius:10px;
                text-align:center; color:white;'>
                    <div style='font-size:1.8rem; font-weight:900;'>{low_risk}</div>
                    <div style='font-size:0.85rem;'>🟢 Risque Faible</div>
                </div>
            """, unsafe_allow_html=True)
        
        with k4:
            st.markdown(f"""
                <div style='background:#1D3557; padding:15px; border-radius:10px;
                text-align:center; color:white;'>
                    <div style='font-size:1.8rem; font-weight:900;'>{avg_score:.1f}%</div>
                    <div style='font-size:0.85rem;'>Score Moyen</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Results table with color coding
        st.markdown("#### 📋 Détail par Dossier")
        
        display_cols = ['claim_id', 'payer', 'service_type', 
                       'Score_Risque_%', 'Niveau_Risque', 'Recommandation'] \
                       if 'claim_id' in results.columns else \
                       ['payer', 'service_type', 
                       'Score_Risque_%', 'Niveau_Risque', 'Recommandation']
        
        st.dataframe(
            results[display_cols].sort_values('Score_Risque_%', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Download results
        st.markdown("#### ⬇️ Exporter les Résultats")
        
        output_buffer = io.StringIO()
        results.to_csv(output_buffer, index=False)
        
        st.download_button(
            label="📥 Télécharger résultats scorés (CSV)",
            data=output_buffer.getvalue(),
            file_name="sihaiq_resultats_scoring.csv",
            mime="text/csv",
            type="primary"
        )
        
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement: {str(e)}")
        st.info("Vérifiez que votre fichier contient toutes les colonnes requises.")