import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
import shap
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report)

st.set_page_config(page_title="Performance — SihaIQ", 
                   page_icon="🔬", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1D3557; }
        [data-testid="stSidebar"] * { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='background: linear-gradient(90deg, #1D3557 0%, #457B9D 100%);
    padding: 20px 30px; border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; margin:0;'>🔬 Performance du Modèle XGBoost</h1>
        <p style='color: #A8DADC; margin:5px 0 0 0;'>
        Évaluation complète — AUC, ROC, Matrice de Confusion, SHAP Global</p>
    </div>
""", unsafe_allow_html=True)

# Load model and data
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

# Prepare test set — same split as training (seed=42, 80/20)
drop_cols = ['claim_id', 'service_date', 'rejection_cause',
             'forclusion_risk', 'droits_verified', 'montant_reclame_mad']
df_model = df.drop(columns=drop_cols)

# Encode categoricals same way as training
from sklearn.preprocessing import LabelEncoder
for col in ['payer', 'service_type', 'ngap_code']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

X = df_model[feature_names]
y = df_model['rejected']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

precision = report['1']['precision']
recall = report['1']['recall']
f1 = report['1']['f1-score']
accuracy = report['accuracy']

# Model Scorecard
st.markdown("### 🏆 Scorecard du Modèle")
c1, c2, c3, c4, c5 = st.columns(5)

metrics = [
    ("AUC", f"{roc_auc:.3f}", "#1D3557"),
    ("Précision", f"{precision:.1%}", "#2A9D8F"),
    ("Rappel", f"{recall:.1%}", "#2A9D8F"),
    ("F1-Score", f"{f1:.3f}", "#457B9D"),
    ("Exactitude", f"{accuracy:.1%}", "#457B9D"),
]

for col, (label, value, color) in zip([c1,c2,c3,c4,c5], metrics):
    with col:
        st.markdown(f"""
            <div style='background:white; padding:20px; border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
            border-left:4px solid {color}; text-align:center;'>
                <div style='font-size:1.8rem; font-weight:900; color:{color};'>
                {value}</div>
                <div style='color:#666; margin-top:5px; font-size:0.9rem;'>
                {label}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# Row 2: ROC + Confusion Matrix
col_roc, col_cm = st.columns(2)

with col_roc:
    st.markdown("#### 📈 Courbe ROC")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'XGBoost (AUC = {roc_auc:.3f})',
        line=dict(color='#1D3557', width=3)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1],
        mode='lines',
        name='Aléatoire (AUC = 0.500)',
        line=dict(color='#E63946', width=2, dash='dash')
    ))
    fig_roc.update_layout(
        xaxis=dict(title='Taux de Faux Positifs'),
        yaxis=dict(title='Taux de Vrais Positifs'),
        legend=dict(x=0.4, y=0.1),
        plot_bgcolor='white',
        height=400,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with col_cm:
    st.markdown("#### 🔲 Matrice de Confusion")
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=['Prédit: Accepté', 'Prédit: Rejeté'],
        y=['Réel: Accepté', 'Réel: Rejeté'],
        colorscale=[[0, '#f8f9fa'], [1, '#1D3557']],
        text=cm,
        texttemplate='%{text}',
        textfont=dict(size=24, color='white'),
        showscale=False
    ))
    fig_cm.update_layout(
        height=400,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# SHAP Global Importance
st.markdown("#### 🧠 Importance Globale des Features — SHAP")

@st.cache_data
def compute_shap(_model, _X_test):
    explainer = shap.TreeExplainer(_model)
    shap_vals = explainer.shap_values(_X_test)
    return shap_vals

shap_values = compute_shap(model, X_test)

mean_shap = pd.DataFrame({
    'Feature': feature_names,
    'Importance SHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('Importance SHAP', ascending=True)

french_labels = {
    'immatriculation_valid': 'Immatriculation Valide',
    'ngap_coding_valid': 'Codage NGAP Valide',
    'pec_obtained': 'PEC Obtenue',
    'days_since_service': 'Jours depuis Prestation',
    'docs_completeness_ratio': 'Complétude Dossier',
    'droits_active': 'Droits AMO Actifs',
    'cin_valid': 'CIN Valide',
    'inpe_present': 'INPE Présent',
    'prescription_legible': 'Ordonnance Lisible',
    'payer': 'Organisme Payeur',
    'service_type': 'Type de Prestation',
    'ngap_code': 'Code NGAP',
    'num_acts': "Nombre d'Actes",
    'patient_age': 'Âge Patient',
    'is_ald': 'ALD',
    'is_ayant_droit': 'Ayant Droit',
    'pec_required': 'PEC Requise'
}

mean_shap['Feature FR'] = mean_shap['Feature'].map(french_labels)

fig_shap = go.Figure(go.Bar(
    x=mean_shap['Importance SHAP'],
    y=mean_shap['Feature FR'],
    orientation='h',
    marker_color='#1D3557',
    text=mean_shap['Importance SHAP'].round(3),
    textposition='outside'
))
fig_shap.update_layout(
    xaxis=dict(title='Importance SHAP Moyenne |valeur|'),
    plot_bgcolor='white',
    height=500,
    margin=dict(t=20, b=20, l=200)
)
st.plotly_chart(fig_shap, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    Modèle entraîné sur 2,400 dossiers synthétiques • 
    Évalué sur 600 dossiers • Seed=42 • CNDP-conforme
</div>
""", unsafe_allow_html=True)