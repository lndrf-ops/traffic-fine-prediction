import streamlit as st
import joblib
import pandas as pd
import os
from PIL import Image

# 1. Modell und Features aus dem models/ Ordner laden
@st.cache_resource
def load_model():
    # Wir passen die Pfade an die neue VS Code Struktur an
    model = joblib.load('models/rf_model.pkl')
    features = joblib.load('models/model_features.pkl')
    return model, features

try:
    model, features = load_model()
except FileNotFoundError:
    st.error("Fehler: Modelle nicht gefunden. Hast du 'python run_pipeline.py' ausgeführt?")
    st.stop()

# 2. Seiten-Layout
st.set_page_config(page_title="Strafzettel KI", page_icon="🚦", layout="wide")

st.title("🚦 Predictive Analytics: Bußgeld-Prozess")
st.markdown("Dieses Dashboard prognostiziert für laufende Fälle, ob ein Bußgeld bezahlt wird oder ob eine Inkasso-Übergabe droht.")

# 3. Eingabebereich (Sidebar)
st.sidebar.header("📝 Falldaten eingeben")
st.sidebar.markdown("Welche Schritte sind bereits passiert?")

user_input = {}
amount = st.sidebar.number_input("Höhe des Bußgeldes (€):", min_value=0.0, value=35.0, step=5.0)

for feature in features:
    if feature == 'amount':
        user_input[feature] = amount
    elif feature == 'Payment':
        user_input[feature] = 0 
    else:
        is_checked = st.sidebar.checkbox(f"Aktivität: {feature}")
        user_input[feature] = 1 if is_checked else 0

# 4. Vorhersage-Logik
if st.sidebar.button("🔍 Fall analysieren", type="primary"):
    input_df = pd.DataFrame([user_input], columns=features)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # 5. Ergebnis anzeigen
    st.divider()
    st.subheader("📊 KI-Analyse des Falls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ **WARNUNG: Hohes Inkasso-Risiko!**")
            st.write("Das System prognostiziert, dass dieser Fall an ein Inkassobüro übergeben werden muss.")
        else:
            st.success("✅ **STATUS: Zahlung wahrscheinlich.**")
            st.write("Das System prognostiziert, dass das Bußgeld im regulären Prozess bezahlt wird.")
            
    with col2:
        st.metric(label="Wahrscheinlichkeit für Inkasso", value=f"{probability * 100:.1f} %")
        st.info("💡 **Empfehlung:** Wenn die Wahrscheinlichkeit über 50% steigt, prüfen Sie alternative Eskalationsstufen (z.B. SMS-Erinnerung).")

    # 6. SHAP Erklärung einbinden (Bonus!)
    st.divider()
    st.subheader("🧠 Warum entscheidet die KI so?")
    st.markdown("Das **SHAP (Explainable AI)** Modell zeigt, welche Faktoren am stärksten zu einem Inkasso-Fall führen (Rote Punkte rechts der Mitte = Treibt das Risiko nach oben).")
    
    shap_path = 'models/shap_summary.png'
    if os.path.exists(shap_path):
        image = Image.open(shap_path)
        st.image(image, use_column_width=True)
    else:
        st.warning("SHAP Diagramm noch nicht generiert. Bitte Pipeline komplett durchlaufen lassen.")