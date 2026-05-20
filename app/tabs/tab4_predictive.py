import streamlit as st
import pandas as pd
import os
from PIL import Image

def render(model, features):
    st.header("Fall-Vorhersage (Live)")
    c_in, c_out = st.columns([1, 2])
    
    with c_in:
        st.subheader("Eingabe")
        input_data = {}
        amt = st.number_input("Bußgeldhöhe (€)", min_value=0.0, value=35.0)
        for f in features:
            if f == 'amount': input_data[f] = amt
            elif f == 'Payment': input_data[f] = 0
            else:
                input_data[f] = 1 if st.checkbox(f"Aktivität: {f}") else 0
        predict_clicked = st.button("Prognose erstellen", type="primary", use_container_width=True)

    with c_out:
        if predict_clicked:
            test_df = pd.DataFrame([input_data], columns=features)
            pred = model.predict(test_df)[0]
            prob = model.predict_proba(test_df)[0][1]
            
            st.subheader("Analyseergebnis (Predictive)")
            if pred == 1:
                st.error(f"🚨 **Hohes Inkasso-Risiko ({prob*100:.1f}%)**")
            else:
                st.success(f"✅ **Zahlung wahrscheinlich (Risiko: {prob*100:.1f}%)**")
            
            st.divider()
            st.subheader("🛠️ Handlungsempfehlung (Prescriptive)")
            st.markdown("Basierend auf dem vorhergesagten Risiko empfiehlt das System nächste Schritte:")
            
            if prob >= 0.80:
                st.error("**Aktionsebene Rot:** Bieten Sie dem Bürger sofort aktiv eine **Ratenzahlung** an.")
            elif prob >= 0.50:
                st.warning("**Aktionsebene Gelb:** Priorisieren Sie diesen Fall. Versenden Sie manuell ein **Warnschreiben**.")
            else:
                st.success("**Aktionsebene Grün:** Keine Intervention nötig. Standard-Workflow beibehalten.")

            st.divider()
            st.subheader("Erklärbarkeit (SHAP)")
            shap_img_path = 'models/shap_summary.png'
            if os.path.exists(shap_img_path):
                st.image(Image.open(shap_img_path), use_container_width=True)