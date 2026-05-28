import streamlit as st
import pandas as pd
import os
from PIL import Image

def render(model):
    features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"feature_{i}" for i in range(model.n_features_in_)]
    st.header("Case Prediction (Live)")
    c_in, c_out = st.columns([1, 2])
    
    with c_in:
        st.subheader("Input")
        input_data = {}
        amt = st.number_input("Fine amount (€)", min_value=0.0, value=35.0)
        for f in features:
            if f == 'amount': input_data[f] = amt
            elif f == 'Payment': input_data[f] = 0
            else:
                input_data[f] = 1 if st.checkbox(f"Activity: {f}") else 0
        predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    with c_out:
        if predict_clicked:
            test_df = pd.DataFrame([input_data], columns=features)
            pred = model.predict(test_df)[0]
            prob = model.predict_proba(test_df)[0][1]
            
            st.subheader("Prediction Result")
            if pred == 1:
                st.error(f"🚨 **High collection risk ({prob*100:.1f}%)**")
            else:
                st.success(f"✅ **Payment likely (risk: {prob*100:.1f}%)**")
            
            st.divider()
            st.subheader("🛠️ Recommendation (Prescriptive)")
            st.markdown("Based on the predicted risk, the system recommends next steps:")
            
            if prob >= 0.80:
                st.error("**Action level RED:** Proactively offer the citizen a **payment plan** immediately.")
            elif prob >= 0.50:
                st.warning("**Action level YELLOW:** Prioritize this case. Send a manual **warning letter**.")
            else:
                st.success("**Action level GREEN:** No intervention needed. Continue standard workflow.")

            st.divider()
            st.subheader("Explainability (SHAP)")
            shap_img_path = 'outputs/plots/shap_k2.png'
            if os.path.exists(shap_img_path):
                st.image(Image.open(shap_img_path), use_container_width=True)