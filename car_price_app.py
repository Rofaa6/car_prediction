# car_price_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Prix de Voitures",
    page_icon="🚗",
    layout="wide"
)

# Titre principal
st.title("🚗 Prédicteur de Prix de Voitures")
st.markdown("**Estimez le prix d'une voiture d'occasion avec l'IA**")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        with open('./random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('./feature_columns.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except:
        st.error("❌ Modèle non trouvé. Vérifiez les fichiers .pkl")
        return None, None

model, feature_columns = load_model()

if model is None:
    st.stop()

# Formulaire de prédiction
st.header("📊 Estimation du Prix")

col1, col2 = st.columns(2)

with col1:
    marque = st.selectbox("Marque", ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Nissan'])
    annee = st.slider("Année", 2000, 2023, 2018)
    kilometrage = st.slider("Kilométrage (km)", 0, 200000, 50000)
    puissance = st.slider("Puissance (ch)", 50, 500, 150)

with col2:
    carburant = st.selectbox("Carburant", ['Essence', 'Diesel', 'Hybride', 'Électrique'])
    transmission = st.radio("Transmission", ['Manuelle', 'Automatique'])
    type_voiture = st.selectbox("Type", ['Berline', 'SUV', 'Compacte', 'Break', 'Coupé'])
    
    if st.button("🎯 Estimer le Prix", type="primary"):
        # Préparation des données
        age = 2023 - annee
        km_par_an = kilometrage / max(age, 1)
        
        input_data = {
            'Année': annee,
            'Kilométrage': kilometrage,
            'Puissance': puissance,
            'Âge': age,
            'Km_Par_An': km_par_an
        }
        
        # Encodage one-hot
        for col in feature_columns:
            if col not in input_data:
                if marque in col or carburant in col or transmission in col or type_voiture in col:
                    input_data[col] = 1
                else:
                    input_data[col] = 0
        
        # Prédiction
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns]
        
        prix_pred = model.predict(input_df)[0]
        
        # Affichage du résultat
        st.success("✅ Estimation terminée !")
        st.metric("Prix estimé", f"{prix_pred:,.0f} €")
        
        st.info(f"""
        **Détails :**
        - 🏷️ {marque} {type_voiture}
        - 📅 {annee} ({age} ans)
        - 🛣️ {kilometrage:,} km
        - ⚡ {puissance} ch
        - ⛽ {carburant}
        - 🔧 {transmission}
        """)

st.markdown("---")
st.caption("Application développée avec Streamlit et Machine Learning")