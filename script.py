# car_price_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Prix de Voitures",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🚗 Prédicteur de Prix de Voitures")
st.markdown("""
**Utilisez l'intelligence artificielle pour estimer le prix d'une voiture d'occasion**
""")

# Sidebar avec informations
st.sidebar.header("ℹ️ À propos")
st.sidebar.info("""
Cette application utilise un modèle de Machine Learning (Random Forest) 
entraîné pour prédire le prix des voitures d'occasion en fonction de leurs caractéristiques.

**Fonctionnalités :**
- Prédiction en temps réel
- Interface intuitive
- Résultats détaillés
""")

# Vérification de l'existence du modèle
@st.cache_resource
def load_model():
    """Charge le modèle entraîné"""
    try:
        with open('saved_models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('saved_models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return model, feature_columns
    except FileNotFoundError:
        st.error("❌ Fichiers du modèle non trouvés. Assurez-vous que les fichiers .pkl sont dans le dossier 'saved_models/'")
        return None, None

# Chargement du modèle
model, feature_columns = load_model()

if model is None:
    st.stop()

# Section de prédiction
st.header("📊 Estimation du Prix")

# Création des colonnes pour l'interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Caractéristiques de la Voiture")
    
    # Formulaire de saisie
    with st.form("car_form"):
        # Marque
        marque = st.selectbox(
            "Marque 🏷️",
            ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Nissan']
        )
        
        # Année
        annee = st.slider(
            "Année de fabrication 📅",
            min_value=2000,
            max_value=2023,
            value=2018,
            step=1
        )
        
        # Kilométrage
        kilometrage = st.slider(
            "Kilométrage 🛣️ (km)",
            min_value=0,
            max_value=200000,
            value=50000,
            step=1000
        )
        
        # Puissance
        puissance = st.slider(
            "Puissance du moteur ⚡ (ch)",
            min_value=50,
            max_value=500,
            value=150,
            step=10
        )
        
        # Carburant
        carburant = st.selectbox(
            "Type de carburant ⛽",
            ['Essence', 'Diesel', 'Hybride', 'Électrique']
        )
        
        # Transmission
        transmission = st.radio(
            "Type de transmission 🔧",
            ['Manuelle', 'Automatique']
        )
        
        # Type de voiture
        type_voiture = st.selectbox(
            "Type de carrosserie 🚙",
            ['Berline', 'SUV', 'Compacte', 'Break', 'Coupé']
        )
        
        # Bouton de prédiction
        submitted = st.form_submit_button("🎯 Estimer le Prix")

with col2:
    st.subheader("Résultats de l'Estimation")
    
    if submitted:
        # Calcul des features dérivées
        age = 2023 - annee
        km_par_an = kilometrage / max(age, 1)
        
        # Création des données d'entrée
        input_data = {
            'Année': annee,
            'Kilométrage': kilometrage,
            'Puissance': puissance,
            'Âge': age,
            'Km_Par_An': km_par_an
        }
        
        # Ajout des colonnes one-hot encodées
        for col in feature_columns:
            if col not in input_data:
                # Vérifier si cette colonne correspond aux caractéristiques sélectionnées
                if marque in col or carburant in col or transmission in col or type_voiture in col:
                    input_data[col] = 1
                else:
                    input_data[col] = 0
        
        # Création du DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_columns]  # Assurer le bon ordre
        
        try:
            # Prédiction
            prix_pred = model.predict(input_df)[0]
            
            # Affichage des résultats
            st.success("✅ Estimation terminée avec succès!")
            
            # Carte de résultat
            st.metric(
                label="**Prix estimé**",
                value=f"{prix_pred:,.0f} €",
                delta=None
            )
            
            # Détails de la prédiction
            st.info(f"""
            **Détails de l'estimation :**
            - 🏷️ **Marque** : {marque}
            - 📅 **Année** : {annee} (Âge : {age} ans)
            - 🛣️ **Kilométrage** : {kilometrage:,} km
            - ⚡ **Puissance** : {puissance} ch
            - ⛽ **Carburant** : {carburant}
            - 🔧 **Transmission** : {transmission}
            - 🚙 **Type** : {type_voiture}
            """)
            
            # Barre de confiance (simulée)
            confidence = min(95, max(70, 100 - (age * 0.5 + kilometrage / 10000)))
            st.progress(int(confidence))
            st.caption(f"Niveau de confiance de l'estimation : {confidence:.0f}%")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
    
    else:
        # Message d'attente
        st.info("👆 Remplissez le formulaire et cliquez sur 'Estimer le Prix' pour obtenir une estimation")
        
        # Exemple de prédiction
        st.subheader("💡 Exemple de configuration typique")
        st.code("""
        Toyota - 2018 - 50,000 km - 150 ch
        Essence - Automatique - Berline
        → Prix estimé : ~15,000-20,000 €
        """)

# Section d'analyse détaillée
st.header("📈 Analyse et Statistiques")

tab1, tab2, tab3 = st.tabs(["📊 Facteurs d'Influence", "🔍 Conseils", "📋 Historique"])

with tab1:
    st.subheader("Facteurs influençant le prix")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Âge du véhicule", "Impact élevé", delta="-15% par an")
        st.metric("Kilométrage", "Impact élevé", delta="-10% par 20,000 km")
    
    with col2:
        st.metric("Marque", "Impact moyen", delta="±30% selon marque")
        st.metric("Puissance", "Impact moyen", delta="+5% par 50 ch")
    
    with col3:
        st.metric("Carburant", "Impact moyen", delta="Diesel > Essence")
        st.metric("Transmission", "Impact faible", delta="Auto +5%")

with tab2:
    st.subheader("Conseils pour une bonne estimation")
    
    tips = [
        "✅ Vérifiez l'état général du véhicule",
        "✅ Considérez l'entretien régulier",
        "✅ Tenez compte des options supplémentaires",
        "✅ Comparez avec des annonces similaires",
        "⚠️ Les accidents réduisent significativement la valeur",
        "⚠️ La couleur peut influencer le prix de 5-10%"
    ]
    
    for tip in tips:
        st.write(tip)

with tab3:
    st.subheader("Simulation d'historique de prédictions")
    
    # Exemple de données historiques
    historique_data = {
        'Date': ['2024-01-15', '2024-01-10', '2024-01-05'],
        'Modèle': ['Toyota Corolla', 'BMW X3', 'Ford Focus'],
        'Prix Estimé': [18500, 32500, 12500],
        'Prix Réel': [18000, 33000, 12000]
    }
    
    df_hist = pd.DataFrame(historique_data)
    df_hist['Différence'] = df_hist['Prix Estimé'] - df_hist['Prix Réel']
    df_hist['Précision'] = (1 - abs(df_hist['Différence']) / df_hist['Prix Réel']) * 100
    
    st.dataframe(df_hist.style.format({
        'Prix Estimé': '{:,.0f} €',
        'Prix Réel': '{:,.0f} €', 
        'Différence': '{:,.0f} €',
        'Précision': '{:.1f} %'
    }))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚗 <strong>Car Price Predictor</strong> - Powered by Machine Learning</p>
    <p><small>Modèle Random Forest entraîné sur données synthétiques</small></p>
</div>
""", unsafe_allow_html=True)
