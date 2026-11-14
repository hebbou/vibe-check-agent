import streamlit as st
import joblib
import pandas as pd

# Charger le modèle
modele = joblib.load('modele_vibe.pkl')

# Interface utilisateur
st.title("🎧 L'Agent Vibe Check")
st.write("Dis-moi ce que tu vis, je te dirai quoi écouter.")

st.markdown("---")

# Les questions de l'Agent
meteo_choix = st.radio("Quel temps fait-il dehors ?", ["☀️ Soleil", "🌧️ Pluie", "☁️ Nuages"])
activite_choix = st.selectbox("Que fais-tu en ce moment ?", ["👨‍💻 Code / Études", "🏋️ Sport", "🛋️ Détente"])
energie_choix = st.slider("Quel est ton niveau d'énergie actuel ?", 1, 10, 5)

# Dictionnaires pour convertir le texte en nombres pour le modèle
dict_meteo = {"☀️ Soleil": 0, "🌧️ Pluie": 1, "☁️ Nuages": 2}
dict_activite = {"👨‍💻 Code / Études": 0, "🏋️ Sport": 1, "🛋️ Détente": 2}

if st.button("🔮 Scanner ma Vibe"):
    # Préparation des données saisies
    meteo_num = dict_meteo[meteo_choix]
    activite_num = dict_activite[activite_choix]
    
    donnees_user = pd.DataFrame([[meteo_num, activite_num, energie_choix]], 
                                columns=['meteo', 'activite', 'energie'])
    
    # L'Agent fait sa prédiction
    prediction = modele.predict(donnees_user)[0]
    
    st.markdown("---")
    st.subheader("Le verdict de l'Agent :")
    
    # Personnalisation du message de l'Agent selon le résultat
   # Personnalisation du message et ajout du bouton Spotify
    if prediction == 'Lo-Fi Chill':
        st.success("🌧️ Vibe détectée : Calme et focus. Recommandation : Playlist Lo-Fi Chill.")
        st.link_button("🎧 Écouter sur Spotify", "https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0ExPn")
    elif prediction == 'Synthwave':
        st.info("👾 Vibe détectée : Concentration nocturne. Recommandation : Synthwave & Cyberpunk.")
        st.link_button("🎧 Écouter sur Spotify", "https://open.spotify.com/playlist/37i9dQZF1DXdLEN7aqioJC")
    elif prediction == 'Rock/Metal':
        st.error("🔥 Vibe détectée : Adrénaline pure ! Recommandation : Gros Rock ou Metal.")
        st.link_button("🎧 Écouter sur Spotify", "https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U")
    elif prediction == 'Musique Classique':
        st.warning("☕ Vibe détectée : Repos total. Recommandation : Musique Classique.")
        st.link_button("🎧 Écouter sur Spotify", "https://open.spotify.com/playlist/37i9dQZF1DWWEJlAGA9gs0")
    else:
        st.success(f"✨ Vibe détectée : Bonne humeur ! Recommandation : {prediction}.")