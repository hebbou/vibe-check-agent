import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Configuration de la page (doit toujours être la 1ère commande Streamlit)
st.set_page_config(page_title="Vibe Check AI", page_icon="🎧", layout="centered")

# ==============================================================================
# BLOC 1 : Génération du dataset (1 000 lignes) et entraînement du modèle
# ==============================================================================
@st.cache_resource
def preparer_modele():
    np.random.seed(42)
    n_samples = 1000

    # Variables d'entrée : 0=Soleil, 1=Pluie, 2=Nuages | 0=Travail, 1=Sport, 2=Détente
    meteo = np.random.choice([0, 1, 2], size=n_samples)
    activite = np.random.choice([0, 1, 2], size=n_samples)
    energie = np.random.randint(1, 11, size=n_samples)

    genres = []
    for m, a, e in zip(meteo, activite, energie):
        if a == 1:  # Sport
            if e >= 8:
                g = np.random.choice(['Rock/Metal', 'Synthwave'], p=[0.85, 0.15])
            elif e >= 5:
                g = np.random.choice(['Synthwave', 'Rock/Metal'], p=[0.82, 0.18])
            else:
                g = np.random.choice(['Acoustique', 'Lo-Fi Chill'], p=[0.82, 0.18])
        elif a == 0:  # Travail / Études
            if e <= 4:
                if m == 1:  # Pluie
                    g = np.random.choice(['Lo-Fi Chill', 'Musique Classique'], p=[0.85, 0.15])
                else:
                    g = np.random.choice(['Musique Classique', 'Lo-Fi Chill'], p=[0.85, 0.15])
            elif e <= 7:
                g = np.random.choice(['Lo-Fi Chill', 'Musique Classique', 'Synthwave'], p=[0.80, 0.12, 0.08])
            else:
                g = np.random.choice(['Synthwave', 'Lo-Fi Chill'], p=[0.85, 0.15])
        else:  # Détente
            if e <= 4:
                if m == 1:
                    g = np.random.choice(['Musique Classique', 'Lo-Fi Chill'], p=[0.82, 0.18])
                else:
                    g = np.random.choice(['Lo-Fi Chill', 'Acoustique'], p=[0.82, 0.18])
            elif e <= 7:
                g = np.random.choice(['Acoustique', 'Lo-Fi Chill'], p=[0.85, 0.15])
            else:
                g = np.random.choice(['Synthwave', 'Acoustique'], p=[0.82, 0.18])
        genres.append(g)

    df = pd.DataFrame({
        'meteo': meteo,
        'activite': activite,
        'energie': energie,
        'genre': genres
    })

    # Séparation 80 % entraînement / 20 % test
    X = df[['meteo', 'activite', 'energie']]
    y = df['genre']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline : remise à la même échelle + algorithme KNN
    modele = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=9, weights='distance'))
    ])
    modele.fit(X_train, y_train)

    # Calcul de la précision sur le jeu de test
    precision = accuracy_score(y_test, modele.predict(X_test))

    return modele, precision, df

modele, precision, df = preparer_modele()

# ==============================================================================
# BLOC 2 : Recommandations Spotify associées à chaque genre
# ==============================================================================
recommandations = {
    'Rock/Metal': {
        'description': "Des riffs puissants et un rythme soutenu pour libérer ton énergie.",
        'lien': "https://open.spotify.com/search/rock%20metal%20workout/playlists"
    },
    'Synthwave': {
        'description': "Une ambiance rétro-futuriste et dynamique pour rester dans le rythme.",
        'lien': "https://open.spotify.com/search/synthwave/playlists"
    },
    'Lo-Fi Chill': {
        'description': "Des beats doux et réguliers pour te concentrer ou te poser sans distraction.",
        'lien': "https://open.spotify.com/search/lofi%20chill/playlists"
    },
    'Musique Classique': {
        'description': "Des mélodies instrumentales apaisantes pour le calme et la concentration profonde.",
        'lien': "https://open.spotify.com/search/musique%20classique%20calme/playlists"
    },
    'Acoustique': {
        'description': "Des sons chaleureux à la guitare ou au piano pour accompagner un moment léger.",
        'lien': "https://open.spotify.com/search/acoustic%20chill/playlists"
    }
}

# ==============================================================================
# BLOC 3 : Interface utilisateur Streamlit
# ==============================================================================
st.title("🎧 Vibe Check AI")
st.write("Indique ton contexte actuel : le modèle prédit le genre musical adapté et te propose une sélection Spotify.")

st.divider()

# Choix des paramètres utilisateur
meteo_options = {"☀️ Soleil": 0, "🌧️ Pluie": 1, "☁️ Nuages": 2}
activite_options = {"💻 Travail / Études": 0, "🏃 Sport": 1, "🛋️ Détente": 2}

col1, col2 = st.columns(2)
with col1:
    meteo_choix = st.radio("Quelle météo fait-il ?", list(meteo_options.keys()))
with col2:
    activite_choix = st.radio("Quelle est ton activité ?", list(activite_options.keys()))

energie = st.slider("Quel est ton niveau d'énergie ?", min_value=1, max_value=10, value=5)

# Bouton de prédiction
if st.button("Trouver ma Vibe 🎵", type="primary", use_container_width=True):
    entree = pd.DataFrame(
        [[meteo_options[meteo_choix], activite_options[activite_choix], energie]],
        columns=['meteo', 'activite', 'energie']
    )

    genre_predit = modele.predict(entree)[0]
    confiance = modele.predict_proba(entree)[0].max()
    info = recommandations[genre_predit]

    st.success(f"### Ambiance recommandée : **{genre_predit}**")
    st.write(info['description'])
    st.caption(f"Indice de confiance du modèle KNN : {confiance:.0%}")
    st.link_button(f"Écouter du {genre_predit} sur Spotify 🎧", info['lien'])

st.divider()

# Détails techniques (utile pour valoriser le projet sur un CV)
with st.expander("📊 Voir les performances du modèle et le dataset"):
    st.metric(label="Précision du modèle (Accuracy sur 200 tests)", value=f"{precision:.1%}")
    st.write("Répartition des 1 000 observations synthétiques utilisées pour l'entraînement :")
    st.bar_chart(df['genre'].value_counts())
    st.dataframe(df.head(10), use_container_width=True)