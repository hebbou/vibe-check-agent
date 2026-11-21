import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Vibe Check AI", page_icon="🎧", layout="centered")

# ==============================================================================
# STYLE CSS PERSONNALISÉ
# ==============================================================================
st.markdown("""
<style>
.hero-title {
    font-size: 2.7rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #1DB954 0%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    text-align: center;
    color: #94A3B8;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
}
.vibe-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.96));
    border-radius: 18px;
    padding: 1.8rem;
    margin: 1.2rem 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}
.vibe-badge {
    display: inline-block;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.vibe-genre {
    font-size: 2rem;
    font-weight: 800;
    color: #F8FAFC;
    margin: 0.2rem 0 0.6rem 0;
}
.vibe-desc {
    color: #CBD5E1;
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom: 1rem;
}
.vibe-meta {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    font-size: 0.88rem;
    color: #94A3B8;
    border-top: 1px solid rgba(148, 163, 184, 0.18);
    padding-top: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# BLOC 1 : Génération du dataset (1 000 lignes) et entraînement du modèle
# ==============================================================================
@st.cache_resource
def preparer_modele():
    np.random.seed(42)
    n_samples = 1000

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
                if m == 1:
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

    X = df[['meteo', 'activite', 'energie']]
    y = df['genre']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modele = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=9, weights='distance'))
    ])
    modele.fit(X_train, y_train)
    precision = accuracy_score(y_test, modele.predict(X_test))

    return modele, precision, df

modele, precision, df = preparer_modele()

# ==============================================================================
# BLOC 2 : Identité visuelle et playlists par genre
# ==============================================================================
recommandations = {
    'Rock/Metal': {
        'emoji': '🎸',
        'tag': 'Haute Intensité',
        'couleur': '#EF4444',
        'description': "Des riffs puissants et un tempo soutenu pour libérer ton énergie.",
        'lien': "https://open.spotify.com/search/rock%20metal%20workout/playlists"
    },
    'Synthwave': {
        'emoji': '🌆',
        'tag': 'Électro & Rythmé',
        'couleur': '#A855F7',
        'description': "Une ambiance rétro-futuriste et dynamique pour garder le rythme.",
        'lien': "https://open.spotify.com/search/synthwave/playlists"
    },
    'Lo-Fi Chill': {
        'emoji': '☕',
        'tag': 'Focus & Douceur',
        'couleur': '#14B8A6',
        'description': "Des beats doux et réguliers pour te concentrer ou te poser sans distraction.",
        'lien': "https://open.spotify.com/search/lofi%20chill/playlists"
    },
    'Musique Classique': {
        'emoji': '🎻',
        'tag': 'Calme Profond',
        'couleur': '#3B82F6',
        'description': "Des mélodies instrumentales apaisantes idéales pour la concentration profonde.",
        'lien': "https://open.spotify.com/search/musique%20classique%20calme/playlists"
    },
    'Acoustique': {
        'emoji': '🪕',
        'tag': 'Chaleureux & Léger',
        'couleur': '#F59E0B',
        'description': "Des sonorités naturelles à la guitare ou au piano pour un moment apaisant.",
        'lien': "https://open.spotify.com/search/acoustic%20chill/playlists"
    }
}

# ==============================================================================
# BLOC 3 : En-tête et navigation par onglets
# ==============================================================================
st.markdown('<div class="hero-title">🎧 Vibe Check AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Recommandation musicale intelligente propulsée par un modèle KNN</div>',
    unsafe_allow_html=True
)

tab_app, tab_ml = st.tabs(["🎵 Trouver ma Vibe", "📊 Analyse du Modèle"])

# --- ONGLET 1 : APPLICATION ---
with tab_app:
    meteo_options = {"☀️ Soleil": 0, "🌧️ Pluie": 1, "☁️ Nuages": 2}
    activite_options = {"💻 Travail / Études": 0, "🏃 Sport": 1, "🛋️ Détente": 2}

    with st.container(border=True):
        st.markdown("##### Décris ton moment présent")
        col1, col2 = st.columns(2)

        with col1:
            meteo_choix = st.pills(
                "Météo actuelle",
                options=list(meteo_options.keys()),
                default="☀️ Soleil",
                selection_mode="single"
            )
        with col2:
            activite_choix = st.pills(
                "Activité en cours",
                options=list(activite_options.keys()),
                default="💻 Travail / Études",
                selection_mode="single"
            )

        energie = st.slider(
            "Niveau d'énergie",
            min_value=1,
            max_value=10,
            value=5,
            help="1 = très calme / fatigué · 10 = plein d'énergie"
        )

        # Étiquette visuelle dynamique sous le curseur
        if energie <= 3:
            st.caption("🔋 Ambiance calme et posée")
        elif energie <= 7:
            st.caption("⚡ Rythme équilibré")
        else:
            st.caption("🔥 Pleine puissance")

        lancer = st.button("Analyser ma Vibe 🎵", type="primary", use_container_width=True)

    if lancer:
        if not meteo_choix or not activite_choix:
            st.warning("Sélectionne une météo et une activité avant de lancer l'analyse.")
        else:
            entree = pd.DataFrame(
                [[meteo_options[meteo_choix], activite_options[activite_choix], energie]],
                columns=['meteo', 'activite', 'energie']
            )

            probas = modele.predict_proba(entree)[0]
            classes = modele.classes_
            idx_max = int(np.argmax(probas))
            genre_predit = classes[idx_max]
            confiance = probas[idx_max]
            info = recommandations[genre_predit]

            # Carte de recommandation stylisée
            st.markdown(f"""
<div class="vibe-card" style="border: 1px solid {info['couleur']}66; border-left: 6px solid {info['couleur']};">
<span class="vibe-badge" style="background-color: {info['couleur']}26; color: {info['couleur']};">{info['tag']}</span>
<div class="vibe-genre">{info['emoji']} {genre_predit}</div>
<div class="vibe-desc">{info['description']}</div>
<div class="vibe-meta">
<span>🎯 Confiance KNN : <strong>{confiance:.0%}</strong></span>
<span>🌤️ {meteo_choix}</span>
<span>🧭 {activite_choix}</span>
<span>⚡ Énergie : {energie}/10</span>
</div>
</div>
""", unsafe_allow_html=True)

            st.link_button(
                f"Écouter du {genre_predit} sur Spotify 🎧",
                info['lien'],
                type="primary",
                use_container_width=True
            )

            # Détail des probabilités pour montrer le fonctionnement du modèle
            with st.expander("Voir les probabilités calculées par le modèle"):
                df_probas = pd.DataFrame({
                    'Genre': classes,
                    'Probabilité': probas
                }).sort_values('Probabilité', ascending=False)

                for _, row in df_probas.iterrows():
                    st.write(f"**{row['Genre']}** — {row['Probabilité']:.0%}")
                    st.progress(float(row['Probabilité']))

# --- ONGLET 2 : ANALYSE MACHINE LEARNING ---
with tab_ml:
    m1, m2, m3 = st.columns(3)
    m1.metric("Précision (Accuracy)", f"{precision:.1%}")
    m2.metric("Observations", f"{len(df):,}".replace(",", " "))
    m3.metric("Algorithme", "KNN (k=9)")

    st.markdown("##### Répartition des genres dans le dataset d'entraînement")
    st.bar_chart(df['genre'].value_counts(), color="#1DB954")

    st.markdown("##### Aperçu des données générées")
    st.dataframe(df.head(15), use_container_width=True, hide_index=True)