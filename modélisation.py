import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

print("Création du jeu de données musical...")

# 1. On génère un dataset synthétique (pour démarrer tout de suite !)
# Météo : 0 = Soleil, 1 = Pluie, 2 = Nuages
# Activité : 0 = Code/Études, 1 = Sport, 2 = Détente
# Énergie : de 1 à 10
donnees = {
    'meteo': [0, 1, 2, 0, 1, 2, 0, 1, 1, 0, 2, 1, 0, 2, 1],
    'activite': [1, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 0, 1, 2, 2],
    'energie': [9, 2, 5, 8, 3, 6, 7, 2, 8, 4, 7, 3, 10, 4, 1],
    'genre_recommande': ['Rock/Metal', 'Lo-Fi Chill', 'Synthwave', 'Pop Énergique', 
                         'Musique Classique', 'Synthwave', 'Pop Énergique', 'Lo-Fi Chill', 
                         'Rock/Metal', 'Acoustique', 'Synthwave', 'Lo-Fi Chill', 
                         'Rock/Metal', 'Acoustique', 'Musique Classique']
}

df = pd.DataFrame(donnees)

# 2. Séparation des features (X) et de la cible (y)
X = df[['meteo', 'activite', 'energie']]
y = df['genre_recommande']

# 3. Entraînement du modèle KNN
# On lui demande de regarder les 3 situations les plus proches (n_neighbors=3)
modele = KNeighborsClassifier(n_neighbors=3)
modele.fit(X, y)

# 4. Sauvegarde du modèle
joblib.dump(modele, 'modele_vibe.pkl')
print(" Modèle entraîné et sauvegardé avec succès dans 'modele_vibe.pkl' !")
