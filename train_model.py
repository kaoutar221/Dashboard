import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les données
file_path = 'data/sampled_df1 (1).csv'
df = pd.read_csv(file_path)

# Séparer les features et la cible
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# Définir le pipeline pour GradientBoostingClassifier
gb_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE()),
    ('classifier', GradientBoostingClassifier())
])

# Définir les paramètres pour la recherche aléatoire pour GradientBoostingClassifier
gb_param_dist = {
    'classifier__learning_rate': [0.01, 0.05],  # Taux d'apprentissage plus élevé
    'classifier__max_depth': [3, 5, 7],  # Profondeur plus élevée
    'classifier__n_estimators': [50, 100],  # Plus d'estimators
    'classifier__min_samples_split': [2, 10],  # Nombre minimum d'échantillons pour diviser un nœud
    'classifier__min_samples_leaf': [1, 5],  # Nombre minimum d'échantillons dans une feuille
    'classifier__subsample': [0.8, 0.9, 1.0]  # Sous-échantillonnage
}

# Définir les scorers
fbeta_scorer = make_scorer(fbeta_score, beta=10, average='macro')
roc_auc_scorer = 'roc_auc'

# Créer la recherche aléatoire pour F-beta
gb_random_search_fbeta = RandomizedSearchCV(estimator=gb_pipeline,
                                            param_distributions=gb_param_dist,
                                            scoring=fbeta_scorer,
                                            refit=True,
                                            cv=3,  # Utiliser moins de folds pour accélérer
                                            n_jobs=-1,
                                            verbose=1,
                                            n_iter=20,  # Limiter le nombre d'itérations
                                            random_state=42)

# Créer la recherche aléatoire pour AUC ROC
gb_random_search_roc_auc = RandomizedSearchCV(estimator=gb_pipeline,
                                              param_distributions=gb_param_dist,
                                              scoring=roc_auc_scorer,
                                              refit=True,
                                              cv=3,  # Utiliser moins de folds pour accélérer
                                              n_jobs=-1,
                                              verbose=1,
                                              n_iter=20,  # Limiter le nombre d'itérations
                                              random_state=42)

# Adapter chaque RandomizedSearchCV aux données d'entraînement
gb_random_search_fbeta.fit(X, y)
gb_random_search_roc_auc.fit(X, y)

# Récupérer les meilleurs paramètres et les meilleurs modèles
best_model_fbeta_gb = gb_random_search_fbeta.best_estimator_
best_model_roc_auc_gb = gb_random_search_roc_auc.best_estimator_

# Sauvegarder le meilleur modèle
model_path = 'model/best_model_fbeta_gb.pkl'
joblib.dump(best_model_fbeta_gb, model_path)

# Sauvegarder les noms des caractéristiques
feature_names_path = 'model/feature_names.txt'
with open(feature_names_path, 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")
