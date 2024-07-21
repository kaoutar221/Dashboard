import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import shap

# Charger le pipeline complet
model_path = 'model/best_model_fbeta_gb.pkl'
pipeline = joblib.load(model_path)

# Charger les noms des caractéristiques
feature_names_path = 'model/feature_names.txt'
with open(feature_names_path, 'r') as f:
    feature_names = [line.strip() for line in f]

# Charger les données
data_path = 'data/sampled_df1 (1).csv'
df = pd.read_csv(data_path)

# Assurer que les identifiants clients sont des entiers
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

# Limiter aux 10 premiers identifiants clients
unique_client_ids = df['SK_ID_CURR'].unique()
limited_client_ids = unique_client_ids[:10]  # Sélectionner les 10 premiers identifiants

# Définir les couleurs avec un contraste élevé
colors = {
    'all_clients': '#1f77b4',  # Blue with high contrast
    'selected_client': '#d62728'  # Red with high contrast
}

# Titre de l'application
st.title("Dashboard Interactif de Prédiction de Risque de Crédit")

# Utiliser la barre latérale pour la recherche d'identifiant client
st.sidebar.header("Recherche d'Identifiant Client")
client_id = st.sidebar.selectbox(
    "Sélectionnez l'identifiant du client:",
    ["Sélectionner un client"] + list(map(str, limited_client_ids))
)

# Ajouter des boutons pour afficher les informations supplémentaires
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choisissez une option:", ["Informations Personnelles", "Importance des Caractéristiques", "Distributions des Caractéristiques"])

threshold = 0.45  # Fixer le seuil à 0.45

# Afficher les résultats dans la page principale
if client_id and client_id != "Sélectionner un client":
    client_id = int(client_id)
    if client_id not in df['SK_ID_CURR'].values:
        st.error("Client ID n'est pas dans le dataset")
    else:
        # Sélectionner les données du client
        client_data = df[df['SK_ID_CURR'] == client_id][feature_names]

        # Prédire la probabilité de défaut en utilisant le pipeline complet
        probability = pipeline.predict_proba(client_data)[:, 1][0]
        prediction = 'accepté' if probability < threshold else 'refusé'

        # Afficher le résultat dans une case colorée
        if prediction == 'accepté':
            st.markdown(
                f'<div style="padding: 10px; background-color: lightgreen; color: black; text-align: center; border-radius: 5px;">'
                f'<strong>Client {client_id} est accepté</strong></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="padding: 10px; background-color: lightcoral; color: black; text-align: center; border-radius: 5px;">'
                f'<strong>Client {client_id} est refusé</strong></div>',
                unsafe_allow_html=True
            )

        if option == "Informations Personnelles":
            # Afficher les informations descriptives du client
            st.subheader("Informations Descriptives du Client")
            st.write(client_data)

            # Ajout de l'interprétation du score
            st.subheader("Score de Prédiction")

            # Afficher le graphique compteur interactif
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability,
                title={'text': "Score de Prédiction"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, threshold], 'color': "lightgreen"},
                        {'range': [threshold, 1], 'color': "lightcoral"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold}}))

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Graphique interactif du score de prédiction. Le graphique montre une jauge avec le score de prédiction du client.")

            # Interprétation pour une personne non experte
            if probability < threshold:
                st.write(
                    f"Le modèle estime que le risque de défaut de paiement est faible (Probabilité de défaut: {probability:.2f}).")
            else:
                st.write(
                    f"Le modèle estime que le risque de défaut de paiement est élevé (Probabilité de défaut: {probability:.2f}).")

            # Indication sur la distance par rapport au seuil
            if abs(probability - threshold) < 0.1:
                st.write(
                    "La probabilité de défaut est proche du seuil. Cela signifie que le client est dans une zone grise et nécessite peut-être une analyse plus approfondie.")
            else:
                st.write("La probabilité de défaut est nettement éloignée du seuil, indiquant une décision plus claire.")

            # Explication sur les risques de crédit
            st.write("""
            **Qu'est-ce que le risque de crédit ?**

            Le risque de crédit est la probabilité que l'emprunteur ne puisse pas rembourser un prêt. Cela peut être influencé par plusieurs facteurs,
            tels que les antécédents de crédit, le revenu, l'emploi, les dettes actuelles et d'autres critères financiers. Un risque de crédit élevé signifie qu'il est plus probable que le client ne rembourse pas le prêt, 
            ce qui peut entraîner des pertes financières pour le prêteur. D'où l'importance de prédire et de gérer ces risques pour les institutions financières.
            """)

        elif option == "Importance des Caractéristiques":
            st.subheader("Importance des Caractéristiques Globales")

            # Explication simple de l'importance des caractéristiques
            st.write("""
            **Qu'est-ce que l'importance des caractéristiques ?**

            L'importance des caractéristiques est une mesure qui montre quelles caractéristiques (ou variables) ont le plus d'impact sur les décisions du modèle de prédiction. 
            Plus une caractéristique est importante, plus elle influence le résultat de la prédiction. Par exemple, si le modèle décide d'approuver ou de refuser un crédit, 
            les caractéristiques importantes sont celles qui ont le plus contribué à cette décision.
            """)

            # Permettre à l'utilisateur de choisir le nombre de caractéristiques à afficher
            num_features = st.slider("Nombre de caractéristiques à afficher:", 1, len(feature_names), 10)

            # Calculer l'importance des caractéristiques
            try:
                importances = pipeline.named_steps['classifier'].feature_importances_
                indices = np.argsort(importances)[::-1][:num_features]
                selected_features = [feature_names[i] for i in indices]
                selected_importances = importances[indices]

                # Afficher un graphique en barres de l'importance des caractéristiques
                fig = go.Figure([go.Bar(x=selected_features, y=selected_importances)])
                fig.update_layout(title="Importance des Caractéristiques Globales",
                                  xaxis_title="Caractéristiques",
                                  yaxis_title="Importance",
                                  yaxis=dict(range=[0, max(selected_importances) * 1.1]))

                st.plotly_chart(fig)
                st.caption("Graphique montrant les caractéristiques les plus importantes pour le modèle. Les barres représentent l'importance relative de chaque caractéristique.")

            except AttributeError:
                st.error("Le modèle sélectionné ne supporte pas l'attribut 'feature_importances_'.")

            st.subheader("Importance des Caractéristiques Locales pour le Client Sélectionné")

            # Permettre à l'utilisateur de choisir le nombre de caractéristiques locales à afficher
            num_local_features = st.slider("Nombre de caractéristiques locales à afficher:", 1, len(feature_names), 10)

            # Calculer les valeurs SHAP pour le client sélectionné
            explainer = shap.Explainer(pipeline.named_steps['classifier'], df[feature_names])
            shap_values = explainer(client_data[feature_names])

            # Afficher le graphique de l'importance des caractéristiques locales
            st.write(
                "Les valeurs SHAP montrent l'impact de chaque caractéristique sur la prédiction du modèle pour le client sélectionné.")

            # Plot SHAP waterfall plot for the individual
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.waterfall_plot(shap_values[0], max_display=num_local_features, show=False)
            st.pyplot(fig)
            st.caption("Graphique montrant l'importance des caractéristiques locales pour le client sélectionné. Les valeurs SHAP indiquent comment chaque caractéristique influence la prédiction du modèle.")

            # Ajouter une interprétation des résultats locaux
            st.write("""
            **Interprétation des caractéristiques locales :**

            Le graphique ci-dessus montre l'impact de chaque caractéristique sur la prédiction pour le client sélectionné. 
            Les caractéristiques en haut du graphique ont le plus grand impact, positif ou négatif, sur le résultat final.

            - **Caractéristiques positives (rouge)** : Ces caractéristiques augmentent la probabilité de défaut du client.
            - **Caractéristiques négatives (bleu)** : Ces caractéristiques réduisent la probabilité de défaut du client.

            Par exemple, si le revenu du client est beaucoup plus bas que la moyenne, cela pourrait augmenter son risque de défaut, et donc apparaître comme une caractéristique rouge.
            À l'inverse, une longue durée d'emploi stable pourrait réduire le risque de défaut, apparaissant comme une caractéristique bleue.
            """)

        elif option == "Distributions des Caractéristiques":
            st.subheader("Comparaison des Informations Descriptives")

            # Sélectionner des variables pour comparaison
            selected_feature = st.selectbox("Sélectionnez une variable pour comparaison:", feature_names)

            # Ajout des descriptions pour les graphiques
            fig, ax = plt.subplots()
            df[selected_feature].hist(ax=ax, bins=30, alpha=0.5, color=colors['all_clients'], label='Tous les clients')
            ax.axvline(client_data[selected_feature].values[0], color=colors['selected_client'], linestyle='dashed',
                       linewidth=2, label='Client sélectionné')
            ax.set_title(f"Distribution de {selected_feature}")
            ax.legend()
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Nombre de clients')

            # Ajouter les statistiques descriptives
            mean_val = df[selected_feature].mean()
            median_val = df[selected_feature].median()
            std_val = df[selected_feature].std()

            st.write(f"Moyenne de {selected_feature} pour tous les clients : {mean_val:.2f}")
            st.write(f"Médiane de {selected_feature} pour tous les clients : {median_val:.2f}")
            st.write(f"Écart-type de {selected_feature} pour tous les clients : {std_val:.2f}")

            # Ajouter une interprétation simple du graphique
            st.write(f"**Interprétation du graphique :**")
            st.write(
                f"Le graphique ci-dessus montre la distribution de la variable '{selected_feature}' pour tous les clients comparée à celle du client sélectionné.")
            st.write(
                f"La barre bleue représente la fréquence des valeurs de '{selected_feature}' pour tous les clients, tandis que la ligne rouge pointillée montre la valeur de cette variable pour le client sélectionné.")
            st.write(
                f"Si la ligne rouge est proche de la moyenne (ligne centrale de la barre bleue), cela signifie que la valeur de cette caractéristique pour le client est proche de celle de la majorité des autres clients.")
            st.write(
                f"Des écarts significatifs peuvent indiquer des différences notables par rapport à la moyenne des clients, ce qui peut aider à identifier des particularités ou des risques potentiels.")

            st.pyplot(fig)
            st.caption(
                "Graphique montrant la distribution de la variable sélectionnée pour tous les clients et la position du client sélectionné.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8501))
    st.run(port=port)
