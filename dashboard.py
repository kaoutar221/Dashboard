import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import shap

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Interactif de Prédiction de Risque de Crédit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le pipeline complet
model_path = 'model/best_model_fbeta_gb.pkl'
pipeline = joblib.load(model_path)

# Charger les noms des caractéristiques
feature_names_path = 'model/feature_names.txt'
with open(feature_names_path, 'r') as f:
    feature_names = [line.strip() for line in f]

# Charger les données
data_path = 'data/sampled_df1 (1).csv'  # Assurez-vous que le fichier est dans le même répertoire que ce script
df = pd.read_csv(data_path)

# Assurer que les identifiants clients sont des entiers
df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)

# Limiter aux 50 premiers identifiants clients
unique_client_ids = df['SK_ID_CURR'].unique()
limited_client_ids = unique_client_ids[:50]  # Sélectionner les 50 premiers identifiants

# Définir les couleurs avec un contraste élevé
colors = {
    'all_clients': '#1f77b4',  # Bleu à contraste élevé
    'accepted_clients': '#2ca02c',  # Vert à contraste élevé
    'rejected_clients': '#d62728'  # Rouge à contraste élevé
}

# Définir les définitions des caractéristiques
feature_definitions = {
    "EXT_SOURCE_3": "Score externe source 3",
    "EXT_SOURCE_2": "Score externe source 2",
    "EXT_SOURCE_1": "Score externe source 1",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": "Nombre moyen de retraits au guichet automatique",
    "CC_CNT_DRAWINGS_CURRENT_MAX": "Nombre maximum de retraits actuels",
    "BURO_DAYS_CREDIT_MEAN": "Nombre moyen de jours de crédit",
    "CC_AMT_BALANCE_MEAN": "Solde moyen des comptes de carte de crédit",
    "CC_AMT_TOTAL_RECEIVABLE_MEAN": "Montant total moyen recevable",
    "DAYS_BIRTH": "Nombre de jours depuis la naissance",
    "PREV_NAME_CONTRACT_STATUS_Refused_MEAN": "Moyenne des contrats refusés précédemment",
    "BURO_CREDIT_ACTIVE_Active_MEAN": "Moyenne des crédits actifs",
    "PREV_CODE_REJECT_REASON_XAP_MEAN": "Moyenne des raisons de rejet XAP précédentes",
    "BURO_DAYS_CREDIT_MIN": "Nombre minimum de jours de crédit",
    "BURO_DAYS_CREDIT_UPDATE_MEAN": "Nombre moyen de jours depuis la mise à jour du crédit",
    "DAYS_EMPLOYED_PERC": "Pourcentage de jours employés",
    "PREV_NAME_CONTRACT_STATUS_Approved_MEAN": "Moyenne des contrats approuvés précédemment",
    "CLOSED_DAYS_CREDIT_MIN": "Nombre minimum de jours de crédit fermé",
    "ACTIVE_DAYS_CREDIT_MEAN": "Nombre moyen de jours de crédit actif",
    "TARGET": "Cible (0 = remboursé, 1 = défaut)",
    "SK_ID_CURR": "Identifiant client"
}

# Titre de l'application
st.title("Dashboard Interactif de Prédiction de Risque de Crédit")

# Afficher le tableau des caractéristiques et définitions sur la page d'accueil
st.subheader("Tableau des Caractéristiques et Définitions")
features_df = pd.DataFrame(list(feature_definitions.items()), columns=["Caractéristique", "Définition"])
st.dataframe(features_df, width=800, height=400)

# Utiliser la barre latérale pour la recherche d'identifiant client
st.sidebar.header("Recherche d'Identifiant Client")
client_id = st.sidebar.selectbox(
    "Sélectionnez l'identifiant du client:",
    ["Sélectionner un client"] + list(map(str, limited_client_ids))
)

# Ajouter des boutons pour afficher les informations supplémentaires
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choisissez une option:", ["Informations Personnelles", "Importance des Caractéristiques",
                                                     "Distributions des Caractéristiques", "Analyse Bi-Variée",
                                                     "Autres Analyses"])

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

            selected_features = st.multiselect(
                "Sélectionnez les caractéristiques à afficher:", feature_names, default=feature_names[:5])

            st.write(client_data[selected_features].T)

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
            st.caption(
                "Graphique interactif du score de prédiction. Le graphique montre une jauge avec le score de prédiction du client.")

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
                st.write(
                    "La probabilité de défaut est nettement éloignée du seuil, indiquant une décision plus claire.")

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
            Plus une caractéristique est importante, plus elle influence le résultat de la prédiction. Par exemple, si le modèle décide d'approuver ou de refuser un crédit, les caractéristiques importantes sont celles qui ont le plus contribué à cette décision.
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
                st.caption(
                    "Graphique montrant les caractéristiques les plus importantes pour le modèle. Les barres représentent l'importance relative de chaque caractéristique.")

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
            st.caption(
                "Graphique montrant l'importance des caractéristiques locales pour le client sélectionné. Les valeurs SHAP indiquent comment chaque caractéristique influence la prédiction du modèle.")

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

            # Prédire les probabilités pour tous les clients
            df['probability'] = pipeline.predict_proba(df[feature_names])[:, 1]
            df['prediction'] = df['probability'].apply(lambda x: 'accepté' if x < threshold else 'refusé')

            # Sélectionner des variables pour comparaison
            selected_features = st.multiselect("Sélectionnez des variables pour comparaison:", feature_names,
                                               default=feature_names[:2])

            for selected_feature in selected_features:
                # Ajout des descriptions pour les graphiques
                fig, ax = plt.subplots()

                df_accepted = df[df['prediction'] == 'accepté']
                df_rejected = df[df['prediction'] == 'refusé']

                bins = np.histogram(np.hstack((df_accepted[selected_feature], df_rejected[selected_feature])), bins=30)[
                    1]  # obtenir des bacs cohérents

                ax.hist(df_accepted[selected_feature], bins=bins, alpha=0.7, color=colors['accepted_clients'],
                        label='Clients acceptés')
                ax.hist(df_rejected[selected_feature], bins=bins, alpha=0.7, color=colors['rejected_clients'],
                        label='Clients refusés')
                ax.axvline(client_data[selected_feature].values[0], color='black', linestyle='dashed', linewidth=2,
                           label='Client sélectionné')
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
                    f"Le graphique ci-dessus montre la distribution de la variable '{selected_feature}' pour les clients acceptés (en vert) et refusés (en rouge), comparée à celle du client sélectionné (ligne noire pointillée).")
                st.write(
                    f"Si la ligne noire est proche de la moyenne (ligne centrale de la barre), cela signifie que la valeur de cette caractéristique pour le client est proche de celle de la majorité des autres clients.")
                st.write(
                    f"Des écarts significatifs peuvent indiquer des différences notables par rapport à la moyenne des clients, ce qui peut aider à identifier des particularités ou des risques potentiels.")

                st.pyplot(fig)
                st.caption(
                    "Graphique montrant la distribution de la variable sélectionnée pour les clients acceptés et refusés, ainsi que la position du client sélectionné.")

        elif option == "Analyse Bi-Variée":
            st.subheader("Analyse Bi-Variée")

            # Sélectionner deux variables pour l'analyse bi-variée
            feature_x = st.selectbox("Sélectionnez la première variable (axe X):", feature_names)
            feature_y = st.selectbox("Sélectionnez la deuxième variable (axe Y):", feature_names, index=1)

            # Calculer les scores de prédiction pour toutes les données
            scores = pipeline.predict_proba(df[feature_names])[:, 1]

            # Créer le graphique bi-varié
            fig = px.scatter(
                df, x=feature_x, y=feature_y, color=scores, color_continuous_scale='Viridis',
                labels={'color': 'Score de Prédiction'},
                title=f"Analyse Bi-Variée entre {feature_x} et {feature_y}"
            )

            # Ajouter la position du client sélectionné
            fig.add_trace(go.Scatter(
                x=[client_data[feature_x].values[0]], y=[client_data[feature_y].values[0]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='x'),
                name='Client Sélectionné'
            ))

            st.plotly_chart(fig)
            st.caption(
                "Graphique montrant l'analyse bi-variée entre les deux caractéristiques sélectionnées avec un dégradé de couleur selon le score des clients et le positionnement du client sélectionné.")

            # Explications pour les personnes non expertes en data science
            st.write("""
            **Qu'est-ce que l'analyse bi-variée ?**

            L'analyse bi-variée consiste à examiner la relation entre deux variables. Dans ce contexte, nous cherchons à voir comment deux caractéristiques (comme l'âge et le revenu) 
            sont liées entre elles et comment elles influencent le score de prédiction de risque de crédit.

            **Comment interpréter ce graphique ?**

            - Les points sur le graphique représentent des clients.
            - La couleur des points montre le score de prédiction du risque de crédit, avec une échelle de couleurs allant du vert (risque faible) au rouge (risque élevé).
            - Le point rouge avec un "x" montre la position du client sélectionné sur ces deux caractéristiques.

            **Exemple :**
            Si vous choisissez l'âge (en jours depuis la naissance) sur l'axe X et le revenu sur l'axe Y :
            - Vous pourriez voir que les clients plus âgés avec un revenu plus élevé ont généralement un risque de crédit plus faible (points verts).
            - À l'inverse, les clients plus jeunes avec un revenu plus bas pourraient avoir un risque de crédit plus élevé (points rouges).

            Cette analyse peut aider à comprendre quels facteurs contribuent au risque de crédit et comment ils interagissent entre eux.
            """)

        elif option == "Autres Analyses":
            st.subheader("Autres Analyses des Clients")

            # Distribution des scores de prédiction pour tous les clients
            st.subheader("Distribution des Scores de Prédiction")
            scores = pipeline.predict_proba(df[feature_names])[:, 1]
            fig, ax = plt.subplots()
            ax.hist(scores, bins=30, alpha=0.7, color='blue')
            ax.axvline(probability, color='red', linestyle='dashed', linewidth=2, label='Client sélectionné')
            ax.set_title('Distribution des Scores de Prédiction')
            ax.set_xlabel('Score de Prédiction')
            ax.set_ylabel('Nombre de Clients')
            ax.legend()
            st.pyplot(fig)
            st.caption(
                "Histogramme montrant la distribution des scores de prédiction pour tous les clients avec une ligne indiquant le score du client sélectionné.")

            # Explication simple de l'importance des caractéristiques
            st.write("""
            **Qu'est-ce que l'importance des caractéristiques ?**

            L'importance des caractéristiques indique quelles variables ont le plus d'impact sur la décision du modèle de prédiction. 
            Cela peut aider à comprendre quelles caractéristiques sont les plus influentes pour prédire le risque de crédit.

            **Pourquoi est-ce important ?**

            Connaître les caractéristiques importantes peut aider à orienter les décisions de crédit et à cibler les facteurs de risque les plus critiques. 
            Cela peut également fournir des informations utiles pour améliorer les modèles de prédiction ou pour des stratégies de gestion du risque de crédit.
            """)
