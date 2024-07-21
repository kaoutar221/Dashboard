import pandas as pd

def load_data(file_path):
    """
    Charge les données à partir d'un fichier CSV.

    :param file_path: Chemin vers le fichier CSV.
    :return: DataFrame contenant les données chargées.
    """
    try:
        df = pd.read_csv(file_path)
        print("Données chargées avec succès.")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None

if __name__ == "__main__":
    file_path = 'data/sampled_df1 (1).csv'  # Assurez-vous que ce chemin est correct
    df = load_data(file_path)

    if df is not None:
        # Afficher les premières lignes des données
        print(df.head())

        # Exemple d'opérations sur les données
        print("\nInformations sur les données :")
        print(df.info())

        print("\nStatistiques descriptives :")
        print(df.describe())

