import streamlit as st
import plotly.express as px
from datetime import datetime
import json
import requests
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Création d'Indicateur",
    page_icon="📊",
    layout="wide"
)

# Configuration Grist
API_KEY = st.secrets["grist_key"]
DOC_ID = st.secrets["grist_doc_id"]
BASE_URL = "https://grist.numerique.gouv.fr/api/docs"
INDICATORS_TABLE = "Indicators"  # Assurez-vous de créer cette table dans Grist

def grist_api_request(endpoint, method="GET", data=None):
    """Fonction utilitaire pour les requêtes API Grist"""
    url = f"{BASE_URL}/{DOC_ID}/tables"
    if endpoint != "tables":
        url = f"{url}/{endpoint}/records"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        response.raise_for_status()
        return response.json() if response.content else None
    except Exception as e:
        st.error(f"Erreur API Grist : {str(e)}")
        return None

def save_indicator(indicator_data):
    """Sauvegarde un nouvel indicateur dans Grist"""
    try:
        data = {
            "records": [{
                "fields": indicator_data
            }]
        }
        result = grist_api_request(INDICATORS_TABLE, "POST", data)
        return result is not None
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {str(e)}")
        return False

def main():
    st.title("Création d'Indicateur")
    
    # Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Retour à l'analyse"):
        st.switch_page("streamlit_app.py")
    if st.sidebar.button("📊 Catalogue des indicateurs"):
        st.switch_page("pages/indicator_catalog.py")
    
    # Vérification des données d'analyse
    if "indicator_info" not in st.session_state:
        st.warning("Aucune analyse n'a été sélectionnée. Veuillez d'abord effectuer une analyse.")
        if st.button("Retour à l'analyse"):
            st.switch_page("streamlit_app.py")
        return
    
    # Récupération des informations de l'analyse
    info = st.session_state.indicator_info
    st.write("### Analyse sélectionnée")
    
    # Affichage du graphique
    try:
        fig = px.Figure(info["graph_config"]["fig_dict"])
        fig.update_layout(info["graph_config"]["layout"])
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur d'affichage du graphique : {str(e)}")
    
    # Formulaire de création d'indicateur
    with st.form("indicator_form"):
        st.write("### Informations de l'indicateur")
        
        # Informations de base
        nom = st.text_input("Nom de l'indicateur", info.get("titre", ""))
        description = st.text_area("Description", "")
        
        # Source des données
        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source des données", 
                                 ", ".join(info.get("tables_source", [])))
        with col2:
            periodicite = st.selectbox("Périodicité", 
                                     ["Annuelle", "Semestrielle", "Trimestrielle", "Mensuelle"])
        
        # Méthode de calcul
        methode_calcul = st.text_area("Méthode de calcul", 
                                     "Décrivez ici la méthode de calcul...")
        
        # Variables
        variables = []
        if info["type"] == "univarié":
            variables.append(info["variable"])
        else:
            variables.extend([info["variable_x"], info["variable_y"]])
        
        variables_str = st.text_area("Variables utilisées", 
                                   ", ".join(variables))
        
        # Tags
        tags = st.multiselect("Tags", 
                            ["Démographie", "Formation", "Recherche", "Budget", "Performance", 
                             "International", "Ressources humaines", "Infrastructure"])
        
        # Notes et commentaires
        notes = st.text_area("Notes et commentaires", "")
        
        # Bouton de soumission
        submit = st.form_submit_button("💾 Enregistrer l'indicateur")
        
        if submit:
            # Préparation des données
            indicator_data = {
                "nom": nom,
                "description": description,
                "source": source,
                "periodicite": periodicite,
                "methode_calcul": methode_calcul,
                "variables": variables_str,
                "tags": json.dumps(tags),
                "notes": notes,
                "graph_config": json.dumps(info["graph_config"]),
                "data": json.dumps(info["data"]),
                "type_analyse": info["type"],
                "date_creation": datetime.now().isoformat(),
                "date_modification": datetime.now().isoformat(),
                "createur": "data-groov"
            }
            
            # Sauvegarde
            if save_indicator(indicator_data):
                st.success("✅ Indicateur créé avec succès!")
                # Nettoyage des données temporaires
                del st.session_state.indicator_info
                # Redirection vers le catalogue
                st.switch_page("pages/indicator_catalog.py")
            else:
                st.error("Erreur lors de la création de l'indicateur")

if __name__ == "__main__":
    main()
