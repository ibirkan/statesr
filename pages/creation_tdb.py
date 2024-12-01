import streamlit as st
import plotly.express as px
from datetime import datetime
import json
import requests
import time

# Configuration de la page
st.set_page_config(
    page_title="Création Tableau de Bord",
    page_icon="📊",
    layout="wide"
)

# Configuration Grist
API_KEY = st.secrets["grist_key"]
DOC_ID = st.secrets["grist_doc_id"]
BASE_URL = "https://grist.numerique.gouv.fr/api/docs"
DASHBOARDS_TABLE = "Dashboards"

def grist_api_request(endpoint, method="GET", data=None):
    """Fonction utilitaire pour les requêtes API Grist"""
    if endpoint == "tables":
        url = f"{BASE_URL}/{DOC_ID}/tables"
    else:
        url = f"{BASE_URL}/{DOC_ID}/tables/{DASHBOARDS_TABLE}/records"
        if endpoint != "records":
            url = f"{url}/{endpoint}"
    
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

def save_dashboard(dashboard_name, elements, layout=None):
    """Sauvegarde un tableau de bord dans Grist"""
    try:
        if layout is None:
            layout = {"cols_per_row": 2}
            
        data = {
            "records": [{
                "fields": {
                    "name": dashboard_name,
                    "elements": json.dumps(elements),
                    "layout": json.dumps(layout),
                    "created_at": datetime.now().isoformat(),
                    "created_by": "data-groov"
                }
            }]
        }
        result = grist_api_request("records", "POST", data)
        return result is not None
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {str(e)}")
        return False

def load_dashboard_elements():
    """Charge les éléments du tableau de bord depuis la session"""
    if "dashboard_elements" not in st.session_state:
        st.session_state.dashboard_elements = []
    return st.session_state.dashboard_elements

def main():
    st.title("Création de Tableau de Bord")
    
    # Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Retour à l'analyse"):
        st.switch_page("streamlit_app.py")
    if st.sidebar.button("📊 Liste des tableaux de bord"):
        st.switch_page("pages/liste_tdb.py")
    
    # Chargement des éléments
    elements = load_dashboard_elements()
    
    if not elements:
        st.warning("Aucun élément n'a été ajouté au tableau de bord. Retournez à l'analyse pour ajouter des visualisations.")
        if st.button("Retour à l'analyse"):
            st.switch_page("streamlit_app.py")
        return
    
    # Interface de création de tableau de bord
    st.write("### Créer un nouveau tableau de bord")
    
    dashboard_title = st.text_input("Titre du tableau de bord")
    cols_per_row = st.selectbox(
        "Nombre de colonnes par ligne",
        options=[1, 2, 3],
        index=1
    )

    # Affichage des visualisations disponibles
    if elements:
        st.write("### Visualisations disponibles")
        for i in range(0, len(elements), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(elements):
                    element = elements[i + j]
                    with col:
                        st.write(f"#### {element['titre']}")
                        try:
                            fig = px.Figure(element['config']['fig_dict'])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur d'affichage : {str(e)}")

    # Boutons d'action
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Créer le tableau de bord", key="save_dashboard"):
            if dashboard_title:
                if save_dashboard(
                    dashboard_name=dashboard_title,
                    elements=elements,
                    layout={"cols_per_row": cols_per_row}
                ):
                    st.success(f"✅ Tableau de bord '{dashboard_title}' créé avec succès!")
                    st.session_state.dashboard_elements = []  # Réinitialiser les éléments
                    time.sleep(0.5)
                    st.switch_page("pages/liste_tdb.py")
            else:
                st.error("Veuillez entrer un titre pour le tableau de bord")

    with col2:
        if st.button("❌ Annuler", key="cancel"):
            st.session_state.dashboard_elements = []
            st.switch_page("streamlit_app.py")

if __name__ == "__main__":
    main()
