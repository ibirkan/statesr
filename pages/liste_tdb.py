import streamlit as st
import plotly.express as px
from datetime import datetime
import json
import requests

# Configuration de la page
st.set_page_config(
    page_title="Liste des Tableaux de Bord",
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

def load_dashboards():
    """Charge tous les tableaux de bord depuis Grist"""
    try:
        result = grist_api_request("records", "GET")
        if result and 'records' in result:
            dashboards = []
            for record in result['records']:
                dashboard = {
                    'id': record['id'],
                    'title': record['fields']['name'],
                    'layout': json.loads(record['fields']['layout']),
                    'elements': json.loads(record['fields']['elements']),
                    'created_at': record['fields'].get('created_at', ''),
                    'created_by': record['fields'].get('created_by', 'Utilisateur inconnu')
                }
                dashboards.append(dashboard)
            return dashboards
        return []
    except Exception as e:
        st.error(f"Erreur lors du chargement : {str(e)}")
        return []

def delete_dashboard(dashboard_id):
    """Supprime un tableau de bord de Grist"""
    try:
        result = grist_api_request(str(dashboard_id), "DELETE")
        return result is not None
    except Exception as e:
        st.error(f"Erreur lors de la suppression : {str(e)}")
        return False

def main():
    st.title("Liste des Tableaux de Bord")
    
    # Menu de navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Nouvelle analyse", key="new_analysis"):
        st.switch_page("analyse.py")
    if st.sidebar.button("➕ Créer un tableau de bord", key="create_dashboard"):
        st.switch_page("pages/creation_tdb.py")
    
    # Chargement des tableaux de bord
    dashboards = load_dashboards()
    
    if not dashboards:
        st.warning("Aucun tableau de bord n'a été créé.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Créer un tableau de bord", key="create_dashboard_empty"):
                st.switch_page("pages/creation_tdb.py")
        with col2:
            if st.button("📊 Faire une analyse", key="new_analysis_empty"):
                st.switch_page("analyse.py")
        return
    
    # Affichage des tableaux de bord
    for dashboard in dashboards:
        with st.expander(f"📊 {dashboard['title']}", expanded=True):
            # En-tête du tableau de bord
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"Créé le : {dashboard['created_at']}")
                st.write(f"Créé par : {dashboard.get('created_by', 'Utilisateur inconnu')}")
            with col2:
                if st.button("🗑️ Supprimer", key=f"delete_{dashboard['id']}"):
                    if delete_dashboard(dashboard['id']):
                        st.success("Tableau de bord supprimé!")
                        st.experimental_rerun()
            
            # Affichage des visualisations
            cols_per_row = dashboard['layout']['cols_per_row']
            elements = dashboard['elements']
            
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
            
            st.markdown("---")
