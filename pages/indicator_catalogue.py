import streamlit as st
import plotly.express as px
from datetime import datetime
import json
import requests
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title="Catalogue des Indicateurs",
    page_icon="📊",
    layout="wide"
)

# Configuration Grist
API_KEY = st.secrets["grist_key"]
DOC_ID = st.secrets["grist_doc_id"]
BASE_URL = "https://grist.numerique.gouv.fr/api/docs"
INDICATORS_TABLE = "Indicators"

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

def load_indicators():
    """Charge tous les indicateurs depuis Grist"""
    try:
        result = grist_api_request(INDICATORS_TABLE)
        if result and 'records' in result:
            indicators = []
            for record in result['records']:
                indicator = {k.lstrip('$'): v for k, v in record['fields'].items()}
                indicator['id'] = record['id']
                # Conversion des chaînes JSON en objets Python
                try:
                    indicator['tags'] = json.loads(indicator.get('tags', '[]'))
                    indicator['graph_config'] = json.loads(indicator.get('graph_config', '{}'))
                    indicator['data'] = json.loads(indicator.get('data', '[]'))
                except:
                    pass
                indicators.append(indicator)
            return indicators
        return []
    except Exception as e:
        st.error(f"Erreur lors du chargement : {str(e)}")
        return []

def delete_indicator(indicator_id):
    """Supprime un indicateur"""
    try:
        result = grist_api_request(f"{INDICATORS_TABLE}/{indicator_id}", "DELETE")
        return result is not None
    except Exception as e:
        st.error(f"Erreur lors de la suppression : {str(e)}")
        return False

def main():
    st.title("Catalogue des Indicateurs")
    
    # Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Nouvelle analyse"):
        st.switch_page("streamlit_app.py")
    if st.sidebar.button("➕ Créer un indicateur"):
        st.switch_page("pages/create_indicator.py")
    
    # Filtres dans la barre latérale
    st.sidebar.title("Filtres")
    
    # Liste de tous les tags uniques
    indicators = load_indicators()
    all_tags = set()
    for ind in indicators:
        all_tags.update(ind.get('tags', []))
    
    # Filtres
    selected_tags = st.sidebar.multiselect(
        "Filtrer par tags",
        list(all_tags)
    )
    
    search_term = st.sidebar.text_input("Rechercher un indicateur")
    
    # Filtrage des indicateurs
    filtered_indicators = indicators
    if selected_tags:
        filtered_indicators = [
            ind for ind in filtered_indicators
            if any(tag in selected_tags for tag in ind.get('tags', []))
        ]
    
    if search_term:
        search_term = search_term.lower()
        filtered_indicators = [
            ind for ind in filtered_indicators
            if search_term in ind.get('name', '').lower() or
               search_term in ind.get('description', '').lower()
        ]
    
    # Affichage des indicateurs
    if not filtered_indicators:
        st.warning("Aucun indicateur trouvé.")
        return
    
    # Organisation par tags
    if selected_tags:
        for tag in selected_tags:
            st.write(f"## {tag}")
            tag_indicators = [ind for ind in filtered_indicators if tag in ind.get('tags', [])]
            for indicator in tag_indicators:
                with st.expander(f"📊 {indicator['name']}", expanded=False):
                    # Métadonnées
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("**Description:**", indicator.get('description', ''))
                        st.write("**Source:**", indicator.get('source', ''))
                        st.write("**Variables:**", indicator.get('variables', ''))
                        st.write("**Méthode de calcul:**", indicator.get('methode_calcul', ''))
                    with col2:
                        st.write("**Périodicité:**", indicator.get('periodicite', ''))
                        st.write("**Créé le:**", indicator.get('created_at', '').split('T')[0])
                        st.write("**Tags:**", ", ".join(indicator.get('tags', [])))
                    
                    # Graphique
                    if indicator.get('graph_config'):
                        try:
                            fig = px.Figure(indicator['graph_config'])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Erreur d'affichage du graphique : {str(e)}")
                    
                    # Notes
                    if indicator.get('notes'):
                        st.write("**Notes:**", indicator['notes'])
                    
                    # Option de suppression
                    if st.button("🗑️ Supprimer", key=f"delete_{indicator['id']}"):
                        if delete_indicator(indicator['id']):
                            st.success("Indicateur supprimé!")
                            st.experimental_rerun()
    else:
        # Affichage simple sans organisation par tags
        for indicator in filtered_indicators:
            with st.expander(f"📊 {indicator['name']}", expanded=False):
                # Même contenu que ci-dessus
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write("**Description:**", indicator.get('description', ''))
                    st.write("**Source:**", indicator.get('source', ''))
                    st.write("**Variables:**", indicator.get('variables', ''))
                    st.write("**Méthode de calcul:**", indicator.get('methode_calcul', ''))
                with col2:
                    st.write("**Périodicité:**", indicator.get('periodicite', ''))
                    st.write("**Créé le:**", indicator.get('created_at', '').split('T')[0])
                    st.write("**Tags:**", ", ".join(indicator.get('tags', [])))
                
                # Graphique
                if indicator.get('graph_config'):
                    try:
                        fig = px.Figure(indicator['graph_config'])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erreur d'affichage du graphique : {str(e)}")
                
                # Notes
                if indicator.get('notes'):
                    st.write("**Notes:**", indicator['notes'])
                
                # Option de suppression
                if st.button("🗑️ Supprimer", key=f"delete_{indicator['id']}"):
                    if delete_indicator(indicator['id']):
                        st.success("Indicateur supprimé!")
                        st.experimental_rerun()

if __name__ == "__main__":
    main()
