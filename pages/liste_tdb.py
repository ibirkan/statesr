import streamlit as st
import plotly.express as px
import json
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Liste des Tableaux de Bord",
    page_icon="📊",
    layout="wide"
)

def load_dashboards():
    """Charge tous les tableaux de bord sauvegardés."""
    try:
        with open('dashboards.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Erreur lors du chargement des tableaux de bord : {str(e)}")
        return []

def delete_dashboard(dashboard_id):
    """Supprime un tableau de bord spécifique."""
    try:
        dashboards = load_dashboards()
        dashboards = [d for d in dashboards if d['id'] != dashboard_id]
        
        with open('dashboards.json', 'w') as f:
            json.dump(dashboards, f)
        return True
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

if __name__ == "__main__":
    main()
