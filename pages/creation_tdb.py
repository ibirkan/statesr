import streamlit as st
import plotly.express as px
import json
import pandas as pd
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Création Tableau de Bord",
    page_icon="📊",
    layout="wide"
)

# Charger les éléments du tableau de bord depuis la session ou le fichier JSON
def load_dashboard_elements():
    try:
        if "dashboard_elements" in st.session_state:
            return st.session_state.dashboard_elements
        
        with open('dashboard_elements.json', 'r') as f:
            elements = json.load(f).get("dashboards", [])
            st.session_state.dashboard_elements = elements
            return elements
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Erreur lors du chargement du tableau de bord : {str(e)}")
        return []

# Sauvegarder la configuration du tableau de bord
def save_dashboard_config(title, layout, elements):
    try:
        dashboards = load_dashboard_elements()
        new_dashboard = {
            "id": len(dashboards) + 1,
            "title": title,
            "layout": layout,
            "elements": elements,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": "data-groov"
        }
        dashboards.append(new_dashboard)
        with open('dashboards.json', 'w') as f:
            json.dump({"dashboards": dashboards}, f)
        return True
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {str(e)}")
        return False

# Fonction principale
def main():
    st.title("Création de Tableau de Bord")
    
    # Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Retour à l'analyse"):
        st.switch_page("analyse")
    if st.sidebar.button("📊 Liste des tableaux de bord"):
        st.switch_page("liste_tdb")
    
    # Chargement des éléments
    elements = load_dashboard_elements()
    
    if not elements:
        st.warning("Aucun élément n'a été ajouté au tableau de bord. Retournez à l'analyse pour ajouter des visualisations.")
        if st.button("Retour à l'analyse"):
            st.switch_page("analyse")
        return

    # Configuration du tableau de bord
    with st.expander("Configuration du tableau de bord", expanded=True):
        dashboard_title = st.text_input(
            "Titre du tableau de bord",
            "Mon tableau de bord"
        )
        
        cols_per_row = st.selectbox(
            "Nombre de colonnes par ligne",
            options=[1, 2, 3],
            index=1
        )
    
    # Affichage des visualisations
    st.write("### Visualisations disponibles")
    
    # Création des colonnes pour l'affichage
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
                        
                        # Option de suppression pour chaque visualisation
                        if st.button("🗑️ Supprimer", key=f"delete_{i+j}"):
                            elements.pop(i + j)
                            st.session_state.dashboard_elements = elements
                            st.experimental_rerun()
                            
                    except Exception as e:
                        st.error(f"Erreur d'affichage : {str(e)}")

    # Boutons d'action
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Enregistrer le tableau de bord", key="save_dashboard"):
            if save_dashboard_config(
                title=dashboard_title,
                layout={"cols_per_row": cols_per_row},
                elements=elements
            ):
                st.success("✅ Tableau de bord sauvegardé avec succès!")
                st.session_state.dashboard_elements = []  # Réinitialiser les éléments
                st.switch_page("liste_tdb")
    
    with col2:
        if st.button("❌ Annuler", key="cancel"):
            st.session_state.dashboard_elements = []  # Réinitialiser les éléments
            st.switch_page("analyse")

if __name__ == "__main__":
    main()
