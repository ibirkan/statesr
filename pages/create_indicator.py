import streamlit as st
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Création d'Indicateur",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Création d'Indicateur")
    
    # Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Retour à l'analyse"):
        st.switch_page("streamlit_app.py")
    if st.sidebar.button("📊 Catalogue des indicateurs"):
        st.switch_page("pages/indicator_catalog.py")
    
    # Formulaire simple de création d'indicateur
    with st.form("indicator_form"):
        st.write("### Informations de l'indicateur")
        
        # Informations de base
        nom = st.text_input("Nom de l'indicateur")
        description = st.text_area("Description")
        
        # Source et périodicité
        col1, col2 = st.columns(2)
        with col1:
            source = st.text_input("Source des données")
        with col2:
            periodicite = st.selectbox("Périodicité", 
                                     ["Annuelle", "Semestrielle", "Trimestrielle", "Mensuelle"])
        
        # Tags
        tags = st.multiselect("Tags", 
                            ["Démographie", "Formation", "Recherche", "Budget", "Performance", 
                             "International", "Ressources humaines", "Infrastructure"])
        
        # Bouton de soumission
        submit = st.form_submit_button("💾 Enregistrer l'indicateur")
        
        if submit:
            st.success("✅ Indicateur créé avec succès!")
            st.switch_page("pages/indicator_catalog.py")

if __name__ == "__main__":
    main()
