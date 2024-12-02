import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Catalogue des Indicateurs",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Catalogue des Indicateurs")
    
    # Navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("🔄 Nouvelle analyse"):
        st.switch_page("streamlit_app.py")
    if st.sidebar.button("➕ Créer un indicateur"):
        st.switch_page("pages/create_indicator.py")
    
    # Contenu temporaire
    st.info("Le catalogue des indicateurs est en cours de développement.")
    
    # Exemple de carte d'indicateur statique
    with st.expander("📊 Exemple d'indicateur", expanded=True):
        st.write("**Nom**: Taux de réussite en licence")
        st.write("**Description**: Mesure le pourcentage d'étudiants obtenant leur licence en 3 ou 4 ans")
        st.write("**Source**: SIES")
        st.write("**Périodicité**: Annuelle")
        st.write("**Tags**: Formation, Performance")

if __name__ == "__main__":
    main()
