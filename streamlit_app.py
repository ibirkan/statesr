import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import json

# Configuration de la page (reste inchangé)
st.set_page_config(
    page_title="Indicateurs ESR",
    page_icon="📊",
    layout="wide"
)

# Palettes de couleurs prédéfinies (reste inchangé)
COLOR_PALETTES = {
    "Bleu": ['#C6DBEF', '#9ECAE1', '#6BAED6', '#4292C6', '#2171B5', '#084594'],
    "Vert": ['#C7E9C0', '#A1D99B', '#74C476', '#41AB5D', '#238B45', '#005A32'],
    "Rouge": ['#FEE5D9', '#FCBBA1', '#FC9272', '#FB6A4A', '#DE2D26', '#A50F15'],
    "Orange": ['#FEE6CE', '#FDD0A2', '#FDAE6B', '#FD8D3C', '#F16913', '#D94801'],
    "Violet": ['#EFEDF5', '#DADAEB', '#BCBDDC', '#9E9AC8', '#807DBA', '#6A51A3'],
    "Gris": ['#F7F7F7', '#D9D9D9', '#BDBDBD', '#969696', '#737373', '#525252']
}

# Configuration Grist
API_KEY = st.secrets["grist_key"]
DOC_ID = st.secrets["grist_doc_id"]
BASE_URL = "https://grist.numerique.gouv.fr/api/docs"
DASHBOARDS_TABLE = "Dashboards"  # Nom de la table pour les tableaux de bord

def grist_api_request(endpoint, method="GET", data=None):
    """Fonction utilitaire pour les requêtes API Grist"""
    # Construction correcte de l'URL
    if "records" in endpoint:
        url = f"https://grist.numerique.gouv.fr/api/docs/{DOC_ID}/tables/{DASHBOARDS_TABLE}/records"
    else:
        url = f"https://grist.numerique.gouv.fr/api/docs/{DOC_ID}/{endpoint}"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Debug
    st.write(f"Requête API vers : {url}")
    st.write(f"Méthode : {method}")
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        # Debug
        st.write(f"Code de statut : {response.status_code}")
        
        if response.status_code == 404:
            st.error(f"URL non trouvée : {url}")
            return None
            
        response.raise_for_status()
        return response.json() if response.content else None
    except Exception as e:
        st.error(f"Erreur API Grist : {str(e)}")
        return None

def get_grist_tables():
    """Récupère la liste des tables disponibles dans Grist."""
    try:
        result = grist_api_request("tables")
        if result and 'tables' in result:
            return [table['id'] for table in result['tables']]
        return []
    except Exception as e:
        st.error(f"Erreur lors de la récupération des tables : {str(e)}")
        return []

def get_grist_data(table_id):
    """Récupère les données d'une table Grist."""
    try:
        result = grist_api_request(f"tables/{table_id}/records")
        if result and 'records' in result and result['records']:
            # Extraction des champs et création du DataFrame
            records = []
            column_order = list(result['records'][0]['fields'].keys())
            for record in result['records']:
                fields = {k.lstrip('$'): v for k, v in record['fields'].items()}
                records.append(fields)

            df = pd.DataFrame(records)
            ordered_columns = [col.lstrip('$') for col in column_order]
            return df[ordered_columns]
        return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {str(e)}")
        return None

def ensure_dashboard_table_exists():
    """Vérifie si la table dashboards existe, la crée si nécessaire."""
    try:
        # Vérifie si la table existe
        tables = get_grist_tables()
        if DASHBOARDS_TABLE not in tables:
            # Création de la table avec les colonnes nécessaires
            data = {
                "tables": [{
                    "tableId": DASHBOARDS_TABLE,
                    "columns": [
                        {"id": "name", "type": "Text"},
                        {"id": "elements", "type": "Text"},
                        {"id": "timestamp", "type": "DateTime"}
                    ]
                }]
            }
            result = grist_api_request("tables", method="POST", data=data)
            if result:
                st.success("Table des tableaux de bord créée avec succès!")
                return True
            else:
                st.error("Erreur lors de la création de la table des tableaux de bord")
                return False
        return True
    except Exception as e:
        st.error(f"Erreur lors de la vérification/création de la table : {str(e)}")
        return False
    
    # Vérifier et créer la table des tableaux de bord si nécessaire
    if not ensure_dashboard_table_exists():
        st.error("Impossible de continuer sans la table des tableaux de bord")
        return
        
def save_dashboard(dashboard_name, elements):
    """Sauvegarde un tableau de bord dans Grist"""
    try:
        data = {
            "records": [{
                "fields": {
                    "name": dashboard_name,
                    "elements": json.dumps(elements),
                    "timestamp": datetime.now().isoformat()
                }
            }]
        }
        result = grist_api_request(f"tables/{DASHBOARDS_TABLE}/records", "POST", data)
        return result is not None
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {str(e)}")
        return False

def load_dashboards():
    """Charge tous les tableaux de bord depuis Grist"""
    try:
        result = grist_api_request(f"tables/{DASHBOARDS_TABLE}/records")
        if result and 'records' in result:
            return [{
                'name': record['fields']['name'],
                'elements': json.loads(record['fields']['elements']),
                'id': record['id']
            } for record in result['records']]
        return []
    except Exception as e:
        st.error(f"Erreur lors du chargement : {str(e)}")
        return []

def get_dashboard_id(dashboard_name):
    """Récupère l'ID d'un tableau de bord par son nom"""
    dashboards = load_dashboards()
    for dashboard in dashboards:
        if dashboard['name'] == dashboard_name:
            return dashboard['id']
    return None

def delete_dashboard(dashboard_name):
    """Supprime un tableau de bord de Grist"""
    try:
        dashboard_id = get_dashboard_id(dashboard_name)
        if dashboard_id:
            result = grist_api_request(f"tables/{DASHBOARDS_TABLE}/records/{dashboard_id}", "DELETE")
            return result is not None
        return False
    except Exception as e:
        st.error(f"Erreur lors de la suppression : {str(e)}")
        return False

def update_dashboard(dashboard_name, new_elements):
    """Met à jour un tableau de bord existant dans Grist"""
    try:
        dashboard_id = get_dashboard_id(dashboard_name)
        if dashboard_id:
            data = {
                "records": [{
                    "id": dashboard_id,
                    "fields": {
                        "elements": json.dumps(new_elements),
                        "timestamp": datetime.now().isoformat()
                    }
                }]
            }
            result = grist_api_request(f"tables/{DASHBOARDS_TABLE}/records", "PATCH", data)
            return result is not None
        return False
    except Exception as e:
        st.error(f"Erreur lors de la mise à jour : {str(e)}")
        return False

def select_or_create_dashboard():
    """Interface pour la sélection/création de tableau de bord"""
    dashboards = load_dashboards()
    dashboard_names = [d["name"] for d in dashboards]
    dashboard_names.append("Créer un nouveau tableau de bord")

    selected_dashboard = st.selectbox("Choisissez un tableau de bord", dashboard_names)
    
    if selected_dashboard == "Créer un nouveau tableau de bord":
        new_dashboard_name = st.text_input("Nom du nouveau tableau de bord")
        if st.button("Créer"):
            if save_dashboard(new_dashboard_name, []):
                st.success(f"Tableau de bord '{new_dashboard_name}' créé!")
                return new_dashboard_name
    elif selected_dashboard in dashboard_names:
        if st.button("Supprimer ce tableau de bord"):
            if delete_dashboard(selected_dashboard):
                st.success(f"Tableau de bord '{selected_dashboard}' supprimé!")
                st.experimental_rerun()
    
    return selected_dashboard

def add_visualization_to_dashboard(dashboard_name, fig, title, var_x=None, var_y=None, graph_type=None, data=None):
    """Ajoute une visualisation au tableau de bord"""
    dashboards = load_dashboards()
    for dashboard in dashboards:
        if dashboard["name"] == dashboard_name:
            elements = dashboard["elements"]
            elements.append({
                "type": "graphique",
                "titre": title,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "fig_dict": fig.to_dict(),
                    "var_x": var_x,
                    "var_y": var_y,
                    "graph_type": graph_type,
                    "data": data.to_dict() if isinstance(data, pd.DataFrame) else None
                }
            })
            if update_dashboard(dashboard_name, elements):
                st.success(f"✅ Visualisation ajoutée au tableau de bord '{dashboard_name}'!")
                return True
    return False

def main():
    st.title("Analyse des données ESR")

    # Initialisation de l'état de session pour les données fusionnées
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None

    # Sélection des tables
    tables = get_grist_tables()
    if not tables:
        st.error("Aucune table disponible.")
        return

    table_selections = st.multiselect(
        "Sélectionnez une ou plusieurs tables à analyser", 
        tables
    )
    
    if not table_selections:
        st.warning("Veuillez sélectionner au moins une table pour l'analyse.")
        return

    # Chargement et fusion des données
    if len(table_selections) == 1:
        df = get_grist_data(table_selections[0])
        if df is not None:
            st.session_state.merged_data = df
        else:
            st.error("Impossible de charger la table sélectionnée.")
            return
    else:
        dataframes = []
        merge_configs = []

        for table_id in table_selections:
            df = get_grist_data(table_id)
            if df is not None:
                dataframes.append(df)

        if len(dataframes) < 2:
            st.warning("Impossible de charger les tables sélectionnées.")
            return

        # Configuration de la fusion
        st.write("### Configuration de la fusion")
        for i in range(len(dataframes) - 1):
            col1, col2 = st.columns(2)
            with col1:
                left_col = st.selectbox(
                    f"Colonne de fusion pour {table_selections[i]}", 
                    dataframes[i].columns.tolist(),
                    key=f"left_{i}"
                )
            with col2:
                right_col = st.selectbox(
                    f"Colonne de fusion pour {table_selections[i + 1]}", 
                    dataframes[i + 1].columns.tolist(),
                    key=f"right_{i}"
                )
            merge_configs.append({"left": left_col, "right": right_col})

        st.session_state.merged_data = merge_multiple_tables(dataframes, merge_configs)

    # Vérification des données fusionnées
    if st.session_state.merged_data is None:
        st.error("Erreur lors du chargement ou de la fusion des tables.")
        return

    # Sélection du type d'analyse
    analysis_type = st.selectbox(
        "Type d'analyse",
        ["Analyse univariée", "Analyse bivariée"],
        key="analysis_type_selector"
    )

    # Analyse univariée
    if analysis_type == "Analyse univariée":
        var = st.selectbox("Sélectionnez la variable:", options=st.session_state.merged_data.columns)
        plot_data = st.session_state.merged_data[var]

        # Vérification du type de la variable et génération de la visualisation appropriée
        fig = None  # Initialisation de la variable fig
        
        # Analyse univariée pour une variable qualitative
        if plot_data.dtype == 'object':
            st.write(f"### Analyse univariée pour {var}")
            freq_table = plot_data.value_counts().reset_index()
            freq_table.columns = ['Valeur', 'Compte']
            freq_table['Pourcentage'] = (freq_table['Compte'] / freq_table['Compte'].sum() * 100).round(2)
            st.dataframe(freq_table)
        else:
            st.write(f"La variable sélectionnée ({var}) n'est pas qualitative.")

        # Configuration de la visualisation
        st.write("### Configuration de la visualisation")
        viz_col1, viz_col2 = st.columns([1, 2])

        is_numeric = pd.api.types.is_numeric_dtype(plot_data)

        with viz_col1:
            if is_numeric:
                graph_type = st.selectbox(
                    "Type de graphique",
                    ["Histogramme", "Boîte à moustaches"],
                    key="univariate_graph",
                    help="Pour les variables numériques, l'histogramme montre la distribution et la boîte à moustaches les statistiques de position"
                )
            else:
                graph_type = st.selectbox(
                    "Type de graphique",
                    ["Barres", "Camembert"],
                    key="univariate_graph",
                    help="Pour les variables catégorielles, les barres montrent les fréquences et le camembert les proportions"
                )
        
        with viz_col2:
            color_scheme = st.selectbox(
                "Palette de couleurs",
                list(COLOR_PALETTES.keys()),
                key="univariate_color"
            )

        # Options avancées
        with st.expander("Options avancées"):
            title = st.text_input(
                "Titre du graphique", 
                f"Distribution de {var}",
                key="title_univariate"
            )
            show_values = st.checkbox("Afficher les valeurs", True, key="show_values_univariate")

        if st.button("Générer la visualisation", key="generate_univariate"):
            try:
                plot_data = st.session_state.merged_data[var].copy()
                
                if graph_type == "Histogramme":
                    if pd.api.types.is_numeric_dtype(plot_data):
                        fig = px.histogram(
                            plot_data,
                            title=title,
                            color_discrete_sequence=COLOR_PALETTES[color_scheme]
                        )
                    else:
                        st.error("L'histogramme n'est disponible que pour les variables numériques.")
                
                elif graph_type == "Boîte à moustaches":
                    if pd.api.types.is_numeric_dtype(plot_data):
                        fig = px.box(
                            plot_data,
                            title=title,
                            color_discrete_sequence=COLOR_PALETTES[color_scheme]
                        )
                    else:
                        st.error("La boîte à moustaches n'est disponible que pour les variables numériques.")
                
                elif graph_type in ["Barres", "Camembert"]:
                    # Préparation des données
                    value_counts = plot_data.value_counts().reset_index()
                    value_counts.columns = ['Valeur', 'Compte']
                    value_counts['Pourcentage'] = (value_counts['Compte'] / value_counts['Compte'].sum() * 100).round(2)
                    
                    if graph_type == "Barres":
                        fig = px.bar(
                            value_counts,
                            x='Valeur',
                            y='Compte',
                            title=title,
                            color_discrete_sequence=COLOR_PALETTES[color_scheme]
                        )
                        if show_values:
                            fig.update_traces(
                                texttemplate='%{y}',
                                textposition='outside'
                            )
                    else:  # Camembert
                        fig = px.pie(
                            value_counts,
                            values='Compte',
                            names='Valeur',
                            title=title,
                            color_discrete_sequence=COLOR_PALETTES[color_scheme]
                        )
                        if show_values:
                            fig.update_traces(
                                textinfo='percent+label'
                            )

                # Mise à jour du layout pour tous les graphiques
                if fig is not None:
                    fig.update_layout(
                        height=600,
                        margin=dict(t=100, b=100),
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                    
                    # Affichage du graphique
                    st.plotly_chart(fig, use_container_width=True)

                    # Sélection ou création de tableau de bord
                    selected_dashboard = select_or_create_dashboard()
                    if selected_dashboard and selected_dashboard != "Créer un nouveau tableau de bord":
                        add_visualization_to_dashboard(
                            dashboard_name=selected_dashboard,
                            fig=fig,
                            title=title,
                            var_x=var,
                            graph_type=graph_type,
                            data=plot_data
                        )

            except Exception as e:
                st.error(f"Erreur lors de la visualisation : {str(e)}")

                # Statistiques descriptives
                st.write("### Statistiques descriptives")
                if pd.api.types.is_numeric_dtype(plot_data):
                    stats = plot_data.describe()
                    stats_df = pd.DataFrame({
                        'Statistique': stats.index,
                        'Valeur': stats.values
                    })
                    st.dataframe(stats_df)
                else:
                    freq_table = value_counts.copy()
                    freq_table['Pourcentage'] = (freq_table['Compte'] / freq_table['Compte'].sum() * 100).round(2)
                    st.dataframe(freq_table)

            except Exception as e:
                st.error(f"Erreur lors de la visualisation : {str(e)}")

    # Analyse bivariée
    elif analysis_type == "Analyse bivariée":
        col1, col2 = st.columns(2)
        
        with col1:
            var_x = st.selectbox(
                "Variable X (axe horizontal)", 
                st.session_state.merged_data.columns,
                key="bivariate_var_x"
            )
        
        with col2:
            var_y = st.selectbox(
                "Variable Y (axe vertical)", 
                [col for col in st.session_state.merged_data.columns if col != var_x],
                key="bivariate_var_y"
            )

        # Configuration de la visualisation
        st.write("### Configuration de la visualisation")
        viz_col1, viz_col2 = st.columns([1, 2])

        plot_data = st.session_state.merged_data[[var_x, var_y]].copy()

        is_x_numeric = pd.api.types.is_numeric_dtype(plot_data[var_x])
        is_y_numeric = pd.api.types.is_numeric_dtype(plot_data[var_y])

        with viz_col1:
            # Déterminer les types de graphiques appropriés
            if is_x_numeric and is_y_numeric:
                graph_type = st.selectbox(
                    "Type de graphique",
                    ["Nuage de points", "Ligne"],
                    key="bivariate_graph",
                    help="Pour deux variables numériques, le nuage de points montre la relation et la ligne montre l'évolution"
                )
            elif is_x_numeric and not is_y_numeric:
                graph_type = st.selectbox(
                    "Type de graphique",
                    ["Boîte à moustaches", "Barres"],
                    key="bivariate_graph",
                    help="Pour une variable numérique et une catégorielle, la boîte à moustaches montre la distribution par catégorie"
                )
            elif not is_x_numeric and is_y_numeric:
                graph_type = st.selectbox(
                    "Type de graphique",
                    ["Barres", "Barres groupées"],
                    key="bivariate_graph",
                    help="Pour une variable catégorielle et une numérique, les barres montrent les moyennes par catégorie"
                )
            else:  # Les deux sont catégorielles
                graph_type = st.selectbox(
                    "Type de graphique",
                    ["Heatmap", "Barres groupées"],
                    key="bivariate_graph",
                    help="Pour deux variables catégorielles, la heatmap montre les fréquences croisées"
                )

        with viz_col2:
            color_scheme = st.selectbox(
                "Palette de couleurs",
                list(COLOR_PALETTES.keys()),
                key="bivariate_color"
            )

        # Options avancées
        with st.expander("Options avancées"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input(
                    "Titre du graphique", 
                    f"Relation entre {var_x} et {var_y}",
                    key="title_bivariate"
                )
                show_values = st.checkbox("Afficher les valeurs", True, key="show_values_bivariate")
            
            with col2:
                sort_order = st.radio(
                    "Tri des données",
                    ["Pas de tri", "Croissant", "Décroissant"],
                    key="sort_order"
                )
                sort_by = st.selectbox(
                    "Trier selon",
                    ["Valeurs", "Fréquences/Moyennes"] if graph_type != "Nuage de points" else ["Valeurs"],
                    key="sort_by"
                )

        if st.button("Générer la visualisation", key="generate_bivariate"):
            try:
                plot_data = st.session_state.merged_data[[var_x, var_y]].copy()

                # Création du graphique selon le type et application du tri
                if sort_order != "Pas de tri":
                    ascending = sort_order == "Croissant"

                if graph_type == "Nuage de points":
                    if sort_order != "Pas de tri":
                        if sort_by == "Valeurs":
                            plot_data = plot_data.sort_values(var_x, ascending=ascending)
                        else:
                            plot_data = plot_data.sort_values(var_y, ascending=ascending)
                    
                    fig = px.scatter(
                        plot_data,
                        x=var_x,
                        y=var_y,
                        title=title,
                        color_discrete_sequence=COLOR_PALETTES[color_scheme]
                    )
                    
                    # Ajout de la ligne de tendance
                    fig.add_traces(
                        px.scatter(
                            plot_data,
                            x=var_x,
                            y=var_y,
                            trendline="ols"
                        ).data
                    )

                elif graph_type == "Ligne":
                    if sort_order != "Pas de tri":
                        if sort_by == "Valeurs":
                            plot_data = plot_data.sort_values(var_x, ascending=ascending)
                        else:
                            plot_data = plot_data.sort_values(var_y, ascending=ascending)
                    
                    fig = px.line(
                        plot_data,
                        x=var_x,
                        y=var_y,
                        title=title,
                        color_discrete_sequence=COLOR_PALETTES[color_scheme]
                    )

                elif graph_type in ["Barres", "Barres groupées"]:
                    if graph_type == "Barres":
                        agg_data = plot_data.groupby(var_x)[var_y].mean().reset_index()
                        if sort_order != "Pas de tri":
                            if sort_by == "Valeurs":
                                agg_data = agg_data.sort_values(var_x, ascending=ascending)
                            else:
                                agg_data = agg_data.sort_values(var_y, ascending=ascending)
                        
                        fig = px.bar(
                            agg_data,
                            x=var_x,
                            y=var_y,
                            title=title,
                            color_discrete_sequence=COLOR_PALETTES[color_scheme]
                        )
                    else:  # Barres groupées
                        if sort_order != "Pas de tri":
                            if sort_by == "Valeurs":
                                plot_data = plot_data.sort_values(var_x, ascending=ascending)
                            else:
                                plot_data = plot_data.sort_values(var_y, ascending=ascending)
                        
                        fig = px.bar(
                            plot_data,
                            x=var_x,
                            y=var_y,
                            title=title,
                            color=var_y,
                            barmode='group',
                            color_discrete_sequence=COLOR_PALETTES[color_scheme]
                        )

                elif graph_type == "Heatmap":
                    pivot_table = pd.pivot_table(
                        plot_data,
                        values=var_y,
                        index=var_x,
                        columns=var_y,
                        aggfunc='count'
                    ).fillna(0)
                    
                    if sort_order != "Pas de tri":
                        if sort_by == "Valeurs":
                            pivot_table = pivot_table.sort_index(ascending=ascending)
                        else:
                            pivot_table = pivot_table.sort_values(var_y, axis=1, ascending=ascending)
                    
                    fig = px.imshow(
                        pivot_table,
                        title=title,
                        color_continuous_scale=COLOR_PALETTES[color_scheme]
                    )

                # Mise à jour du layout pour tous les graphiques
                fig.update_layout(
                    height=600,
                    margin=dict(t=100, b=100),
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )

                # Affichage des valeurs si demandé
                if show_values and graph_type in ["Barres", "Barres groupées"]:
                    fig.update_traces(textposition='outside', texttemplate='%{y:.2f}')

                # Affichage du graphique
                st.plotly_chart(fig, use_container_width=True)

                # Sélection ou création de tableau de bord
                selected_dashboard = select_or_create_dashboard()
                if selected_dashboard and selected_dashboard != "Créer un nouveau tableau de bord":
                    add_visualization_to_dashboard(
                        dashboard_name=selected_dashboard,
                        fig=fig,
                        title=title,
                        var_x=var_x,
                        var_y=var_y,
                        graph_type=graph_type,
                        data=plot_data
                    )

                # Statistiques descriptives
                st.write("### Statistiques descriptives")

                # Vérifier les types de variables
                is_var_x_numeric = pd.api.types.is_numeric_dtype(plot_data[var_x])
                is_var_y_numeric = pd.api.types.is_numeric_dtype(plot_data[var_y])
                is_var_x_categorical = isinstance(plot_data[var_x].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(plot_data[var_x])
                is_var_y_categorical = isinstance(plot_data[var_y].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(plot_data[var_y])

                if is_var_x_numeric and is_var_y_numeric:
                    correlation = plot_data[var_x].corr(plot_data[var_y])
                    st.write(f"Coefficient de corrélation : {correlation:.4f}")
                elif is_var_x_categorical and is_var_y_numeric:
                    grouped_stats = plot_data.groupby(var_x)[var_y].describe().reset_index()
                    st.dataframe(grouped_stats)
                elif is_var_x_numeric and is_var_y_categorical:
                    grouped_stats = plot_data.groupby(var_y)[var_x].describe().reset_index()
                    st.dataframe(grouped_stats)
                else:
                    st.write("Les statistiques descriptives ne sont pas disponibles pour cette combinaison de variables.")

            except Exception as e:
                st.error(f"Erreur lors de la visualisation : {str(e)}")

# Exécution de l'application
if __name__ == "__main__":
    main()
