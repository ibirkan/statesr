import streamlit as st 
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Inclus dans plotly
import plotly.figure_factory as ff  # Inclus dans plotly
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

# Configuration de la page
st.set_page_config(
    page_title="Indicateurs ESR",
    page_icon="📊",
    layout="wide"
)

# Palettes de couleurs prédéfinies
COLOR_PALETTES = {
    "Bleu": sns.color_palette("Blues", 6).as_hex(),
    "Vert": sns.color_palette("Greens", 6).as_hex(),
    "Rouge": sns.color_palette("Reds", 6).as_hex(),
    "Orange": sns.color_palette("Oranges", 6).as_hex(),
    "Violet": sns.color_palette("Purples", 6).as_hex(),
    "Gris": sns.color_palette("Greys", 6).as_hex()
}

# Configuration Grist
API_KEY = st.secrets["grist_key"]
DOC_ID = st.secrets["grist_doc_id"]
BASE_URL = "https://grist.numerique.gouv.fr/api/docs"

# fonctions api de base
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

# fonctions de gestion des tables
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
        result = grist_api_request(table_id)
        if result and 'records' in result and result['records']:
            records = []
            for record in result['records']:
                if 'fields' in record:
                    fields = {k.lstrip('$'): v for k, v in record['fields'].items()}
                    records.append(fields)
            
            if records:
                return pd.DataFrame(records)
        return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {str(e)}")
        return None

# Fonctions pour les différentes pages
def page_analyse():
    st.title("Analyse des données ESR")

def main():
    # App layout and navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à", ["Analyse des données ESR", "Page 1", "Page 2"])

    if page == "Analyse des données ESR":
        page_analyse()
    elif page == "Page 1":
        page_1()
    elif page == "Page 2":
        page_2()

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
        # Sélection de la variable
        var = st.selectbox("Sélectionnez la variable:", options=st.session_state.merged_data.columns)
        plot_data = st.session_state.merged_data[var]
        
        # Vérification des données non nulles
        if plot_data is not None and not plot_data.empty:
            # Détecter le type de variable
            is_numeric = pd.api.types.is_numeric_dtype(plot_data)
        
            # Affichage des statistiques de base
            st.write(f"### Statistiques principales de la variable {var}")
        
            if is_numeric:
                # Statistiques pour variable quantitative
                stats_df = pd.DataFrame({
                    'Statistique': ['Effectif total', 'Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum'],
                    'Valeur': [
                        len(plot_data),
                        plot_data.mean().round(2),
                        plot_data.median().round(2),
                        plot_data.std().round(2),
                        plot_data.min(),
                        plot_data.max()
                    ]
                })
                st.dataframe(stats_df)
                
                # Options de regroupement
                st.write("### Options de regroupement")
                grouping_method = st.selectbox(
                    "Méthode de regroupement",
                    ["Aucune", "Quantile", "Manuelle"],
                    key="grouping_method"
                )
                
                if grouping_method == "Quantile":
                    quantile_type = st.selectbox(
                        "Type de regroupement",
                        ["Quartile (4 groupes)", "Quintile (5 groupes)", "Décile (10 groupes)"],
                        key="quantile_type"
                    )
                    
                    n_groups = {"Quartile (4 groupes)": 4, 
                               "Quintile (5 groupes)": 5, 
                               "Décile (10 groupes)": 10}[quantile_type]
                    
                    grouped_data = pd.qcut(plot_data, q=n_groups)
                    value_counts = grouped_data.value_counts().reset_index()
                    value_counts.columns = ['Groupe', 'Effectif']
                    value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100).round(2)
                    
                    # Statistiques par groupe
                    group_stats = plot_data.groupby(grouped_data).agg(['mean', 'max']).round(2)
                    group_stats.columns = ['Moyenne', 'Maximum']
                    
                    st.write("### Statistiques par groupe")
                    st.dataframe(pd.concat([value_counts.set_index('Groupe'), 
                                          group_stats], axis=1))
                    
                elif grouping_method == "Manuelle":
                    n_groups = st.number_input("Nombre de groupes", min_value=2, value=3)
                    breaks = []
                    for i in range(n_groups + 1):
                        if i == 0:
                            val = plot_data.min()
                        elif i == n_groups:
                            val = plot_data.max()
                        else:
                            val = st.number_input(f"Seuil {i}", 
                                                value=float(plot_data.min() + (i/n_groups)*(plot_data.max()-plot_data.min())))
                        breaks.append(val)
                    
                    grouped_data = pd.cut(plot_data, bins=breaks)
                    value_counts = grouped_data.value_counts().reset_index()
                    value_counts.columns = ['Groupe', 'Effectif']
                    value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100).round(2)
                    
                    st.write("### Répartition des groupes")
                    st.dataframe(value_counts)
                    
            else:
                # Statistiques pour variable qualitative
                value_counts = plot_data.value_counts().reset_index()
                value_counts.columns = ['Modalité', 'Effectif']
                value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100).round(2)
                st.dataframe(value_counts)
        
            # Visualisation
            st.write("### Configuration de la visualisation")
            
            # Option d'évolution temporelle
            show_evolution = st.checkbox("Afficher l'évolution temporelle", key="show_evolution")
            
            if show_evolution:
                # Détection des colonnes de type date
                date_columns = st.session_state.merged_data.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns
                
                if len(date_columns) > 0:
                    time_var = st.selectbox(
                        "Sélectionner la variable temporelle",
                        date_columns,
                        key="time_variable"
                    )
                    
                    if pd.api.types.is_datetime64_any_dtype(st.session_state.merged_data[time_var]):
                        # Configuration du graphique d'évolution
                        viz_col1, viz_col2 = st.columns([1, 2])
                        with viz_col1:
                            if is_numeric:
                                evolution_type = st.selectbox(
                                    "Valeur à afficher",
                                    ["Somme globale", "Moyenne", "Maximum"],
                                    key="evolution_value_type"
                                )
                            else:
                                evolution_type = st.selectbox(
                                    "Valeur à afficher",
                                    ["Effectifs", "Taux (%)"],
                                    key="evolution_value_type"
                                )
                            
                            graph_type = st.selectbox(
                                "Type de graphique",
                                ["Line plot", "Bar plot", "Lollipop plot"],
                                key="evolution_graph_type"
                            )
                        
                        with viz_col2:
                            color_scheme = st.selectbox(
                                "Palette de couleurs",
                                list(COLOR_PALETTES.keys()),
                                key="evolution_color"
                            )
                    else:
                        st.warning("La colonne sélectionnée n'est pas de type datetime.")
                        show_evolution = False
                else:
                    st.warning("Aucune colonne de type date n'a été détectée dans les données.")
                    show_evolution = False
            
            else:  # Visualisation standard (non temporelle)
                viz_col1, viz_col2 = st.columns([1, 2])
                with viz_col1:
                    if is_numeric:
                        if grouping_method == "Aucune":
                            graph_type = st.selectbox(
                                "Type de graphique",
                                ["Histogramme", "Density plot"],
                                key="static_graph_type"
                            )
                        else:
                            graph_type = st.selectbox(
                                "Type de graphique",
                                ["Bar plot", "Lollipop plot", "Treemap"],
                                key="static_graph_type"
                            )
                            value_type = "Effectifs" if grouping_method == "Manuelle" else st.selectbox(
                                "Valeur à afficher",
                                ["Effectifs", "Moyenne", "Maximum"],
                                key="static_value_type"
                            )
                    else:
                        graph_type = st.selectbox(
                            "Type de graphique",
                            ["Bar plot", "Lollipop plot", "Treemap"],
                            key="static_graph_type"
                        )
                
                with viz_col2:
                    color_scheme = st.selectbox(
                        "Palette de couleurs",
                        list(COLOR_PALETTES.keys()),
                        key="static_color"
                    )
        
            # Options avancées
            with st.expander("Options avancées"):
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    title = st.text_input("Titre du graphique", f"Distribution de {var}")
                    x_axis = st.text_input("Titre de l'axe X", var)
                    y_axis = st.text_input("Titre de l'axe Y", "Valeur")
                with adv_col2:
                    source = st.text_input("Source des données", "")
                    show_values = st.checkbox("Afficher les valeurs", True)
                                
            # Génération du graphique
            if st.button("Générer la visualisation"):
                try:
                    if show_evolution:
                        # Préparation des données temporelles
                        evolution_data = plot_data.groupby(time_var)
                        
                        if is_numeric:
                            if evolution_type == "Somme globale":
                                y_values = evolution_data.sum()
                            elif evolution_type == "Moyenne":
                                y_values = evolution_data.mean()
                            else:  # Maximum
                                y_values = evolution_data.max()
                        else:
                            if evolution_type == "Effectifs":
                                y_values = evolution_data.count()
                            else:  # Taux
                                y_values = (evolution_data.count() / len(plot_data)) * 100
                        
                        evolution_data = pd.DataFrame({'date': y_values.index, 'value': y_values.values})
                        
                        # Création du graphique d'évolution
                        if graph_type == "Line plot":
                            fig = px.line(evolution_data, x='date', y='value',
                                        title=title,
                                        color_discrete_sequence=COLOR_PALETTES[color_scheme])
                        elif graph_type == "Bar plot":
                            fig = px.bar(evolution_data, x='date', y='value',
                                       title=title,
                                       color_discrete_sequence=COLOR_PALETTES[color_scheme])
                        else:  # Lollipop plot avec matplotlib
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            markerline, stemlines, baseline = ax.stem(
                                evolution_data['date'],
                                evolution_data['value'],
                                linefmt=COLOR_PALETTES[color_scheme][0],
                                markerfmt=f'o{COLOR_PALETTES[color_scheme][0]}',
                                basefmt=' '
                            )
                            
                            plt.setp(markerline, markersize=10)
                            plt.setp(stemlines, linewidth=2)
                            
                            if show_values:
                                for x, y in zip(evolution_data['date'], evolution_data['value']):
                                    ax.text(x, y, f'{y:.0f}', ha='center', va='bottom')
                            
                            ax.set_title(title, pad=20)
                            ax.set_xlabel(x_axis)
                            ax.set_ylabel(y_axis)
                            ax.grid(True, linestyle='--', alpha=0.3)
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                            plt.close()
                            return
            
                    else:  # Visualisation standard
                        if is_numeric:
                            if grouping_method == "Aucune":
                                if graph_type == "Histogramme":
                                    fig = px.histogram(plot_data, title=title,
                                                       color_discrete_sequence=COLOR_PALETTES[color_scheme])
                                    if show_values:
                                        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                                else:  # Density plot with Seaborn
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    sns.kdeplot(plot_data.dropna(), ax=ax, fill=True, color=COLOR_PALETTES[color_scheme][0])
                                    ax.set_title(title)
                                    ax.set_xlabel(x_axis)
                                    ax.set_ylabel(y_axis)
                                    st.pyplot(fig)
                                    plt.close()
                          
                            else:
                                data = value_counts
                                if value_type != "Effectifs":
                                    y_col = 'Moyenne' if value_type == "Moyenne" else 'Maximum'
                                    data = pd.concat([value_counts, group_stats[y_col]], axis=1)
                                
                            if graph_type == "Bar plot":
                                fig = px.bar(value_counts, x='Modalité', y='Effectif',
                                           title=title,
                                           color_discrete_sequence=COLOR_PALETTES[color_scheme])

                            elif graph_type == "Lollipop plot":
                                fig, ax = plt.subplots(figsize=(12, 6))
                                color = COLOR_PALETTES[color_scheme][0]
                                x = value_counts['Modalité'].tolist()
                                y = value_counts['Effectif'].tolist()
                            
                                markerline, stemlines, baseline = ax.stem(
                                    x, y,
                                    linefmt='-',
                                    markerfmt='o',
                                    basefmt=' '
                                )
                            
                                markerline.set_color(color)
                                markerline.set_markersize(10)
                                plt.setp(stemlines, color=color, linewidth=2)
                            
                                if show_values:
                                    for i, v in enumerate(y):
                                        ax.text(x[i], v, f'{v:.0f}', ha='center', va='bottom')
                            
                                ax.set_title(title)
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel(y_axis)
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                            
                                st.pyplot(fig)
                                plt.close()
                                return
                            
                            elif graph_type == "Treemap":
                               fig, ax = plt.subplots(figsize=(12, 8))
                               values = value_counts['Effectif'].values
                               norm_values = (values - values.min()) / (values.max() - values.min())
                               colors = [COLOR_PALETTES[color_scheme][int(v * (len(COLOR_PALETTES[color_scheme])-1))] 
                                         for v in norm_values]
                               
                               squarify.plot(
                                   sizes=values,
                                   label=value_counts['Modalité'],
                                   color=colors,
                                   alpha=0.8,
                                   text_kwargs={'fontsize':10},
                                   pad=True,
                                   value=values if show_values else None
                               )
                               
                               plt.title(title, pad=20)
                               plt.axis('off')
                               plt.tight_layout()
                               
                               st.pyplot(fig)
                               plt.close()
                               return

                        
                        else:  # Pour les variables qualitatives
                            if graph_type == "Bar plot":
                                fig = px.bar(value_counts, x='Modalité', y='Effectif',
                                           title=title,
                                           color_discrete_sequence=COLOR_PALETTES[color_scheme])

                            elif graph_type == "Lollipop plot":
                                fig, ax = plt.subplots(figsize=(12, 6))
                                color = COLOR_PALETTES[color_scheme][0]
                                x = value_counts['Modalité'].tolist()
                                y = value_counts['Effectif'].tolist()
                            
                                markerline, stemlines, baseline = ax.stem(
                                    x, y,
                                    linefmt='-',
                                    markerfmt='o',
                                    basefmt=' '
                                )
                            
                                markerline.set_color(color)
                                markerline.set_markersize(10)
                                plt.setp(stemlines, color=color, linewidth=2)
                            
                                if show_values:
                                    for i, v in enumerate(y):
                                        ax.text(x[i], v, f'{v:.0f}', ha='center', va='bottom')
                            
                                ax.set_title(title)
                                ax.set_xlabel(x_axis)
                                ax.set_ylabel(y_axis)
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                            
                                st.pyplot(fig)
                                plt.close()
                                return
                            
                            elif graph_type == "Treemap":
                               fig, ax = plt.subplots(figsize=(12, 8))
                               values = value_counts['Effectif'].values
                               norm_values = (values - values.min()) / (values.max() - values.min())
                               colors = [COLOR_PALETTES[color_scheme][int(v * (len(COLOR_PALETTES[color_scheme])-1))] 
                                         for v in norm_values]
                               
                               squarify.plot(
                                   sizes=values,
                                   label=value_counts['Modalité'],
                                   color=colors,
                                   alpha=0.8,
                                   text_kwargs={'fontsize':10},
                                   pad=True,
                                   value=values if show_values else None
                               )
                               
                               plt.title(title, pad=20)
                               plt.axis('off')
                               plt.tight_layout()
                               
                               st.pyplot(fig)
                               plt.close()
                               return
                        
                        # Mise à jour du layout pour les graphiques Plotly
                        if fig is not None and isinstance(fig, go.Figure):
                            fig.update_layout(
                                height=600,
                                margin=dict(t=100, b=100),
                                showlegend=True,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                xaxis_title=x_axis,
                                yaxis_title=y_axis
                            )
                            
                            if source:
                                fig.add_annotation(
                                    text=f"Source: {source}",
                                    xref="paper",
                                    yref="paper",
                                    x=0,
                                    y=-0.15,
                                    showarrow=False,
                                    font=dict(size=10),
                                    align="left"
                                )
                        
                            if show_values and hasattr(fig.data[0], "text"):
                                if isinstance(fig.data[0], go.Bar):
                                    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                                else:
                                    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                            
                            st.plotly_chart(fig, use_container_width=True)
            
                except Exception as e:
                    st.error(f"Erreur lors de la génération du graphique : {str(e)}")

    # Analyse bivariée
    elif analysis_type == "Analyse bivariée":
        try:
            # Sélection des variables
            var_x = st.selectbox("Variable X (axe horizontal)", st.session_state.merged_data.columns)
            var_y = st.selectbox("Variable Y (axe vertical)", 
                                 [col for col in st.session_state.merged_data.columns if col != var_x])
    
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
                    if fig is not None:
                        fig.update_layout(
                            height=600,
                            margin=dict(t=100, b=100),
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis_title=var_x,
                            yaxis_title=var_y
                        )
    
                        # Ajout de la source si spécifiée
                        if source:
                            fig.add_annotation(
                                text=f"Source: {source}",
                                xref="paper",
                                yref="paper",
                                x=0,
                                y=-0.15,
                                showarrow=False,
                                font=dict(size=10),
                                align="left"
                            )
    
                        # Affichage des valeurs si demandé
                        if show_values and hasattr(fig.data[0], "text"):
                            fig.update_traces(texttemplate='%{y:.2f}', textposition='top center')
    
                    # Création d'une clé unique pour le graphique
                    unique_key = f"plot_bi_{var_x}_{var_y}_{graph_type}"
    
                    # Affichage du graphique avec clé unique
                    st.plotly_chart(fig, use_container_width=True, key=unique_key)
                    st.write("### Statistiques détaillées")
                    if is_x_numeric and is_y_numeric:
                        correlation = plot_data[var_x].corr(plot_data[var_y])
                        st.write(f"Coefficient de corrélation : {correlation:.4f}")
    
                        stats = pd.DataFrame({
                            'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max'],
                            f'{var_x}': [plot_data[var_x].mean(), plot_data[var_x].median(), 
                                         plot_data[var_x].std(), plot_data[var_x].min(), 
                                         plot_data[var_x].max()],
                            f'{var_y}': [plot_data[var_y].mean(), plot_data[var_y].median(), 
                                         plot_data[var_y].std(), plot_data[var_y].min(), 
                                         plot_data[var_y].max()]
                        }).round(2)
                        st.dataframe(stats)
                    else:
                        cross_tab = pd.crosstab(plot_data[var_x], plot_data[var_y], normalize='index') * 100
                        st.write("Distribution croisée (%):")
                        st.dataframe(cross_tab.round(2))
    
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
        except Exception as e:
            st.error(f"Erreur lors de la configuration de l'analyse bivariée : {str(e)}")

# Exécution de l'application
if __name__ == "__main__":
    main()

