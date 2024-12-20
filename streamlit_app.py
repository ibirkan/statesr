# Importations
import streamlit as st 
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
from io import BytesIO
import squarify
import seaborn as sns
from scipy import stats

# Configuration de la page Streamlit - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Indicateurs de l'ESR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration de la police Marianne et des styles
st.markdown("""
    <style>
        @import url('https://cdn.jsdelivr.net/npm/@gouvfr/dsfr@1.7.2/dist/fonts/Marianne-Light.woff2');
        @import url('https://cdn.jsdelivr.net/npm/@gouvfr/dsfr@1.7.2/dist/fonts/Marianne-Regular.woff2');
        @import url('https://cdn.jsdelivr.net/npm/@gouvfr/dsfr@1.7.2/dist/fonts/Marianne-Medium.woff2');
        @import url('https://cdn.jsdelivr.net/npm/@gouvfr/dsfr@1.7.2/dist/fonts/Marianne-Bold.woff2');

        /* Appliquer Marianne à tous les éléments */
        * {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Styles spécifiques pour les différents éléments Streamlit */
        .stMarkdown, .stText, .stCode, .stTextInput, .stNumberInput, 
        .stSelectbox, .stMultiselect, .stTextArea, .stButton, 
        .stDataFrame, .stTable, .stHeader, .stSubheader, .stCaption {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }

        /* Styles pour les titres */
        h1, h2, h3, h4, h5, h6 {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-weight: 600 !important;
        }

        /* Style pour les widgets */
        .stSelectbox > div > div > div {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }

        /* Style pour les boutons */
        .stButton > button {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-weight: 500 !important;
        }

        /* Style pour les tableaux de données */
        .dataframe {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        .dataframe th {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-weight: 600 !important;
        }

        /* Style pour les expanders */
        .streamlit-expanderHeader {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-weight: 500 !important;
        }

        /* Style pour les légendes */
        .stCaption {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# Configuration de base
sns.set_theme()
sns.set_style("whitegrid")

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

# Fonctions API Grist
def grist_api_request(endpoint, method="GET", data=None):
    """Fonction utilitaire pour les requêtes API Grist"""
    # L'endpoint contient déjà le chemin complet, pas besoin d'ajouter "tables"
    url = f"{BASE_URL}/{DOC_ID}/{endpoint}"
    
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
        elif method == "PUT":  # Ajout de la méthode PUT
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
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
            tables_dict = {}
            for table in result['tables']:
                # Convertir l'ID en nom plus lisible
                display_name = table['id'].replace('_', ' ')  # Remplacer les underscores par des espaces
                tables_dict[display_name] = table['id']
            
            return tables_dict
        return {}
    except Exception as e:
        st.error(f"Erreur lors de la récupération des tables : {str(e)}")
        return {}

def get_grist_data(table_id):
    """Récupère les données d'une table Grist avec les noms lisibles des colonnes."""
    try:
        # Récupérer les données
        result = grist_api_request(f"tables/{table_id}/records")
        # Récupérer les métadonnées des colonnes pour avoir les noms lisibles
        columns_metadata = grist_api_request(f"tables/{table_id}/columns")
        
        if result and 'records' in result and columns_metadata and 'columns' in columns_metadata:
            # Créer un dictionnaire de mapping id -> label pour les noms lisibles
            column_mapping = {
                col['id']: col['fields']['label'] 
                for col in columns_metadata['columns']
                if 'fields' in col and 'label' in col['fields']
            }
            
            records = []
            for record in result['records']:
                if 'fields' in record:
                    fields = record['fields']
                    records.append(fields)
            
            if records:
                df = pd.DataFrame(records)
                # Renommer les colonnes avec leurs labels lisibles (avec espaces et accents)
                df = df.rename(columns=column_mapping)
                return df
            
        return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {str(e)}")
        return None
        
# Fonctions de gestion des données
def merge_multiple_tables(dataframes, merge_configs):
    """Fusionne plusieurs DataFrames selon les configurations spécifiées."""
    merged_df = dataframes[0]
    for i in range(1, len(dataframes)):
        merge_config = merge_configs[i - 1]
        if merged_df[merge_config['left']].dtype != dataframes[i][merge_config['right']].dtype:
            if pd.api.types.is_numeric_dtype(merged_df[merge_config['left']]) and pd.api.types.is_numeric_dtype(dataframes[i][merge_config['right']]):
                merged_df[merge_config['left']] = merged_df[merge_config['left']].astype(float)
                dataframes[i][merge_config['right']] = dataframes[i][merge_config['right']].astype(float)
            else:
                merged_df[merge_config['left']] = merged_df[merge_config['left']].astype(str)
                dataframes[i][merge_config['right']] = dataframes[i][merge_config['right']].astype(str)
        merged_df = merged_df.merge(dataframes[i], left_on=merge_config['left'], right_on=merge_config['right'], how='outer')
    return merged_df

def is_numeric_column(df, column):
    """Vérifie si une colonne est numérique de manière sûre."""
    try:
        return pd.api.types.is_numeric_dtype(df[column])
    except Exception as e:
        st.error(f"Erreur lors de la vérification du type de la colonne {column}: {str(e)}")
        return False

def check_normality(data, var):
    """Vérifie la normalité d'une variable avec adaptation pour les grands échantillons."""
    n = len(data)
    if n > 5000:
        _, p_value = stats.normaltest(data[var])
    else:
        _, p_value = stats.shapiro(data[var])
    return p_value > 0.05

def check_duplicates(df, var_x, var_y):
    """Vérifie la présence de doublons dans les variables."""
    duplicates_x = df[var_x].duplicated().any()
    duplicates_y = df[var_y].duplicated().any()
    return duplicates_x or duplicates_y

def calculate_grouped_stats(data, var, groupby_col, agg_method='mean'):
    """Calcule les statistiques avec agrégation."""
    clean_data = data.dropna(subset=[var, groupby_col])
    
    detailed_stats = {
        'sum': clean_data[var].sum(),
        'mean': clean_data[var].mean(),
        'median': clean_data[var].median(),
        'std': clean_data[var].std(),
        'count': len(clean_data)
    }
    
    agg_data = clean_data.groupby(groupby_col).agg({var: agg_method}).reset_index()
    
    agg_stats = {
        'sum': agg_data[var].sum(),
        'mean': agg_data[var].mean(),
        'median': agg_data[var].median(),
        'std': agg_data[var].std(),
        'count': len(agg_data)
    }
    
    return detailed_stats, agg_stats, agg_data

def create_interactive_stats_table(stats_df):
    """Crée un tableau de statistiques interactif."""
    # Style personnalisé pour le tableau
    styled_df = stats_df.style\
        .set_properties(**{
            'font-size': '14px',
            'padding': '10px',
            'border': '1px solid #e6e6e6',
            'font-family': 'Marianne, sans-serif',
            'text-align': 'left'
        })\
        .set_table_styles([
            {'selector': '',
             'props': [('margin', '0 auto')]},
            {'selector': 'th',
             'props': [
                 ('background-color', '#f0f2f6'),
                 ('color', '#262730'),
                 ('font-weight', 'bold'),
                 ('padding', '10px'),
                 ('font-size', '14px'),
                 ('border', '1px solid #e6e6e6'),
                 ('font-family', 'Marianne, sans-serif'),
                 ('text-align', 'left')
             ]},
            {'selector': 'tr:nth-child(even)',
             'props': [('background-color', '#f9f9f9')]},
            {'selector': 'tr:nth-child(odd)',
             'props': [('background-color', 'white')]}
        ])
    
    return st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

def calculate_regression(x, y):
    """Calcule la régression linéaire de manière robuste."""
    try:
        z = np.polyfit(x, y, 1)
        return z, True
    except np.linalg.LinAlgError:
        try:
            import statsmodels.api as sm
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()
            return [results.params[1], results.params[0]], True
        except:
            return None, False

# Fonctions de visualisation univariée
def plot_qualitative_bar(data, title, x_label, y_label, color_palette, show_values=True):
    """Crée un graphique en barres pour une variable qualitative avec une mise en page adaptative."""
    # Initialisation de la liste des annotations
    annotations = []
    if not isinstance(data, pd.DataFrame):
        st.error("Les données ne sont pas dans le format attendu")
        return None
    
    columns = data.columns.tolist()
    category_col = columns[0]
    value_col = 'Effectif'
    
    n_categories = len(data[category_col])
    bar_width = min(0.5, 1.0 / (n_categories + 1))
    
    max_value = data[value_col].max()
    y_max = max_value * 1.2
    
    fig = go.Figure(data=[
        go.Bar(
            x=data[category_col],
            y=data[value_col],
            marker_color=color_palette[0],
            width=[bar_width] * n_categories
        )
    ])
    
    width = max(400, min(800, 200 * n_categories))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=500,
        margin=dict(t=100, b=100, l=50, r=50),
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            tickangle=45 if n_categories > 2 else 0
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray',
            range=[0, y_max]
        )
    )

    if show_values:
        text_positions = ['outside' if val / max_value > 0.15 else 'auto' for val in data[value_col]]
        fig.update_traces(
            text=data[value_col].round(1),
            textposition=text_positions,
            texttemplate='%{text:.1f}',
            textfont=dict(size=12)
        )

    return fig

def plot_qualitative_lollipop(data, title, x_label, y_label, color_palette, show_values=True):
    """Crée un graphique en lollipop pour une variable qualitative."""
    if not isinstance(data, pd.DataFrame):
        st.error("Les données ne sont pas dans le format attendu")
        return None
    
    columns = data.columns.tolist()
    category_col = columns[0]
    value_col = 'Effectif'
    
    n_categories = len(data[category_col])
    max_value = data[value_col].max()
    y_max = max_value * 1.2
    
    fig = go.Figure()
    
    # Ajouter les lignes
    fig.add_trace(go.Scatter(
        x=data[category_col],
        y=data[value_col],
        mode='lines',
        line=dict(color=color_palette[0], width=2),
        showlegend=False
    ))
    
    # Ajouter les points
    fig.add_trace(go.Scatter(
        x=data[category_col],
        y=data[value_col],
        mode='markers',
        marker=dict(
            color=color_palette[0],
            size=12,
            line=dict(color='white', width=1)
        ),
        showlegend=False
    ))
    
    width = max(400, min(800, 200 * n_categories))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=500,
        margin=dict(t=100, b=100, l=50, r=50),
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            tickangle=45 if n_categories > 2 else 0
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray',
            range=[0, y_max]
        )
    )

    if show_values:
        text_positions = ['top center' if val / max_value > 0.15 else 'middle center' for val in data[value_col]]
        fig.add_trace(go.Scatter(
            x=data[category_col],
            y=data[value_col],
            mode='text',
            text=data[value_col].round(1),
            textposition=text_positions,
            texttemplate='%{text:.1f}',
            textfont=dict(size=12),
            showlegend=False
        ))

    return fig

def plot_qualitative_treemap(data, title, color_palette):
    """Crée un treemap pour une variable qualitative."""
    if not isinstance(data, pd.DataFrame):
        st.error("Les données ne sont pas dans le format attendu")
        return None
    
    columns = data.columns.tolist()
    category_col = columns[0]
    value_col = 'Effectif'
    
    fig = go.Figure(go.Treemap(
        labels=data[category_col],
        parents=[''] * len(data),
        values=data[value_col],
        textinfo='label+value',
        marker=dict(colors=color_palette),
        texttemplate='%{label}<br>%{value:.1f}',
        hovertemplate='%{label}<br>Valeur: %{value:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        width=800,
        height=500,
        margin=dict(t=100, b=100, l=20, r=20)
    )
    
    return fig
    
def plot_density(plot_data, var, title, x_axis, y_axis):
    """Crée un graphique de densité."""
    fig = ff.create_distplot(
        [plot_data],
        [var],
        show_hist=False,
        show_rug=False,
        colors=[COLOR_PALETTES['Bleu'][0]]
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        height=600,
        margin=dict(t=100, b=100),
    )
    
    return fig

def plot_quantile_distribution(data, title, y_label, color_palette, plot_type, is_integer_variable):
    """
    Crée une visualisation améliorée de distribution pour les données quantitatives groupées par quantiles.
    """
    fig = go.Figure()
    
    if plot_type == "Boîte à moustaches":
        # Créer une boîte à moustaches pour chaque quantile
        quantiles = pd.qcut(data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        for i, quantile in enumerate(sorted(quantiles.unique())):
            subset = data[quantiles == quantile]
            fig.add_trace(go.Box(
                y=subset,
                name=quantile,
                marker_color=color_palette[i % len(color_palette)],
                boxpoints='outliers',  # Montrer uniquement les points aberrants
                boxmean=True  # Montrer la moyenne
            ))
    
    elif plot_type == "Violin plot":
        # Créer un violin plot pour chaque quantile
        quantiles = pd.qcut(data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        for i, quantile in enumerate(sorted(quantiles.unique())):
            subset = data[quantiles == quantile]
            fig.add_trace(go.Violin(
                y=subset,
                name=quantile,
                box_visible=True,
                meanline_visible=True,
                marker_color=color_palette[i % len(color_palette)]
            ))
    
    elif plot_type == "Box plot avec points":
        # Créer un box plot avec points pour chaque quantile
        quantiles = pd.qcut(data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        for i, quantile in enumerate(sorted(quantiles.unique())):
            subset = data[quantiles == quantile]
            fig.add_trace(go.Box(
                y=subset,
                name=quantile,
                marker_color=color_palette[i % len(color_palette)],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
    
    # Calcul des statistiques pour les annotations
    stats = pd.DataFrame({
        'Quantile': pd.qcut(data, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']),
        'Valeur': data
    }).groupby('Quantile').agg(['min', 'max', 'mean', 'median'])
    
    # Annotations améliorées
    annotations = []
    for i, quantile in enumerate(stats.index):
        stats_text = (
            f"{quantile}<br>"
            f"Min: {int(stats.loc[quantile, ('Valeur', 'min')]) if is_integer_variable else round(stats.loc[quantile, ('Valeur', 'min')], 2)}<br>"
            f"Max: {int(stats.loc[quantile, ('Valeur', 'max')]) if is_integer_variable else round(stats.loc[quantile, ('Valeur', 'max')], 2)}<br>"
            f"Moyenne: {int(stats.loc[quantile, ('Valeur', 'mean')]) if is_integer_variable else round(stats.loc[quantile, ('Valeur', 'mean')], 2)}"
        )
        
        annotations.append(dict(
            x=i,
            y=stats.loc[quantile, ('Valeur', 'max')],
            text=stats_text,
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        ))
    
    fig.update_layout(
        title=title,
        yaxis_title=y_label,
        height=600,
        showlegend=True,
        legend_title="Quantiles",
        annotations=annotations,
        plot_bgcolor='white',
        yaxis=dict(
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray'
        ),
        xaxis_title="Quantiles"
    )
    
    return fig

# Fonctions de visualisation bivariée
def plot_mixed_bivariate(df, qual_var, quant_var, color_palette, plot_options):
    """Crée un box plot pour l'analyse mixte."""
    # Nettoyage des données
    data = df.copy()
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    data[qual_var] = data[qual_var].replace(missing_values, np.nan)
    data[quant_var] = data[quant_var].replace(missing_values, np.nan)
    data = data.dropna(subset=[qual_var, quant_var])
    
    fig = go.Figure()
    
    for i, modalite in enumerate(sorted(data[qual_var].unique())):
        subset = data[data[qual_var] == modalite][quant_var]
        fig.add_trace(go.Box(
            y=subset,
            name=str(modalite),
            marker_color=color_palette[i % len(color_palette)]
        ))
    
    fig.update_layout(
        title=plot_options['title'],
        yaxis_title=plot_options['y_label'],
        xaxis_title=plot_options['x_label'],
        showlegend=False,
        height=600,
        margin=dict(t=100, b=100),
        plot_bgcolor='white'
    )
    
    return fig

def plot_quantitative_bivariate_interactive(df, var_x, var_y, color_scheme, plot_options, groupby_col=None, agg_method=None):
    """Crée un scatter plot interactif pour l'analyse quantitative."""
    df_clean = df.dropna(subset=[var_x, var_y])
    regression_coeffs, regression_success = calculate_regression(
        df_clean[var_x].values,
        df_clean[var_y].values
    )
    
    fig = go.Figure()
    
    # Nuage de points
    hover_text = []
    for idx, row in df.iterrows():
        if pd.isna(row[var_x]) or pd.isna(row[var_y]):
            continue
        
        text_parts = []
        if groupby_col:
            text_parts.append(f"<b>{groupby_col}</b>: {row[groupby_col]}")
        text_parts.extend([
            f"<b>{var_x}</b>: {row[var_x]:,.2f}",
            f"<b>{var_y}</b>: {row[var_y]:,.2f}"
        ])
        hover_text.append("<br>".join(text_parts))
    
    fig.add_trace(go.Scatter(
        x=df[var_x],
        y=df[var_y],
        mode='markers',
        name='Observations',
        marker=dict(
            color=color_scheme[0],
            size=10,
            opacity=0.7
        ),
        hovertext=hover_text,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        )
    ))
    
    # Ligne de régression
    if regression_success:
        x_range = np.linspace(df[var_x].min(), df[var_x].max(), 100)
        y_range = regression_coeffs[0] * x_range + regression_coeffs[1]
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name=f'Régression (y = {regression_coeffs[0]:.2f}x + {regression_coeffs[1]:.2f})',
            line=dict(
                color=color_scheme[0],
                dash='dash'
            )
        ))
    
    # Configuration du layout
    title = plot_options['title']
    if groupby_col and agg_method:
        title += f"<br><sup>Données agrégées par {groupby_col} ({agg_method})</sup>"
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title=plot_options['x_label'],
        yaxis_title=plot_options['y_label'],
        hovermode='closest',
        plot_bgcolor='white',
        width=900,
        height=600,
        margin=dict(t=100, b=100),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

def add_annotations(fig, source=None, note=None, is_treemap=False):
    """
    Ajoute les annotations (source et note) au graphique de manière adaptée selon le type.
    
    Args:
        fig: Figure plotly
        source: Source des données
        note: Note explicative
        is_treemap: Boolean indiquant si c'est un treemap (nécessite un traitement spécial)
    """
    annotations = []
    
    if is_treemap:
        # Pour les treemaps, positionner les annotations différemment
        current_y = -0.1
        x_pos = 0.02
    else:
        # Pour les autres types de graphiques
        current_y = -0.2
        x_pos = 0.02
    
    if source:
        annotations.append(dict(
            text=f"Source : {source}",
            xref="paper", yref="paper",
            x=x_pos, y=current_y,
            showarrow=False,
            font=dict(size=10),
            align="left"
        ))
        current_y -= 0.07
    
    if note:
        annotations.append(dict(
            text=f"Note : {note}",
            xref="paper", yref="paper",
            x=x_pos, y=current_y,
            showarrow=False,
            font=dict(size=10),
            align="left"
        ))
    
    fig.update_layout(annotations=annotations)
    return fig

# Fonctions d'analyse bivariée
def analyze_qualitative_bivariate(df, var_x, var_y, exclude_missing=True):
    """Analyse bivariée pour deux variables qualitatives."""
    data = df.copy()
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    
    data[var_x] = data[var_x].replace(missing_values, np.nan)
    data[var_y] = data[var_y].replace(missing_values, np.nan)
    
    if exclude_missing:
        data = data.dropna(subset=[var_x, var_y])
        response_rate_x = (df[var_x].notna().sum() / len(df)) * 100
        response_rate_y = (df[var_y].notna().sum() / len(df)) * 100
        response_stats = {
            f"{var_x}": f"{response_rate_x:.1f}%",
            f"{var_y}": f"{response_rate_y:.1f}%"
        }
    
    crosstab_n = pd.crosstab(data[var_x], data[var_y])
    crosstab_pct = pd.crosstab(data[var_x], data[var_y], normalize='index') * 100
    col_means = crosstab_pct.mean()
    
    combined_table = pd.DataFrame(index=crosstab_n.index, columns=crosstab_n.columns)
    
    for idx in crosstab_n.index:
        for col in crosstab_n.columns:
            n = crosstab_n.loc[idx, col]
            pct = crosstab_pct.loc[idx, col]
            combined_table.loc[idx, col] = f"{pct:.1f}% ({n})"
    
    row_totals = crosstab_n.sum(axis=1)
    combined_table['Total'] = [f"100% ({n})" for n in row_totals]
    
    mean_row = []
    for col in crosstab_n.columns:
        mean_val = col_means[col]
        total_n = crosstab_n[col].sum()
        mean_row.append(f"{mean_val:.1f}% ({total_n})")
    mean_row.append(f"100% ({crosstab_n.values.sum()})")
    
    combined_table.loc['Moyenne'] = mean_row
    
    return (combined_table, response_stats) if exclude_missing else combined_table

def analyze_mixed_bivariate(df, qual_var, quant_var):
    """Analyse bivariée pour une variable qualitative et une quantitative."""
    data = df.copy()
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    data[qual_var] = data[qual_var].replace(missing_values, np.nan)
    data[quant_var] = data[quant_var].replace(missing_values, np.nan)
    data = data.dropna(subset=[qual_var, quant_var])
    
    stats_df = data.groupby(qual_var)[quant_var].agg([
        ('Effectif', 'count'),
        ('Total', lambda x: x.sum()),
        ('Moyenne', 'mean'),
        ('Médiane', 'median'),
        ('Écart-type', 'std'),
        ('Minimum', 'min'),
        ('Maximum', 'max')
    ]).round(2)
    
    total_stats = pd.DataFrame({
        'Effectif': data[quant_var].count(),
        'Total': data[quant_var].sum(),
        'Moyenne': data[quant_var].mean(),
        'Médiane': data[quant_var].median(),
        'Écart-type': data[quant_var].std(),
        'Minimum': data[quant_var].min(),
        'Maximum': data[quant_var].max()
    }, index=['Total']).round(2)
    
    stats_df = pd.concat([stats_df, total_stats])
    response_rate = (data[qual_var].count() / len(df)) * 100
    
    return stats_df, response_rate

def analyze_quantitative_bivariate(df, var_x, var_y, groupby_col=None, agg_method='sum'):
    """Analyse bivariée pour deux variables quantitatives."""
    data = df.copy()
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    
    data[var_x] = data[var_x].replace(missing_values, np.nan)
    data[var_y] = data[var_y].replace(missing_values, np.nan)
    if groupby_col:
        data[groupby_col] = data[groupby_col].replace(missing_values, np.nan)
    
    if groupby_col is not None:
        data = data.dropna(subset=[var_x, var_y, groupby_col])
        data = data.groupby(groupby_col).agg({
            var_x: agg_method,
            var_y: agg_method
        }).reset_index()
    else:
        data = data.dropna(subset=[var_x, var_y])
    
    # Test de normalité et corrélation
    is_normal_x = check_normality(data, var_x)
    is_normal_y = check_normality(data, var_y)
    
    if is_normal_x and is_normal_y:
        correlation_method = "Pearson"
        correlation, p_value = stats.pearsonr(data[var_x], data[var_y])
    else:
        correlation_method = "Spearman"
        correlation, p_value = stats.spearmanr(data[var_x], data[var_y])
    
    results_dict = {
        "Test de corrélation": [correlation_method],
        "Coefficient": [round(correlation, 3)],
        "P-value": [round(p_value, 3)],
        "Interprétation": ["Significatif" if p_value < 0.05 else "Non significatif"],
        "Nombre d'observations": [len(data)]
    }
    
    if groupby_col is not None:
        results_dict["Note"] = [f"Données agrégées par {groupby_col} ({agg_method})"]
    
    results_df = pd.DataFrame(results_dict)
    response_rate_x = (df[var_x].count() / len(df)) * 100
    response_rate_y = (df[var_y].count() / len(df)) * 100
    
    return results_df, response_rate_x, response_rate_y

def plot_qualitative_bivariate(df, var_x, var_y, plot_type, color_palette, plot_options):
    """
    Crée un graphique pour l'analyse bivariée de deux variables qualitatives.
    """
    # Calcul du tableau croisé
    crosstab = pd.crosstab(df[var_x], df[var_y], normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if plot_type == "Grouped Bar Chart":
        crosstab.plot(kind='bar', ax=ax, color=color_palette)
        
    elif plot_type == "Stacked Bar Chart":
        crosstab.plot(kind='bar', stacked=True, ax=ax, color=color_palette)
        
    elif plot_type == "Mosaic Plot":
        # Normaliser les données pour le mosaic plot
        contingency = pd.crosstab(df[var_x], df[var_y])
        from matplotlib.patches import Rectangle
        
        # Calculer les dimensions pour chaque rectangle
        total = contingency.sum().sum()
        width = 1.0
        x = 0
        
        # Dessiner les rectangles
        for i, (idx, row) in enumerate(contingency.iterrows()):
            height = row.sum() / total
            y = 0
            for j, val in enumerate(row):
                rect_height = (val / row.sum()) * height
                rect = Rectangle((x, y), width/len(contingency), rect_height,
                               facecolor=color_palette[j % len(color_palette)],
                               edgecolor='white')
                ax.add_patch(rect)
                y += rect_height
            x += width/len(contingency)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Configuration commune
    plt.title(plot_options['title'])
    plt.xlabel(plot_options['x_label'])
    plt.ylabel(plot_options['y_label'])
    
    # Rotation des étiquettes
    plt.xticks(rotation=45, ha='right')
    
    # Légende
    plt.legend(title=var_y, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Grille
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Ajustement de la mise en page
    plt.tight_layout()
    
    return fig

def detect_variable_to_aggregate(df, var_x, var_y, groupby_col):
    """
    Détecte quelles variables doivent être agrégées et lesquelles doivent être conservées telles quelles.
    """
    vars_to_aggregate = []
    vars_to_keep_raw = []
    
    for var in [var_x, var_y]:
        if df.groupby(groupby_col)[var].nunique().max() > 1:
            vars_to_aggregate.append(var)
        else:
            vars_to_keep_raw.append(var)
    
    return vars_to_aggregate, vars_to_keep_raw

# Fonctions de gestion des indicateurs
def show_indicator_form(statistics, analysis_type, variables_info):
    """Interface de création d'indicateur."""
    st.write("### Test création d'indicateur")
    
    with st.form("test_indicator_form"):
        name = st.text_input("Nom de l'indicateur")
        description = st.text_input("Description")
        creation_date = datetime.now().strftime("%Y-%m-%d")
        
        if st.form_submit_button("Enregistrer"):
            try:
                save_test_indicator({
                    "name": name,
                    "description": description,
                    "creation_date": creation_date
                })
                st.success("✅ Test d'enregistrement réussi!")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")

def save_test_indicator(test_data):
    """Sauvegarde un indicateur de test."""
    try:
        grist_data = {
            "records": [
                {
                    "fields": {
                        "name": test_data["name"],
                        "description": test_data["description"],
                        "creation_date": test_data["creation_date"]
                    }
                }
            ]
        }
        
        response = grist_api_request("4", method="POST", data=grist_data)
        return response
    except Exception as e:
        raise Exception(f"Erreur test : {str(e)}")

# Structure principale de l'application
def create_interactive_qualitative_table(data_series, var_name, exclude_missing=False, missing_label="Non réponse"):
    """Crée un tableau statistique interactif pour les variables qualitatives."""
    # Définition des valeurs manquantes
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'NA', 'nr', 'NR']

    # Initialisation du state si nécessaire
    if 'original_data' not in st.session_state:
        st.session_state.original_data = data_series.copy()
        st.session_state.groupings = []
        st.session_state.current_data = data_series.copy()

    # Préparation des données
    processed_series = st.session_state.current_data.copy()
    if exclude_missing:
        processed_series = processed_series.replace(missing_values, np.nan).dropna()
    else:
        processed_series = processed_series.replace(missing_values, missing_label)
    
    # Création du DataFrame initial
    value_counts = processed_series.value_counts().reset_index()
    value_counts.columns = ['Modalité', 'Effectif']
    value_counts['Taux (%)'] = (value_counts['Effectif'] / len(processed_series) * 100).round(2)
    value_counts['Nouvelle modalité'] = value_counts['Modalité']

    # Configuration des options avancées dans un expander
    with st.expander("Options avancées du tableau statistique"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("##### Édition des modalités")
            
            # Interface pour créer un nouveau regroupement
            st.write("**Nouveau regroupement**")
            
            # Obtenir toutes les modalités disponibles
            available_modalities = value_counts['Modalité'].tolist()
            
            # Sélection multiple des modalités à regrouper
            selected_modalities = st.multiselect(
                "Sélectionner les modalités à regrouper",
                options=available_modalities
            )

            if selected_modalities:
                new_group_name = st.text_input(
                    "Nom du nouveau groupe",
                    value=f"Groupe {', '.join(selected_modalities)}"
                )

                if st.button("Appliquer le regroupement"):
                    # Sauvegarder le regroupement
                    st.session_state.groupings.append({
                        'modalites': selected_modalities,
                        'nouveau_nom': new_group_name
                    })
                    
                    # Appliquer tous les regroupements depuis le début
                    processed_series = st.session_state.original_data.copy()
                    for group in st.session_state.groupings:
                        processed_series = processed_series.replace(
                            group['modalites'],
                            group['nouveau_nom']
                        )
                    
                    st.session_state.current_data = processed_series
                    st.rerun()

            # Afficher les regroupements existants
            if st.session_state.groupings:
                st.write("**Regroupements existants:**")
                for idx, group in enumerate(st.session_state.groupings):
                    st.info(
                        f"Groupe {idx + 1}: {', '.join(group['modalites'])} → {group['nouveau_nom']}"
                    )

            # Bouton pour réinitialiser
            if st.button("Réinitialiser tous les regroupements"):
                st.session_state.groupings = []
                st.session_state.current_data = st.session_state.original_data.copy()
                st.rerun()

            st.write("##### Renommer les modalités")
            for idx, row in value_counts.iterrows():
                value_counts.at[idx, 'Nouvelle modalité'] = st.text_input(
                    f"Renommer '{row['Modalité']}'",
                    value=row['Nouvelle modalité'],
                    key=f"modal_{idx}",
                    label_visibility="collapsed"
                )

        with col2:
            st.write("##### Paramètres du tableau")
            table_title = st.text_input(
                "Titre du tableau",
                value=f"Distribution de la variable {var_name}"
            )
            var_name_display = st.text_input(
                "Nom de la variable :",
                value="Modalités"
            )
            table_source = st.text_input(
                "Source",
                placeholder="Ex: Enquête XX, 2023"
            )
            table_note = st.text_input(
                "Note de lecture",
                placeholder="Ex: Lecture : XX% des répondants..."
            )

        with col3:
            st.write("##### Gestion des non-réponses")
            exclude_missing = st.checkbox("Exclure les non-réponses")
            if not exclude_missing:
                missing_label = st.text_input(
                    "Libellé pour les non-réponses",
                    value="Non réponse"
                )
            # Application immédiate du traitement des non-réponses
            if exclude_missing:
                processed_series = data_series.replace(missing_values, np.nan).dropna()
            else:
                processed_series = data_series.replace(missing_values, missing_label)
            
            # Seulement mettre à jour value_counts si on n'a pas de regroupements actifs
            if not st.session_state.groupings:
                value_counts = processed_series.value_counts().reset_index()
                value_counts.columns = ['Modalité', 'Effectif']
                value_counts['Taux (%)'] = (value_counts['Effectif'] / len(processed_series) * 100).round(2)
                value_counts['Nouvelle modalité'] = value_counts['Modalité']
                st.session_state.value_counts = value_counts.copy()

    # Création du DataFrame final
    final_df = value_counts.copy()
    final_df['Modalité'] = final_df['Nouvelle modalité']
    final_df = final_df.drop('Nouvelle modalité', axis=1)
    
    # Renommer la première colonne avec le nom de variable personnalisé
    final_df.columns = [var_name_display, 'Effectif', 'Taux (%)']

    # CSS personnalisé pour le tableau
    st.markdown("""
        <style>
        [data-testid="stDataFrame"] > div {
            width: auto !important;
            max-width: 800px !important;
            margin: 0 auto;
        }
        
        .dataframe {
            width: 100% !important;
            margin: 0 !important;
        }
        
        .dataframe td, .dataframe th {
            text-align: center !important;
            white-space: nowrap !important;
            padding: 8px !important;
        }
        
        .dataframe td:first-child {
            text-align: left !important;
        }
        
        .dataframe td:first-child { width: 60% !important; }
        .dataframe td:nth-child(2) { width: 20% !important; }
        .dataframe td:nth-child(3) { width: 20% !important; }
        </style>
    """, unsafe_allow_html=True)

    # Style du tableau
    styled_df = final_df.style\
        .format({
            'Effectif': '{:,.0f}',
            'Taux (%)': '{:.1f}%'
        })\
        .set_properties(**{
            'font-family': 'Marianne, sans-serif',
            'font-size': '14px',
            'padding': '8px'
        })\
        .set_table_styles([
            {'selector': 'th',
             'props': [
                 ('background-color', '#f0f2f6'),
                 ('color', '#262730'),
                 ('font-weight', 'bold'),
                 ('text-align', 'center'),
                 ('padding', '10px'),
                 ('font-size', '14px')
             ]},
            {'selector': 'td:nth-child(1)',
             'props': [
                 ('text-align', 'left'),
                 ('width', '60%'),
                 ('padding-left', '15px')
             ]},
            {'selector': 'td:nth-child(2)',
             'props': [
                 ('text-align', 'center'),
                 ('width', '20%')
             ]},
            {'selector': 'td:nth-child(3)',
             'props': [
                 ('text-align', 'center'),
                 ('width', '20%')
             ]},
            {'selector': 'tbody tr:nth-child(even)',
             'props': [('background-color', '#f9f9f9')]},
            {'selector': 'tbody tr:nth-child(odd)',
             'props': [('background-color', 'white')]}
        ])

    # Affichage du tableau et des métadonnées
    if table_title:
        st.markdown(f"### {table_title}")

    # Conteneur personnalisé pour le tableau
    st.markdown('<div style="max-width: 800px; margin: 0 auto;">', unsafe_allow_html=True)
    st.dataframe(
        styled_df,
        hide_index=True,
        height=min(35 * (len(final_df) + 1), 400)
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Source et note
    if table_source or table_note:
        st.markdown('<div style="max-width: 800px; margin: 5px auto;">', unsafe_allow_html=True)
        if table_source:
            st.caption(f"Source : {table_source}")
        if table_note:
            st.caption(f"Note : {table_note}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Options d'export
    with st.expander("Options d'export"):
        col1, col2 = st.columns(2)
        
        with col1:
                    if st.button("Exporter en image"):
                        # Création de la figure avec un style personnalisé
                        fig, ax = plt.subplots(figsize=(12, len(final_df) + 2))
                        ax.axis('off')
                        
                        # Configuration du style de base
                        plt.rcParams['font.family'] = 'sans-serif'
                        plt.rcParams['font.sans-serif'] = ['Arial']
                        
                        # Préparation des données pour l'affichage
                        cell_text = final_df.values.astype(str)
                        
                        # Création du tableau
                        table = ax.table(
                            cellText=cell_text,
                            colLabels=[var_name_display, 'Effectif', 'Taux (%)'],  # Utiliser le nom personnalisé ici
                            loc='center',
                            cellLoc='center',
                            bbox=[0, 0.1, 1, 0.9]
                        )
                        
                        # Style du tableau
                        table.auto_set_font_size(False)
                        table.set_fontsize(9)
                        
                        # Largeurs des colonnes
                        col_widths = [0.6, 0.2, 0.2]
                        for idx, width in enumerate(col_widths):
                            table.auto_set_column_width([idx])
                            for cell in table._cells:
                                if cell[1] == idx:
                                    table._cells[cell].set_width(width)
        
                        # Style des en-têtes
                        header_color = '#f0f2f6'
                        header_text_color = '#262730'
                        for j, cell in enumerate(table._cells[(0, j)] for j in range(len(final_df.columns))):
                            cell.set_facecolor(header_color)
                            cell.set_text_props(weight='bold', color=header_text_color)
                            cell.set_height(0.1)
                            cell.set_edgecolor('#e6e6e6')
        
                        # Style des cellules
                        for i in range(len(final_df) + 1):  # +1 pour inclure l'en-tête
                            for j in range(len(final_df.columns)):
                                cell = table._cells[(i, j)]
                                cell.set_edgecolor('#e6e6e6')
                                
                                # Alignement du texte
                                if j == 0 and i > 0:  # Première colonne (Modalités) mais pas l'en-tête
                                    cell.get_text().set_horizontalalignment('left')
                                    # Ajouter un peu d'espace à gauche
                                    cell.get_text().set_x(0.1)  # Ajuster cette valeur entre 0 et 1 pour le padding gauche
                                
                                # Lignes alternées
                                if i > 0:  # Exclure l'en-tête
                                    if i % 2 == 0:
                                        cell.set_facecolor('#f9f9f9')
                                    else:
                                        cell.set_facecolor('white')
        
                                # Ajustement de la hauteur des cellules
                                cell.set_height(0.05)
        
                        # Titre
                        if table_title:
                            plt.title(table_title, pad=20, fontsize=12, fontweight='bold')
                        
                        # Notes de bas de page
                        footer_text = []
                        if table_source:
                            footer_text.append(f"Source : {table_source}")
                        if table_note:
                            footer_text.append(f"Note : {table_note}")
                        
                        if footer_text:
                            plt.figtext(0.1, 0.02, '\n'.join(footer_text), fontsize=8)
                        
                        # Ajustement de la mise en page
                        plt.tight_layout()
                        
                        # Sauvegarde avec fond blanc
                        buf = BytesIO()
                        plt.savefig(buf, format='png', 
                                  bbox_inches='tight', 
                                  dpi=300,
                                  facecolor='white',
                                  edgecolor='none',
                                  pad_inches=0.1)
                        plt.close()
                        
                        # Téléchargement
                        st.download_button(
                            label="Télécharger l'image",
                            data=buf.getvalue(),
                            file_name="tableau_statistique.png",
                            mime="image/png"
                        )

        with col2:
            if st.button("Copier pour Excel"):
                # Préparation des données pour Excel
                excel_data = []
                if table_title:
                    excel_data.append(table_title)
                    excel_data.append("")  # Ligne vide

                # En-têtes et données
                excel_data.append("\t".join(final_df.columns))
                for _, row in final_df.iterrows():
                    excel_data.append("\t".join(str(val) for val in row))

                # Métadonnées
                if table_source or table_note:
                    excel_data.append("")  # Ligne vide
                    if table_source:
                        excel_data.append(f"Source : {table_source}")
                    if table_note:
                        excel_data.append(f"Note : {table_note}")

                # Conversion en texte tabulé
                copy_text = "\n".join(excel_data)

                # Affichage dans un textarea pour faciliter la copie
                st.text_area(
                    "Copiez le texte ci-dessous pour Excel :",
                    value=copy_text,
                    height=150
                )
    
    return final_df, var_name_display  # Return the DataFrame and the dynamic column name

def display_univariate_analysis(data, var):
    """Gère l'affichage de l'analyse univariée."""
    plot_data = data[var].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(plot_data)
    
    # Initialiser toutes les variables qui seront utilisées plus tard
    grouping_method = "Aucune"  # Valeur par défaut
    grouped_data = None
    value_counts = None
    value_type = "Effectif"
    is_integer_variable = False
    
    st.write(f"### Statistiques principales de la variable {var}")
    
    # Initialize value_type and grouping_method with default values
    value_type = "Effectif"
    grouping_method = "Aucune"
    
    if is_numeric:
        # Statistiques numériques
        stats_df = pd.DataFrame({
            'Statistique': ['Effectif total', 'Somme', 'Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum'],
            'Valeur': [
                len(plot_data),
                plot_data.sum().round(2),
                plot_data.mean().round(2),
                plot_data.median().round(2),
                plot_data.std().round(2),
                plot_data.min(),
                plot_data.max()
            ]
        })
        create_interactive_stats_table(stats_df)

        is_integer_variable = all(float(x).is_integer() for x in plot_data)
        
        # Options de regroupement
        grouping_method = st.selectbox(
            "Méthode de regroupement",
            ["Aucune", "Quantile", "Manuelle"]
        )
        
        if grouping_method != "Aucune":
            if grouping_method == "Quantile":
                quantile_type = st.selectbox(
                    "Type de regroupement",
                    ["Quartile (4 groupes)", "Quintile (5 groupes)", "Décile (10 groupes)"]
                )
                n_groups = {"Quartile (4 groupes)": 4, "Quintile (5 groupes)": 5, "Décile (10 groupes)": 10}[quantile_type]
                labels = [f"{i}er {quantile_type.split(' ')[0].lower()}" if i == 1 else f"{i}ème {quantile_type.split(' ')[0].lower()}" 
                         for i in range(1, n_groups + 1)]
                grouped_data = pd.qcut(plot_data, q=n_groups, labels=labels)
                
            else:  # Manuelle
                n_groups = st.number_input("Nombre de groupes", min_value=2, value=3)
                breaks = []
                for i in range(n_groups + 1):
                    if i == 0:
                        val = plot_data.min()
                    elif i == n_groups:
                        val = plot_data.max()
                    else:
                        val = st.number_input(
                            f"Seuil {i}",
                            value=float(plot_data.min() + (i/n_groups)*(plot_data.max()-plot_data.min())),
                            step=1 if is_integer_variable else 0.1
                        )
                    breaks.append(val)
                grouped_data = pd.cut(plot_data, bins=breaks)
    else:
        # Statistiques qualitatives
        value_counts, var_name_display = create_interactive_qualitative_table(
            plot_data, 
            var, 
            exclude_missing=exclude_missing if 'exclude_missing' in locals() else False,
            missing_label=missing_label if 'missing_label' in locals() else "Non réponse"
        )

    # Configuration de la visualisation
    st.write("### Configuration de la visualisation")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if is_numeric:
            if grouping_method == "Aucune":
                graph_type = st.selectbox("Type de graphique", ["Histogramme", "Density plot"])
            else:
                graph_type = st.selectbox("Type de graphique", ["Bar plot", "Lollipop plot", "Treemap"])
        else:
            graph_type = st.selectbox("Type de graphique", ["Bar plot", "Lollipop plot", "Treemap"])

    with col2:
        color_scheme = st.selectbox("Palette de couleurs", list(COLOR_PALETTES.keys()))

    # Options avancées
    with st.expander("Options avancées"):
        adv_col1, adv_col2, adv_col3 = st.columns(3) 
        with adv_col1:
            title = st.text_input("Titre du graphique", f"Distribution de {var}")
            x_axis = st.text_input("Titre de l'axe X", var)
            y_axis = st.text_input("Titre de l'axe Y", "Valeur")
        with adv_col2:
            source = st.text_input("Source des données", "")
            note = st.text_input("Note de lecture", "")
            show_values = st.checkbox("Afficher les valeurs", True)
            if not is_numeric or (is_numeric and grouping_method != "Aucune"):
                value_type = st.radio("Type de valeur à afficher", ["Effectif", "Taux (%)"])
        with adv_col3:
            # gestion des non-réponses
            st.markdown("##### Gestion des non-réponses")
            exclude_missing = st.checkbox("Exclure les non-réponses", key="exclude_missing")
            if not exclude_missing:
                missing_label = st.text_input("Libellé pour les non-réponses", "Non réponse", key="missing_label")

    # Définition de value_type en dehors des options avancées pour les variables qualitatives
    if not is_numeric:
        value_type = st.radio("Type de valeur à afficher", ["Effectif", "Taux (%)"])
    
    # Génération du graphique
    if st.button("Générer la visualisation"):
        try:
            # Préparation des données
            if not is_numeric:
                data_to_plot = value_counts.copy()  # Ensure value_counts is a DataFrame
                if value_type == "Taux (%)":
                    data_to_plot['Effectif'] = data_to_plot['Taux (%)']
                    y_axis = "Taux (%)" if y_axis == "Valeur" else y_axis
            else:
                if grouping_method == "Aucune":
                    data_to_plot = plot_data  # plot_data should be a Series
                else:
                    value_counts = grouped_data.value_counts().reset_index()
                    value_counts.columns = ['Modalité', 'Effectif']
                    data_to_plot = value_counts.copy()
                    if value_type == "Taux (%)":
                        data_to_plot['Effectif'] = (data_to_plot['Effectif'] / len(plot_data) * 100).round(2)
                        y_axis = "Taux (%)"

            # Création du graphique
            if is_numeric and grouping_method == "Aucune":
                if graph_type == "Histogramme":
                    fig = px.histogram(data_to_plot, title=title,
                                     color_discrete_sequence=COLOR_PALETTES[color_scheme])
                else:
                    fig = plot_density(data_to_plot, var, title, x_axis, y_axis)
            else:
                # Dynamically find the column name for 'Modalité'
                modality_col = 'Modalité' if 'Modalité' in data_to_plot.columns else data_to_plot.columns[0]
                
                if graph_type == "Bar plot":
                    fig = plot_qualitative_bar(data_to_plot.rename(columns={modality_col: 'Modalité'}), title, x_axis, y_axis,
                                             COLOR_PALETTES[color_scheme], show_values)
                elif graph_type == "Lollipop plot":
                    fig = plot_qualitative_lollipop(data_to_plot.rename(columns={modality_col: 'Modalité'}), title, x_axis, y_axis,
                                                  COLOR_PALETTES[color_scheme], show_values)
                else:
                    fig = plot_qualitative_treemap(data_to_plot.rename(columns={modality_col: 'Modalité'}), title, COLOR_PALETTES[color_scheme])

            # Ajout des annotations
            if source or note:
                fig = add_annotations(fig, source, note)

            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique : {str(e)}")
            st.error(f"Détails : {str(type(e).__name__)}")
            
def main():
    st.title("Analyse des données ESR")

    # Initialisation de l'état de session pour les données fusionnées
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None

    # Sélection des tables
    tables_dict = get_grist_tables()
    if not tables_dict:
        st.error("Aucune table disponible.")
        return

    # Choix du mode de sélection
    selection_mode = st.radio(
        "Mode de sélection des tables",
        ["Une seule table", "Plusieurs tables"]
    )

    if selection_mode == "Une seule table":
        # Sélection d'une seule table avec selectbox
        table_name = st.selectbox(
            "Sélectionnez la table à analyser",
            options=list(tables_dict.keys())
        )
        
        if table_name:
            table_id = tables_dict[table_name]  # Obtenir l'ID correspondant au nom
            # Debug: afficher l'ID utilisé
            print(f"ID de la table sélectionnée: {table_id}")
            df = get_grist_data(table_id)
            if df is not None:
                st.session_state.merged_data = df
            else:
                st.error("Impossible de charger la table sélectionnée.")
                return
    else:
        # Sélection multiple avec multiselect
        table_names = st.multiselect(
            "Sélectionnez les tables à analyser", 
            options=list(tables_dict.keys())  # Les clés sont les noms affichables
        )

        if len(table_names) > 1:  # Seulement si plus d'une table est sélectionnée
            # Stocker les noms des tables sélectionnées
            table_selections = table_names
            
            # Chargement des DataFrames
            dataframes = []
            merge_configs = []
        
            for table_name in table_selections:
                table_id = tables_dict[table_name]
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
        
        if not table_names:
            st.warning("Veuillez sélectionner au moins une table pour l'analyse.")
            return

        if len(table_names) == 1:
            table_id = tables_dict[table_names[0]]
            df = get_grist_data(table_id)
            if df is not None:
                st.session_state.merged_data = df
            else:
                st.error("Impossible de charger la table sélectionnée.")
                return
        else:
            dataframes = []
            merge_configs = []

            for table_name in table_names:
                table_id = tables_dict[table_name]
                df = get_grist_data(table_id)
                if df is not None:
                    dataframes.append(df)

            if len(dataframes) < 2:
                st.warning("Impossible de charger les tables sélectionnées.")
                return
        
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
        
    if analysis_type == "Analyse univariée":
        # Sélection de la variable
        var = st.selectbox(
            "Sélectionnez la variable:", 
            options=["---"] + list(st.session_state.merged_data.columns)
        )
        
        if var != "---":
            # Préparation des données
            plot_data = st.session_state.merged_data[var].copy()
            plot_data = plot_data.dropna()
            
            if plot_data is not None and not plot_data.empty:
                # Détection du type de variable
                is_numeric = pd.api.types.is_numeric_dtype(plot_data)
                st.write(f"### Statistiques principales de la variable {var}")
                
                if is_numeric:
                    # Gestion des doublons pour variables numériques
                    has_duplicates = st.session_state.merged_data.duplicated(subset=[var]).any()
                    if has_duplicates:
                        st.warning("⚠️ Certaines observations sont répétées dans le jeu de données. "
                                 "Vous pouvez choisir d'agréger les données avant l'analyse.")
                        do_aggregate = st.checkbox("Agréger les données avant l'analyse")
                        
                        if do_aggregate:
                            groupby_cols = [col for col in st.session_state.merged_data.columns if col != var]
                            groupby_col = st.selectbox("Sélectionner la colonne d'agrégation", groupby_cols)
                            agg_method = st.radio(
                                "Méthode d'agrégation", 
                                ['sum', 'mean', 'median'],
                                format_func=lambda x: {
                                    'sum': 'Somme',
                                    'mean': 'Moyenne',
                                    'median': 'Médiane'
                                }[x]
                            )
                            clean_data = st.session_state.merged_data.dropna(subset=[var, groupby_col])
                            agg_data = clean_data.groupby(groupby_col).agg({var: agg_method}).reset_index()
                            plot_data = agg_data[var]
    
                    # Statistiques descriptives
                    stats_df = pd.DataFrame({
                        'Statistique': ['Effectif total', 'Somme', 'Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum'],
                        'Valeur': [
                            len(plot_data),
                            plot_data.sum().round(2),
                            plot_data.mean().round(2),
                            plot_data.median().round(2),
                            plot_data.std().round(2),
                            plot_data.min(),
                            plot_data.max()
                        ]
                    })
                    create_interactive_stats_table(stats_df)
    
                    if 'do_aggregate' in locals() and do_aggregate:
                        st.info("Note : Les statistiques sont calculées à l'échelle de la variable d'agrégation sélectionnée.")
    
                    # Options de regroupement
                    st.write("### Options de regroupement")
                    grouping_method = st.selectbox("Méthode de regroupement", ["Aucune", "Quantile", "Manuelle"])
                    is_integer_variable = all(float(x).is_integer() for x in plot_data)
    
                    if grouping_method == "Quantile":
                        quantile_type = st.selectbox(
                            "Type de regroupement",
                            ["Quartile (4 groupes)", "Quintile (5 groupes)", "Décile (10 groupes)"]
                        )
                        n_groups = {"Quartile (4 groupes)": 4, "Quintile (5 groupes)": 5, "Décile (10 groupes)": 10}[quantile_type]
                        
                        # Création des labels personnalisés selon le type de quantile
                        if quantile_type == "Quartile (4 groupes)":
                            labels = [f"{i}er quartile" if i == 1 else f"{i}ème quartile" 
                                     for i in range(1, 5)]
                        elif quantile_type == "Quintile (5 groupes)":
                            labels = [f"{i}er quintile" if i == 1 else f"{i}ème quintile" 
                                     for i in range(1, 6)]
                        else:  # Déciles
                            labels = [f"{i}er décile" if i == 1 else f"{i}ème décile" 
                                     for i in range(1, 11)]
                        
                        # Création des groupes avec les labels personnalisés
                        grouped_data = pd.qcut(plot_data, q=n_groups, labels=labels)
                        value_counts = pd.DataFrame({
                            'Groupe': labels,
                            'Effectif': grouped_data.value_counts().reindex(labels)
                        })
                        
                        # Calcul des taux avec entiers si approprié
                        value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100)
                        if is_integer_variable:
                            value_counts['Effectif'] = value_counts['Effectif'].astype(int)
                            value_counts['Taux (%)'] = value_counts['Taux (%)'].apply(
                                lambda x: int(x) if x.is_integer() else round(x, 1)
                            )
                        
                        # Statistiques par groupe
                        group_stats = plot_data.groupby(grouped_data).agg(['sum', 'mean', 'max'])
                        if is_integer_variable:
                            group_stats = group_stats.applymap(lambda x: int(x) if float(x).is_integer() else round(x, 2))
                        else:
                            group_stats = group_stats.round(2)
                        group_stats.columns = ['Somme', 'Moyenne', 'Maximum']
                        
                        st.write("### Statistiques par groupe")
                        st.dataframe(pd.concat([value_counts, group_stats], axis=1))
    
                    elif grouping_method == "Manuelle":
                        n_groups = st.number_input("Nombre de groupes", min_value=2, value=3)
                        breaks = []
                        
                        # Conversion en float pour assurer la cohérence des types
                        min_val = float(plot_data.min())
                        max_val = float(plot_data.max())
                        
                        for i in range(n_groups + 1):
                            if i == 0:
                                val = min_val
                            elif i == n_groups:
                                val = max_val
                            else:
                                suggested_val = min_val + (i/n_groups)*(max_val-min_val)
                                if is_integer_variable:
                                    val = st.number_input(
                                        f"Seuil {i}", 
                                        value=int(suggested_val),
                                        min_value=int(min_val),
                                        max_value=int(max_val),
                                        step=1
                                    )
                                else:
                                    val = st.number_input(
                                        f"Seuil {i}", 
                                        value=float(suggested_val),
                                        min_value=float(min_val),
                                        max_value=float(max_val),
                                        step=0.1
                                    )
                            breaks.append(val)
                        
                        grouped_data = pd.cut(plot_data, bins=breaks)
                        value_counts = grouped_data.value_counts().reset_index()
                        value_counts.columns = ['Groupe', 'Effectif']
                        
                        # Tri et formatage
                        value_counts = value_counts.sort_values('Groupe', key=lambda x: x.map(lambda y: y.left))
                        value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100)
                        
                        if is_integer_variable:
                            value_counts['Effectif'] = value_counts['Effectif'].astype(int)
                            value_counts['Taux (%)'] = value_counts['Taux (%)'].apply(
                                lambda x: int(x) if x.is_integer() else round(x, 1)
                            )
                        
                        st.write("### Répartition des groupes")
                        st.dataframe(value_counts)
    
                else:
                    # Statistiques pour variable qualitative
                    value_counts, var_name_display = create_interactive_qualitative_table(
                        plot_data, 
                        var, 
                        exclude_missing=exclude_missing if 'exclude_missing' in locals() else False,
                        missing_label=missing_label if 'missing_label' in locals() else "Non réponse"
                    )
                
                # Configuration de la visualisation
                st.write("### Configuration de la visualisation")
                viz_col1, viz_col2 = st.columns([1, 2])
                
                with viz_col1:
                    if is_numeric:
                        if grouping_method == "Aucune":
                            graph_type = st.selectbox("Type de graphique", ["Histogramme", "Density plot"], key="graph_type_no_group")
                        elif grouping_method == "Quantile":
                            # Changé la clé pour éviter les doublons
                            graph_type = st.selectbox(
                                "Type de graphique",
                                ["Boîte à moustaches", "Violin plot", "Box plot avec points"],
                                key="graph_type_quantile"  # Nouvelle clé unique
                            )
                        else:
                            graph_type = st.selectbox("Type de graphique", ["Bar plot", "Lollipop plot", "Treemap"], key="graph_type_group")
                    else:
                        graph_type = st.selectbox("Type de graphique", ["Bar plot", "Lollipop plot", "Treemap"], key="graph_type_qual")
                    if not is_numeric:
                        value_type = st.radio("Type de valeur à afficher", ["Effectif", "Taux (%)"], key="value_type_qual")
                with viz_col2:
                    color_scheme = st.selectbox("Palette de couleurs", list(COLOR_PALETTES.keys()), key="color_scheme")
                                
                # Options avancées générales
                with st.expander("Options avancées"):
                    adv_col1, adv_col2 = st.columns(2)
                    with adv_col1:
                        title = st.text_input("Titre du graphique", f"Distribution de {var}", key="title_adv")
                        x_axis = st.text_input("Titre de l'axe X", var, key="x_axis_adv")
                        y_axis = st.text_input("Titre de l'axe Y", "Valeur", key="y_axis_adv")
                    with adv_col2:
                        source = st.text_input("Source des données", "", key="source_adv")
                        note = st.text_input("Note de lecture", "", key="note_adv")
                        show_values = st.checkbox("Afficher les valeurs", True, key="show_values_adv")
                
                # Génération du graphique
                if st.button("Générer la visualisation"):
                    try:
                        if not is_numeric:
                            data_to_plot = value_counts.copy()
                            if value_type == "Taux (%)":
                                data_to_plot = pd.DataFrame({
                                    data_to_plot.columns[0]: data_to_plot[data_to_plot.columns[0]],
                                    'Effectif': data_to_plot['Taux (%)']
                                })
                        
                            # Création du graphique selon le type
                            if graph_type == "Bar plot":
                                fig = plot_qualitative_bar(data_to_plot, title, x_axis, y_axis, 
                                                        COLOR_PALETTES[color_scheme], show_values)
                            elif graph_type == "Lollipop plot":
                                fig = plot_qualitative_lollipop(data_to_plot, title, x_axis, y_axis, 
                                                            COLOR_PALETTES[color_scheme], show_values)
                            elif graph_type == "Treemap":
                                fig = plot_qualitative_treemap(data_to_plot, title, 
                                                            COLOR_PALETTES[color_scheme])

                            # Ajout des annotations si nécessaire
                            if fig is not None and (source or note):
                                is_treemap = (graph_type == "Treemap")
                                fig = add_annotations(fig, source, note, is_treemap=is_treemap)

                            else:  # Variables numériques
                                if grouping_method == "Aucune":
                                    if graph_type == "Histogramme":
                                        fig = px.histogram(plot_data, title=title, color_discrete_sequence=COLOR_PALETTES[color_scheme])
                                        if show_values:
                                            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                                    else:  # Density plot
                                        fig = plot_density(plot_data, var, title, x_axis, y_axis)
                                
                                elif grouping_method == "Quantile":
                                    fig = plot_quantile_distribution(
                                        data=plot_data,
                                        title=title,
                                        y_label=y_axis,
                                        color_palette=COLOR_PALETTES[color_scheme],
                                        plot_type=graph_type,
                                        is_integer_variable=is_integer_variable
                                    )
                                
                                else:  # Groupement manuel
                                    data_to_plot = pd.DataFrame({
                                        'Modalité': value_counts['Groupe'].astype(str),
                                        'Effectif': value_counts['Effectif' if value_type == "Effectif" else 'Taux (%)']
                                    })
                                    
                                    if graph_type == "Bar plot":
                                        fig = plot_qualitative_bar(data_to_plot, title, x_axis, y_axis, COLOR_PALETTES[color_scheme], show_values)
                                    elif graph_type == "Lollipop plot":
                                        fig = plot_qualitative_lollipop(data_to_plot, title, x_axis, y_axis, COLOR_PALETTES[color_scheme], show_values)
                                    else:  # Treemap
                                        fig = plot_qualitative_treemap(data_to_plot, title, COLOR_PALETTES[color_scheme])
                                
                                # Affichage du graphique avec annotations
                                if fig is not None and (source or note):
                                    is_treemap = (graph_type == "Treemap")
                                    fig = add_annotations(fig, source, note, is_treemap=is_treemap)

                                # Affichage final du graphique
                                if fig is not None:
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                    except Exception as e:
                        st.error(f"Erreur lors de la génération du graphique : {str(e)}")
                        st.error(f"Détails : {str(type(e).__name__)}")
    
            else:
                st.warning("Aucune donnée valide disponible pour cette variable")
        else:
            st.info("Veuillez sélectionner une variable à analyser")
    
    # Analyse bivariée
    elif analysis_type == "Analyse bivariée":
        try:
            # Initialisation de var_y dans session_state s'il n'existe pas
            if 'previous_var_y' not in st.session_state:
                st.session_state.previous_var_y = None
    
            # Ajout de l'option vide pour var_x avec une clé unique
            var_x = st.selectbox(
                "Variable X", 
                ["---"] + list(st.session_state.merged_data.columns), 
                key='bivariate_var_x'
            )
    
            if var_x != "---":
                # Filtrer les colonnes pour var_y en excluant var_x
                available_columns_y = [col for col in st.session_state.merged_data.columns if col != var_x]
                
                # Si la valeur précédente de var_y est valide, la mettre en premier dans la liste
                if st.session_state.previous_var_y in available_columns_y:
                    available_columns_y.remove(st.session_state.previous_var_y)
                    available_columns_y.insert(0, st.session_state.previous_var_y)
    
                # Ajout de l'option vide pour var_y avec une clé unique
                var_y = st.selectbox(
                    "Variable Y", 
                    ["---"] + available_columns_y, 
                    key='bivariate_var_y'
                )
    
                # Ne continuer que si les deux variables sont sélectionnées
                if var_y != "---":
                    # Sauvegarder la valeur de var_y pour la prochaine itération
                    st.session_state.previous_var_y = var_y
    
                    # Détection des types de variables avec gestion d'erreur
                    try:
                        is_x_numeric = is_numeric_column(st.session_state.merged_data, var_x)
                        is_y_numeric = is_numeric_column(st.session_state.merged_data, var_y)
    
                        # Analyse pour deux variables qualitatives
                        if not is_x_numeric and not is_y_numeric:
                            st.write("### Analyse Bivariée - Variables Qualitatives")
       
                            # Option d'inversion des variables
                            invert_vars = st.checkbox("Inverser les variables X et Y", key='invert_vars_qual')
            
                            # Variables actuelles
                            current_x = var_y if invert_vars else var_x
                            current_y = var_x if invert_vars else var_y
            
                            # Affichage du tableau croisé avec les taux de réponse
                            combined_table, response_stats = analyze_qualitative_bivariate(
                                st.session_state.merged_data, current_x, current_y, exclude_missing=True
                            )
            
                            # Affichage des taux de réponse
                            st.write("Taux de réponse :")
                            for var, rate in response_stats.items():
                                st.write(f"- {var} : {rate}")
            
                            st.write("Tableau croisé (Pourcentages en ligne et effectifs)")
                            st.dataframe(combined_table)
            
                            # Configuration de la visualisation
                            st.write("### Configuration de la visualisation")
                            col1, col2 = st.columns(2)
            
                            with col1:
                                plot_type = st.selectbox(
                                    "Type de graphique",
                                    ["Grouped Bar Chart", "Stacked Bar Chart", "Mosaic Plot"],
                                    key='plot_type_qual'
                                )
            
                            with col2:
                                color_scheme = st.selectbox(
                                    "Palette de couleurs",
                                    list(COLOR_PALETTES.keys()),
                                    key='color_scheme_qual'
                                )
            
                            # Options avancées
                            with st.expander("Options avancées"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    title = st.text_input("Titre du graphique", 
                                                          f"Distribution de {current_x} par {current_y}",
                                                          key='title_qual')
                                    x_label = st.text_input("Titre de l'axe X", current_x,
                                                            key='x_label_qual')
                                    y_label = st.text_input("Titre de l'axe Y", "Valeur",
                                                            key='y_label_qual')
                                with col2:
                                    source = st.text_input("Source des données", "",
                                                           key='source_qual')
                                    note = st.text_input("Note de lecture", "",
                                                         key='note_qual')
                                    show_values = st.checkbox("Afficher les valeurs", True,
                                                              key='show_values_qual')
            
                            # Options du graphique
                            plot_options = {
                                'title': title,
                                'x_label': x_label,
                                'y_label': y_label,
                                'source': source,
                                'note': note,
                                'show_values': show_values
                            }
            
                            # Création et affichage du graphique
                            fig = plot_qualitative_bivariate(
                                st.session_state.merged_data,
                                current_x,
                                current_y,
                                plot_type,
                                COLOR_PALETTES[color_scheme],
                                plot_options
                            )
                            st.pyplot(fig)
                            plt.close()
            
                        # Analyse pour une variable qualitative et une quantitative
                        elif (is_x_numeric and not is_y_numeric) or (not is_x_numeric and is_y_numeric):
                            st.write("### Analyse Bivariée - Variable Qualitative et Quantitative")
            
                            # Réorganisation des variables (qualitative en X, quantitative en Y)
                            if is_x_numeric:
                                quant_var = var_x
                                qual_var = var_y
                                st.info("Les variables ont été réorganisées : variable qualitative en X et quantitative en Y")
                            else:
                                qual_var = var_x
                                quant_var = var_y
            
                            # Affichage des statistiques descriptives et du taux de réponse
                            stats_df, response_rate = analyze_mixed_bivariate(
                                st.session_state.merged_data, 
                                qual_var, 
                                quant_var
                            )
            
                            st.write(f"Taux de réponse : {response_rate:.1f}%")
                            st.write("Statistiques descriptives par modalité")
                            grid_response = create_interactive_stats_table(stats_df)
                            st.info("Note : Les statistiques de la ligne total sont calculées à l'échelle de l'unité d'observation de la table")
                            
                            # Bouton de création d'indicateur
                            if st.button("Créer un indicateur à partir de ces statistiques", key="create_indicator_mixed"):
                                variables_info = {
                                    'var_qual': qual_var,
                                    'var_quant': quant_var,
                                    'source': source if 'source' in locals() else None
                                }
                                show_indicator_form(stats_df.to_dict('records'), 'mixed', variables_info)
                            
                            # Configuration de la visualisation
                            st.write("### Configuration de la visualisation")
            
                            # Sélection de la palette de couleurs
                            color_scheme = st.selectbox(
                                "Palette de couleurs",
                                list(COLOR_PALETTES.keys()),
                                key='color_scheme_mixed'
                            )
            
                            # Options avancées
                            with st.expander("Options avancées"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    title = st.text_input(
                                        "Titre du graphique", 
                                        f"Distribution de {quant_var} par {qual_var}",
                                        key='title_mixed'
                                    )
                                    x_label = st.text_input(
                                        "Titre de l'axe X", 
                                        qual_var,
                                        key='x_label_mixed'
                                    )
                                    y_label = st.text_input(
                                        "Titre de l'axe Y", 
                                        quant_var,
                                        key='y_label_mixed'
                                    )
                                with col2:
                                    source = st.text_input(
                                        "Source des données", 
                                        "",
                                        key='source_mixed'
                                    )
                                    note = st.text_input(
                                        "Note de lecture", 
                                        "",
                                        key='note_mixed'
                                    )
            
                            # Options du graphique
                            plot_options = {
                                'title': title,
                                'x_label': x_label,
                                'y_label': y_label,
                                'source': source,
                                'note': note,
                                'show_values': True
                            }
            
                            # Création et affichage du graphique
                            fig = plot_mixed_bivariate(
                                st.session_state.merged_data,
                                qual_var,
                                quant_var,
                                COLOR_PALETTES[color_scheme],
                                plot_options
                            )
            
                            st.plotly_chart(fig, use_container_width=True)
            
                        # Analyse pour deux variables quantitatives
                        else:
                            st.write("### Analyse Bivariée - Variables Quantitatives")
            
                            # Détection des doublons potentiels
                            has_duplicates = check_duplicates(st.session_state.merged_data, var_x, var_y)
            
                            if has_duplicates:
                                st.warning("⚠️ Certaines observations sont répétées dans le jeu de données. "
                                          "Vous pouvez choisir d'agréger les données avant l'analyse.")
            
                                # Option d'agrégation
                                do_aggregate = st.checkbox("Agréger les données avant l'analyse")
            
                                if do_aggregate:
                                    # Sélection de la colonne d'agrégation
                                    groupby_cols = [col for col in st.session_state.merged_data.columns 
                                                    if col not in [var_x, var_y]]
                                    groupby_col = st.selectbox("Sélectionner la colonne d'agrégation", groupby_cols)
            
                                    # Méthode d'agrégation
                                    agg_method = st.radio("Méthode d'agrégation", 
                                                          ['sum', 'mean', 'median'],
                                                          format_func=lambda x: {
                                                              'sum': 'Somme',
                                                              'mean': 'Moyenne',
                                                              'median': 'Médiane'
                                                          }[x])
            
                                    # Détection des variables à agréger
                                    vars_to_aggregate, vars_to_keep_raw = detect_variable_to_aggregate(st.session_state.merged_data, var_x, var_y, groupby_col)
                                    agg_dict = {var: agg_method for var in vars_to_aggregate}
                                    agg_dict.update({var: 'first' for var in vars_to_keep_raw})
            
                                    # Création des données agrégées
                                    agg_data = st.session_state.merged_data.groupby(groupby_col).agg(agg_dict).reset_index()
            
                                    # Calcul et affichage des statistiques
                                    results_df, response_rate_x, response_rate_y = analyze_quantitative_bivariate(
                                        st.session_state.merged_data,
                                        var_x,
                                        var_y,
                                        groupby_col=groupby_col,
                                        agg_method=agg_method
                                    )
                                else:
                                    results_df, response_rate_x, response_rate_y = analyze_quantitative_bivariate(
                                        st.session_state.merged_data,
                                        var_x,
                                        var_y
                                    )
                                    agg_data = st.session_state.merged_data
                                    groupby_col = None
                                    agg_method = None
                            else:
                                results_df, response_rate_x, response_rate_y = analyze_quantitative_bivariate(
                                    st.session_state.merged_data,
                                    var_x,
                                    var_y
                                )
                                agg_data = st.session_state.merged_data
                                groupby_col = None
                                agg_method = None
            
                            # Affichage des taux de réponse
                            st.write("Taux de réponse :")
                            st.write(f"- {var_x} : {response_rate_x:.1f}%")
                            st.write(f"- {var_y} : {response_rate_y:.1f}%")
            
                            st.write("Statistiques de corrélation")
                            grid_response = create_interactive_stats_table(results_df)
                            
                            # Bouton de création d'indicateur
                            if st.button("Créer un indicateur à partir de ces statistiques", key="create_indicator_mixed"):
                                variables_info = {
                                    'var_x': var_x,
                                    'var_y': var_y,
                                    'source': source if 'source' in locals() else None
                                }
                                show_indicator_form(results_df.to_dict('records'), 'quantitative', variables_info)
            
                            # Configuration de la visualisation
                            st.write("### Configuration de la visualisation")
            
                            # Sélection de la palette de couleurs
                            color_scheme = st.selectbox(
                                "Palette de couleurs",
                                list(COLOR_PALETTES.keys()),
                                key='color_scheme_quant'
                            )
            
                            # Options avancées
                            with st.expander("Options avancées"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    title = st.text_input(
                                        "Titre du graphique", 
                                        f"Relation entre {var_x} et {var_y}",
                                        key='title_quant'
                                    )
                                    x_label = st.text_input(
                                        "Titre de l'axe X", 
                                        var_x,
                                        key='x_label_quant'
                                    )
                                    y_label = st.text_input(
                                        "Titre de l'axe Y", 
                                        var_y,
                                        key='y_label_quant'
                                    )
                                with col2:
                                    source = st.text_input(
                                        "Source des données", 
                                        "",
                                        key='source_quant'
                                    )
                                    note = st.text_input(
                                        "Note de lecture", 
                                        "",
                                        key='note_quant'
                                    )
            
                            # Options du graphique
                            plot_options = {
                                'title': title,
                                'x_label': x_label,
                                'y_label': y_label,
                                'source': source,
                                'note': note
                            }
            
                            # Création et affichage du graphique
                            fig = plot_quantitative_bivariate_interactive(
                                agg_data,
                                var_x,
                                var_y,
                                COLOR_PALETTES[color_scheme],
                                plot_options,
                                groupby_col if do_aggregate else None,
                                agg_method if do_aggregate else None
                            )
            
                            st.plotly_chart(fig, use_container_width=True)
            
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse des types de variables : {str(e)}")
                else:
                    st.info("Veuillez sélectionner une variable Y")
            else:
                st.info("Veuillez sélectionner une variable X")
                
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'analyse bivariée : {str(e)}")

# Exécution de l'application
if __name__ == "__main__":
    main()
