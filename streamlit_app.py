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
import textwrap
import seaborn as sns
from scipy import stats
import math
from kaleido.scopes.plotly import PlotlyScope

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
            background-color: #000091;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            transition: background-color 0.3s;
        }

        .stButton > button:hover {
            background-color: #00006f;
        }

        /* Style pour les tableaux de données */
        .dataframe {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        .dataframe th {
            font-family: "Marianne", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-weight: 600 !important;
            background-color: #f0f2f6;
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
        
        /* Badge République Française */
        .fr-badge {
            display: flex;
            align-items: center;
            background-color: #000091;
            color: white;
            padding: 0.3rem 0.7rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.9rem;
            width: fit-content;
            margin-bottom: 1rem;
        }
        
        .fr-badge:before {
            content: "";
            display: inline-block;
            width: 1rem;
            height: 1rem;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path fill="%23fff" d="M0,0H10.7V32H0ZM10.7,0H21.3V32H10.7ZM21.3,0H32V32H21.3Z"/></svg>');
            background-repeat: no-repeat;
            margin-right: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Ajout du badge République Française
st.markdown('<div class="fr-badge">RÉPUBLIQUE<br>FRANÇAISE</div>', unsafe_allow_html=True)

# Configuration de base
sns.set_theme()
sns.set_style("whitegrid")

# Palettes de couleurs prédéfinies
COLOR_PALETTES = {
    "Bleu": ['#000091', '#000080', '#00006f', '#00005e', '#00004d', '#00003c'],  # Bleu Marianne
    "Rouge": ['#E1000F', '#C9000E', '#B1000C', '#99000B', '#810009', '#690007'],  # Rouge Marianne
    "Vert": ['#169B62', '#148957', '#12774C', '#106541', '#0E5336', '#0C412B'],  # Vert Marianne
    "Gris": ['#53657D', '#4A5B70', '#415163', '#384756', '#2F3D49', '#26333C'],  # Gris Marianne
    "Jaune": ['#FFC800', '#E6B400', '#CCA000', '#B38C00', '#997800', '#806400'],  # Jaune DSE
    "Orange": ['#FF9940', '#E68A39', '#CC7B33', '#B36C2D', '#995D26', '#804E20'],  # Orange DSE
    "Rose": ['#FF8D7E', '#E67F71', '#CC7164', '#B36357', '#99554A', '#80473D'],    # Rose DSE
    "Violet": ['#7D4E9E', '#70468E', '#633E7E', '#56366E', '#492E5E', '#3C264E'],  # Violet DSE
    "Cyan": ['#2AA3FF', '#2693E6', '#2283CC', '#1E73B3', '#196399', '#155380'],    # Cyan DSE
    "Vert menthe": ['#00C98E', '#00B580', '#00A172', '#008D64', '#007956', '#006548']  # Vert menthe DSE
}

# Configuration Plotly pour l'export haute qualité
config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'graph_export',
        'height': None,
        'width': 1200,
        'scale': 3
    },
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 
        'autoScale2d', 'resetScale2d', 'toggleSpikelines'
    ]
}

# Configuration Grist
API_KEY = st.secrets["grist_key"]
DOC_ID = st.secrets["grist_doc_id"]
BASE_URL = "https://grist.numerique.gouv.fr/api/docs"

# Fonctions améliorées pour l'API Grist avec mise en cache
@st.cache_data(ttl=3600)  # Cache pour 1 heure
def grist_api_request(endpoint, method="GET", data=None):
    """Fonction utilitaire pour les requêtes API Grist avec mise en cache"""
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
        elif method == "PUT":
            response = requests.put(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        response.raise_for_status()
        return response.json() if response.content else None
    except Exception as e:
        st.error(f"Erreur API Grist : {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache pour 1 heure
def get_grist_tables():
    """Récupère la liste des tables disponibles dans Grist avec mise en cache."""
    try:
        result = grist_api_request("tables")
        if result and 'tables' in result:
            tables_dict = {}
            for table in result['tables']:
                display_name = table['id'].replace('_', ' ')
                tables_dict[display_name] = table['id']
            
            return tables_dict
        return {}
    except Exception as e:
        st.error(f"Erreur lors de la récupération des tables : {str(e)}")
        return {}

@st.cache_data(ttl=3600)  # Cache pour 1 heure
def get_grist_data(table_id):
    """Récupère les données d'une table Grist avec mise en cache."""
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
                # Renommer les colonnes avec leurs labels lisibles
                df = df.rename(columns=column_mapping)
                return df
            
        return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {str(e)}")
        return None

# Fonctions améliorées de visualisation pour l'analyse univariée qualitative

def sanitize_column(df, col):
    """
    Vérifie et corrige une colonne avant de l'utiliser.
    Retourne une version nettoyée et utilisable de la colonne.
    """
    if col not in df.columns:
        st.warning(f"⚠️ La colonne '{col}' est absente de la table sélectionnée.")
        return None  

    col_data = df[col]

    # ✅ Vérification que col_data est bien une série Pandas
    if not isinstance(col_data, pd.Series):
        st.error(f"🚨 Erreur : La colonne '{col}' n'est pas une série Pandas. Valeur retournée : {type(col_data)}")
        return None

    col_data = col_data.fillna("Valeur manquante")

    if col_data.apply(lambda x: isinstance(x, list)).any():
        col_data = col_data.explode()  

    if col_data.apply(lambda x: isinstance(x, dict)).any():
        col_data = col_data.apply(json.dumps)  
    
    # ✅ Vérification que la colonne a bien un type avant d'accéder à `.dtype`
    if hasattr(col_data, "dtype"):
        if col_data.dtype == 'O':  
            col_data = col_data.astype(str)
    else:
        st.error(f"🚨 Erreur : Impossible de détecter le type de la colonne '{col}'.")
        return None

    return col_data  

def plot_qualitative_bar(data, title, x_axis, y_axis, color_palette, show_values, source=None, note=None, value_type="Effectif"):
    """
    Génère un Bar Plot vertical avec les modalités affichées sur plusieurs lignes et sans chevauchement.
    La largeur des barres reste constante quelle que soit le nombre de modalités.
    
    Args:
        data (DataFrame): DataFrame contenant les modalités et leurs fréquences.
        title (str): Titre du graphique.
        x_axis (str): Nom de l'axe X.
        y_axis (str): Nom de l'axe Y.
        color_palette (list): Liste de couleurs pour le graphique.
        show_values (bool): Afficher les valeurs sur les barres.
        source (str, optional): Source des données.
        note (str, optional): Note de lecture.
        value_type (str): "Effectif" ou "Taux (%)" pour adapter l'affichage.

    Returns:
        plotly Figure: Graphique Plotly prêt à être affiché dans Streamlit.
    """
    # ✅ Préparation et adaptation des données
    data = data.copy()
    
    # ✅ Vérifier et renommer la première colonne si nécessaire
    if data.columns[0] != 'Modalités':
        data = data.rename(columns={data.columns[0]: 'Modalités'})
        
    # ✅ Vérifier si l'utilisateur veut afficher les effectifs ou les taux
    y_column = "Taux (%)" if value_type == "Taux (%)" else "Effectif"

    # ✅ Appliquer un retour à la ligne automatique sur les modalités longues (coupure tous les 23 caractères)
    wrapped_labels = [
        "<br>".join(textwrap.wrap(label, width=23))  
        for label in data["Modalités"]
    ]

    # ✅ Calcul de la hauteur de la marge basse en fonction de la longueur des modalités
    max_label_lines = max([label.count("<br>") + 1 for label in wrapped_labels])  # Nombre max de lignes de texte
    bottom_margin = 100 + (max_label_lines * 12)  # Ajustement dynamique de la marge

    # ✅ Création du graphique en barres verticales avec la bonne colonne
    fig = px.bar(
        data, 
        x=wrapped_labels,  # ✅ Modalités modifiées avec sauts de ligne
        y=y_column,  # ✅ Adaptation dynamique selon "Effectif" ou "Taux (%)"
        text=y_column if show_values else None,  # ✅ Affichage des valeurs en fonction du type choisi
        title=title,
        color_discrete_sequence=color_palette
    )

    # ✅ Mise en forme des valeurs affichées sur les barres
    fig.update_traces(
        texttemplate='%{text:.1f}%' if value_type == "Taux (%)" else '%{text:,}',  # ✅ Format correct des pourcentages
        textposition="outside",
        marker_line_width=1.2,
        width=0.2  # ✅ MODIFICATION PRINCIPALE: Définir une largeur fixe pour toutes les barres
    )

    # ✅ Ajustement de l'affichage pour éviter les chevauchements
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        title_font=dict(size=16, family="Arial", color="black"),
        margin=dict(l=50, r=50, t=60, b=bottom_margin),  # ✅ Ajustement dynamique de la marge basse
        bargap=0.3,  # ✅ Augmenter l'espace entre les barres pour un meilleur rendu visuel
        plot_bgcolor='white'  # ✅ Fond blanc pour plus de clarté
    )

    # ✅ Ajustement de la position de la source et de la note en fonction de la marge basse
    annotation_y = -0.4 - (0.02 * max_label_lines)  # ✅ Remonter la source/notes en fonction du texte

    annotations = []
    if source:
        annotations.append(dict(
            xref="paper", yref="paper", 
            x=0, y=annotation_y, 
            text=f" Source : {source}", 
            showarrow=False, font=dict(size=12, color="gray")
        ))
    if note:
        annotations.append(dict(
            xref="paper", yref="paper", 
            x=0, y=annotation_y - 0.05, 
            text=f" Note : {note}", 
            showarrow=False, font=dict(size=12, color="gray")
        ))

    fig.update_layout(annotations=annotations)

    return fig

def plot_dotplot(data, title, x_label, y_label, color_palette, show_values=True, source="", note="", width=850, value_type="Effectif"):
    """
    Crée un graphique dot plot avec une échelle de valeurs pour l'analyse univariée qualitative.

    Args:
        data (DataFrame): DataFrame avec les colonnes 'Modalités' et 'Effectif'
        title (str): Titre du graphique
        x_label (str): Étiquette de l'axe X (valeurs)
        y_label (str): Étiquette de l'axe Y (modalités)
        color_palette (list): Liste de couleurs
        show_values (bool): Afficher les valeurs sur le graphique
        source (str): Source des données
        note (str): Note explicative
        width (int): Largeur du graphique
        value_type (str): "Effectif" ou "Taux (%)" pour adapter l'affichage

    Returns:
        go.Figure: Figure Plotly
    """
    import textwrap

    # ✅ Préparation et adaptation des données
    data = data.copy()
    
    # ✅ Vérifier et renommer la première colonne si nécessaire
    if data.columns[0] != 'Modalités':
        data = data.rename(columns={data.columns[0]: 'Modalités'})

    # ✅ Sélection de la bonne colonne (Effectif ou Taux)
    y_column = "Taux (%)" if value_type == "Taux (%)" else "Effectif"

    # ✅ Vérifier que la colonne "Taux (%)" existe si nécessaire
    if value_type == "Taux (%)" and "Taux (%)" not in data.columns:
        total = data["Effectif"].sum()
        data["Taux (%)"] = (data["Effectif"] / total * 100).round(1)

    # ✅ Trier les données pour un affichage clair
    data = data.sort_values(y_column, ascending=True).reset_index(drop=True)

    # ✅ Appliquer un retour à la ligne automatique sur les modalités longues
    wrapped_labels = [
        "<br>".join(textwrap.wrap(str(label), width=20)) if isinstance(label, str) and len(label) > 20 else str(label)
        for label in data['Modalités']
    ]

    # ✅ Déterminer le maximum et créer une échelle
    max_val = data[y_column].max()
    step_size = max(1, round(max_val / 5))
    scale_values = list(range(0, int(max_val) + step_size, step_size))

    # ✅ Ajustement automatique de la hauteur
    num_bars = len(data)
    base_height = 100 + (num_bars * 60)

    # ✅ Création du graphique
    fig = go.Figure()

    # ✅ Ajouter l'échelle de valeurs (lignes verticales)
    for val in scale_values:
        fig.add_shape(
            type="line",
            x0=val,
            y0=-0.5,
            x1=val,
            y1=len(data) - 0.5,
            line=dict(color="lightgray", width=1, dash="dot"),
        )

    # ✅ Ajouter les lignes horizontales pour chaque modalité
    for i in range(len(data)):
        fig.add_shape(
            type="line",
            x0=0,
            y0=i,
            x1=data.iloc[i][y_column],
            y1=i,
            line=dict(color="lightgray", width=1),
        )

    # ✅ Ajouter les points
    fig.add_trace(go.Scatter(
        x=data[y_column],
        y=list(range(len(data))),
        mode='markers',
        marker=dict(
            size=15,
            color=color_palette[0],
            line=dict(width=1, color='white')
        ),
        name='Valeurs',
        hovertemplate='<b>%{text}</b><br>Valeur: %{x}<extra></extra>',
        text=data['Modalités']
    ))

    # ✅ Ajouter les valeurs avec un **décalage dynamique**
    if show_values:
        fig.add_trace(go.Scatter(
            x=data[y_column] + max_val * 0.05,  # ✅ Décalage pour éviter le chevauchement
            y=list(range(len(data))),
            mode='text',
            text=data[y_column].apply(lambda x: f"{x:.1f}%" if value_type == "Taux (%)" else f"{int(x)}"),
            textposition='middle right',
            textfont=dict(size=12, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))

    # ✅ Ajout dynamique de la Source et de la Note
    annotation_y = -0.2 - (0.02 * num_bars)  # ✅ Ajustement dynamique basé sur le nombre de modalités

    annotations = []
    if source:
        annotations.append(dict(
            text=f" Source : {source}",
            x=0,
            y=annotation_y,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(family="Marianne, sans-serif", size=12, color="gray"),
            align='left',
            xanchor='left'
        ))

    if note:
        annotations.append(dict(
            text=f" Note : {note}",
            x=0,
            y=annotation_y - 0.08,  # ✅ Espacement correct entre source et note
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(family="Marianne, sans-serif", size=12, color="gray"),
            align='left',
            xanchor='left'
        ))

    # ✅ Configuration du layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Marianne, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=y_label,
            showgrid=False,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray',
            range=[-max_val * 0.05, max_val * 1.2]  # ✅ Ajustement pour le décalage
        ),
        yaxis=dict(
            title=x_label,
            tickmode='array',
            tickvals=list(range(len(data))),
            ticktext=wrapped_labels,
            zeroline=False,
            showgrid=False,
            autorange="reversed"  # ✅ La modalité la plus grande en haut
        ),
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(l=250, r=100, t=100, b=150),  # ✅ Ajustement dynamique des marges
        width=width,
        height=base_height,
        annotations=annotations
    )

    return fig

def plot_modern_horizontal_bars(data, title, x_label, value_type="Effectif", color_palette=None, source="", note="", is_export=False):
    """
    Crée un graphique à barres horizontales optimisé pour l'affichage et l'export.

    Args:
        data (DataFrame): Données avec 'Modalités' et 'Effectif' ou 'Taux (%)'.
        title (str): Titre du graphique.
        x_label (str): Nom de l'axe X.
        value_type (str): "Effectif" ou "Taux (%)".
        color_palette (list): Liste de couleurs.
        source (str): Source des données.
        note (str): Note explicative.
        is_export (bool): Mode export (True) ou affichage Streamlit (False).
    
    Returns:
        go.Figure: Graphique Plotly.
    """
    import textwrap

    # ✅ Copie et préparation des données
    data = data.copy()
    if data.columns[0] != 'Modalités':
        data = data.rename(columns={data.columns[0]: 'Modalités'})

    # ✅ Tri des données
    data = data.sort_values('Effectif', ascending=True).reset_index(drop=True)

    # ✅ Calcul des taux si nécessaire
    if value_type == "Taux (%)" and "Taux (%)" not in data.columns:
        total = data["Effectif"].sum()
        data["Taux (%)"] = (data["Effectif"] / total * 100).round(1)

    # ✅ Colonne à afficher
    y_column = "Taux (%)" if value_type == "Taux (%)" else "Effectif"

    # ✅ Appliquer un retour à la ligne automatique sur les modalités longues
    wrapped_labels = [
        "<br>".join(textwrap.wrap(str(label), width=20)) if isinstance(label, str) and len(label) > 20 else str(label)
        for label in data['Modalités']
    ]

    # ✅ Nombre de modalités et ajustement de la hauteur
    num_bars = len(data)
    base_height = 100 + (num_bars * 60)

    # ✅ Ajustement dynamique des marges
    bottom_margin = 120 + (30 * (bool(source) + bool(note)))  # ✅ Ajustement en fonction de source/note
    left_margin = 300 if max(len(label) for label in wrapped_labels) > 20 else 250  # ✅ Ajustement selon la longueur des labels

    # ✅ Création du graphique
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=wrapped_labels,
        x=data[y_column],
        orientation='h',
        text=data[y_column].apply(lambda v: f"{v:.1f}%" if value_type == "Taux (%)" else f"{int(v) if float(v).is_integer() else v:.1f}"),
        textposition='outside',
        textfont=dict(family="Marianne, sans-serif", size=14),
        marker=dict(color=color_palette[0] if color_palette else '#000091'),
        width=0.7,
        hovertemplate="%{y}<br>%{x}<extra></extra>"
    ))

    # ✅ Configuration des axes
    max_value = data[y_column].max()
    padding = max_value * 0.15  # Ajout d’un padding pour éviter le chevauchement
    
    fig.update_xaxes(
        title=value_type,
        range=[0, max_value + padding],
        tickfont=dict(family="Marianne, sans-serif", size=14),
        gridcolor='lightgray'
    )

    fig.update_yaxes(
        title=x_label,
        autorange="reversed",
        tickfont=dict(family="Marianne, sans-serif", size=14)
    )

    # ✅ Ajout dynamique de la Source et de la Note
    annotation_y = -0.4 - (0.02 * num_bars)  # ✅ Ajustement dynamique basé sur le nombre de modalités

    annotations = []
    if source:
        annotations.append(dict(
            text=f" Source : {source}",
            x=0,
            y=annotation_y,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(family="Marianne, sans-serif", size=12, color="gray"),
            align='left',
            xanchor='left'
        ))

    if note:
        annotations.append(dict(
            text=f" Note : {note}",
            x=0,
            y=annotation_y - 0.10,  # ✅ Espacement correct entre source et note
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(family="Marianne, sans-serif", size=12, color="gray"),
            align='left',
            xanchor='left'
        ))

    fig.update_layout(
        annotations=annotations,
        autosize=False,
        width=900,
        height=base_height,
        margin=dict(
            l=left_margin,  # ✅ Ajustement dynamique selon la taille des modalités
            r=100,
            t=100,
            b=bottom_margin if not is_export else 180  # ✅ Ajustement dynamique de la marge basse
        ),
        title=dict(
            text=title,
            font=dict(family="Marianne, sans-serif", size=18),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # ✅ Marquer pour l'export
    fig._is_horizontal_bar = True

    return fig

def plot_qualitative_lollipop(data, title, x_label, y_label, color_palette, show_values=True, source="", note="", value_type="Effectif"):
    """
    Crée un graphique Lollipop optimisé pour l'affichage des modalités et des valeurs.
    """
    import textwrap
    
    # ✅ Préparation et adaptation des données
    data = data.copy()
    
    # ✅ Vérifier et renommer la première colonne si nécessaire
    if data.columns[0] != 'Modalités':
        data = data.rename(columns={data.columns[0]: 'Modalités'})

    # ✅ Vérifier si on affiche les effectifs ou les taux
    y_column = "Taux (%)" if value_type == "Taux (%)" else "Effectif"

    # ✅ Calcul des taux si nécessaire
    if value_type == "Taux (%)" and "Taux (%)" not in data.columns:
        total = data["Effectif"].sum()
        data["Taux (%)"] = (data["Effectif"] / total * 100).round(1)

    # ✅ Trier les données par ordre croissant
    data = data.sort_values(y_column, ascending=True).reset_index(drop=True)

    # ✅ Appliquer un retour à la ligne automatique sur les modalités longues
    wrapped_labels = [
        "<br>".join(textwrap.wrap(label, width=23))  
        for label in data["Modalités"]
    ]

    # ✅ Calcul de la hauteur de la marge basse pour éviter le chevauchement
    max_label_lines = max([label.count("<br>") + 1 for label in wrapped_labels])  
    bottom_margin = 100 + (max_label_lines * 12)  

    # ✅ Création du graphique Lollipop
    fig = go.Figure()

    # ✅ Ajouter les lignes verticales reliant les points à l'axe X
    for x, y in zip(wrapped_labels, data[y_column]):
        fig.add_trace(go.Scatter(
            x=[x, x],
            y=[0, y],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='none'
        ))

    # ✅ Ajouter les points (lollipops)
    fig.add_trace(go.Scatter(
        x=wrapped_labels,
        y=data[y_column],
        mode='markers+text' if show_values else 'markers',
        marker=dict(size=14, color=color_palette[0], line=dict(width=1, color='white')),
        text=data[y_column].apply(lambda x: f"{x:.1f}%" if value_type == "Taux (%)" else f"{int(x)}") if show_values else None,
        textposition='top center',
        showlegend=False,
        hovertemplate="%{x}<br>Valeur: %{y:.1f}<extra></extra>"
    ))

    # ✅ Ajustement dynamique de l’axe Y pour éviter les chevauchements
    max_value = data[y_column].max()
    padding = max_value * 0.15  # Ajout d’un espace pour éviter les chevauchements
    
    # ✅ Ajustement dynamique des annotations (Source et Note)
    annotation_y = -0.5 - (0.02 * max_label_lines)  

    annotations = []
    if source:
        annotations.append(dict(
            xref="paper", yref="paper", 
            x=0, y=annotation_y, 
            text=f" Source : {source}", 
            showarrow=False, font=dict(size=12, color="gray")
        ))
    if note:
        annotations.append(dict(
            xref="paper", yref="paper", 
            x=0, y=annotation_y - 0.05, 
            text=f" Note : {note}", 
            showarrow=False, font=dict(size=12, color="gray")
        ))

    # ✅ Appliquer les annotations au graphique
    fig.update_layout(annotations=annotations)

    # ✅ Configuration de l'affichage
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Marianne, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title=x_label,
            tickvals=list(range(len(data))),
            ticktext=wrapped_labels,  # ✅ Affichage clair des modalités
            tickangle=0,  # ✅ Texte bien horizontal
            showgrid=False
        ),
        yaxis=dict(
            title=y_label,
            range=[0, max_value + padding],  # ✅ Ajout d’un espace pour éviter le chevauchement
            showgrid=True,
            gridcolor='#e0e0e0'
        ),
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=bottom_margin),  # ✅ Ajustement dynamique de la marge basse
    )

    return fig

def plot_qualitative_treemap(data, title, color_palette, source="", note=""):
    """
    Crée un treemap amélioré pour l'analyse univariée qualitative.
    
    Args:
        data (DataFrame): DataFrame avec les colonnes 'Modalités' et 'Effectif'
        title (str): Titre du graphique
        color_palette (list): Liste de couleurs
        source (str): Source des données
        note (str): Note explicative
        
    Returns:
        go.Figure: Figure Plotly
    """
    # ✅ Renommer les colonnes si nécessaire
    data = data.copy()
    if data.columns[0] != 'Modalités':
        data = data.rename(columns={data.columns[0]: 'Modalités'})
    
    # ✅ Calculer les pourcentages
    data['Pourcentage'] = (data['Effectif'] / data['Effectif'].sum() * 100).round(1)  # ✅ Le calcul était bon

    # ✅ Créer le treemap
    fig = go.Figure(go.Treemap(
        labels=data['Modalités'],
        parents=[""] * len(data),
        values=data['Effectif'],
        text=data['Pourcentage'].astype(str) + "%",  # ✅ Correction : Affichage correct des pourcentages
        texttemplate="<b>%{label}</b><br>Effectif: %{value}<br>Pourcentage: %{text}",  # ✅ Utilisation correcte
        hovertemplate="<b>%{label}</b><br>Effectif: %{value}<br>Pourcentage: %{text}<extra></extra>",
        marker=dict(
            colors=color_palette[:len(data)] if len(color_palette) >= len(data) else color_palette,
            line=dict(width=1, color='white')
        ),
        tiling=dict(
            packing='squarify',  # ✅ Utilisation de Squarify pour une meilleure disposition
            pad=3
        ),
        pathbar=dict(visible=False)  # ✅ Masquer la barre de navigation
    ))

    # ✅ Annotations pour la source et la note
    annotations = []
    
    if source or note:
        if source:
            annotations.append(dict(
                text=f" Source : {source}",
                x=0,
                y=-0.08,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
                align='left',
                xanchor='left'
            ))
        
        if note:
            note_y_pos = -0.08 - 0.04 if source else -0.08
            annotations.append(dict(
                text=f" Note : {note}",
                x=0,
                y=note_y_pos,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
                align='left',
                xanchor='left'
            ))

    # ✅ Configuration du layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Marianne, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        width=900,
        height=650,
        margin=dict(l=50, r=50, t=100, b=100),
        annotations=annotations
    )

    return fig

def plot_radar(data, title, color_palette, source="", note="", value_type="Effectif"):
    """
    Crée un graphique radar pour comparer des modalités en Effectif ou Taux (%).
    
    Args:
        data (DataFrame): DataFrame avec les colonnes 'Modalités' et 'Effectif'
        title (str): Titre du graphique
        color_palette (list): Liste de couleurs
        source (str): Source des données
        note (str): Note explicative
        value_type (str): "Effectif" ou "Taux (%)" pour l'affichage
        
    Returns:
        go.Figure: Figure Plotly
    """
    # ✅ Renommer les colonnes si nécessaire
    data = data.copy()
    if data.columns[0] != 'Modalités':
        data = data.rename(columns={data.columns[0]: 'Modalités'})
    
    # ✅ Sélection de la colonne de valeurs (Effectif ou Taux)
    if value_type == "Taux (%)":
        if "Taux (%)" not in data.columns:
            data["Taux (%)"] = (data["Effectif"] / data["Effectif"].sum() * 100).round(1)  # ✅ Calcul du taux
        y_column = "Taux (%)"
    else:
        y_column = "Effectif"

    # ✅ Préparation des données pour le radar
    categories = data['Modalités'].tolist()
    values = data[y_column].tolist()
    
    # ✅ Fermer la boucle en répétant le premier point
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    # ✅ Conversion de la couleur hex en rgba pour la transparence
    def hex_to_rgba(hex_color, alpha=0.5):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    fill_color = hex_to_rgba(color_palette[0])
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor=fill_color,
        line=dict(color=color_palette[0], width=2),
        name=y_column  # ✅ Adaptation du nom selon Effectif ou Taux (%)
    ))
    
    # ✅ Annotations pour la source et la note
    annotations = []
    
    if source or note:
        if source:
            annotations.append(dict(
                text=f" Source : {source}",
                x=0,
                y=-0.15,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
                align='left',
                xanchor='left'
            ))
        
        if note:
            note_y_pos = -0.15 - 0.03 if source else -0.15
            annotations.append(dict(
                text=f" Note : {note}",
                x=0,
                y=note_y_pos,
                xref='paper',
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
                align='left',
                xanchor='left'
            ))
    
    # ✅ Configuration du layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Marianne, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                showticklabels=True,
                range=[0, max(values) * 1.1]
            ),
            angularaxis=dict(
                showticklabels=True,
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        width=800,
        height=650,
        margin=dict(l=80, r=80, t=100, b=100),
        annotations=annotations
    )
    
    return fig

def plot_bullet_chart(data, title, x_label, targets=None, color_palette=None, source="", note=""):
    """
    Crée un graphique de type "bullet chart" pour comparer des valeurs à des cibles/objectifs.
    
    Args:
        data (DataFrame): DataFrame avec les colonnes 'Modalités' et 'Effectif'
        title (str): Titre du graphique
        x_label (str): Étiquette de l'axe X
        targets (dict): Dictionnaire de valeurs cibles pour chaque modalité
        color_palette (list): Liste de couleurs
        source (str): Source des données
        note (str): Note explicative
        
    Returns:
        go.Figure: Figure Plotly
    """
    # Renommer les colonnes si nécessaire
    data = data.copy()
    if data.columns[0] != 'Modalités':
        data = data.rename(columns={data.columns[0]: 'Modalités'})
    
    # Trier les données par effectif décroissant
    data = data.sort_values('Effectif', ascending=False).reset_index(drop=True)
    
    # Si targets n'est pas fourni, créer des cibles fictives
    if targets is None:
        targets = {row['Modalités']: row['Effectif'] * 1.2 for _, row in data.iterrows()}
    
    # Couleurs par défaut
    if color_palette is None:
        color_palette = ['#000091', '#E1000F']  # Bleu Marianne, Rouge Marianne
    
    fig = go.Figure()
    
    # Ajouter les barres principales
    for i, (_, row) in enumerate(data.iterrows()):
        fig.add_trace(go.Bar(
            y=[row['Modalités']],
            x=[row['Effectif']],
            orientation='h',
            marker=dict(
                color=color_palette[0],
                line=dict(width=0)
            ),
            text=f"{row['Effectif']:.1f}" if not float(row['Effectif']).is_integer() else f"{int(row['Effectif'])}",
            textposition='inside',
            textfont=dict(
                family="Marianne, sans-serif",
                size=12,
                color="white"
            ),
            hovertemplate=f"<b>{row['Modalités']}</b><br>Valeur: {row['Effectif']}<extra></extra>",
            name='Valeur actuelle',
            showlegend=i == 0
        ))
    
    # Ajouter les objectifs/cibles
    for i, (_, row) in enumerate(data.iterrows()):
        modalite = row['Modalités']
        if modalite in targets:
            target = targets[modalite]
            fig.add_trace(go.Scatter(
                y=[modalite],
                x=[target],
                mode='markers',
                marker=dict(
                    symbol='line-ns',
                    size=12,
                    color=color_palette[1],
                    line=dict(width=2)
                ),
                text=f"Objectif: {target:.1f}" if not float(target).is_integer() else f"Objectif: {int(target)}",
                hovertemplate=f"<b>{modalite}</b><br>Objectif: {target}<extra></extra>",
                name='Objectif',
                showlegend=i == 0
            ))
    
    # Configuration des axes
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family="Marianne, sans-serif",
                size=20,
                color="#333333"
            ),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(500, len(data) * 50 + 200),
        margin=dict(l=200, r=80, t=100, b=100),
        yaxis=dict(
            title=x_label,
            titlefont=dict(
                family="Marianne, sans-serif",
                size=14
            ),
            tickfont=dict(
                family="Marianne, sans-serif",
                size=12
            ),
            autorange="reversed"
        ),
        xaxis=dict(
            title="Valeur",
            titlefont=dict(
                family="Marianne, sans-serif",
                size=14
            ),
            tickfont=dict(
                family="Marianne, sans-serif",
                size=12
            ),
            showgrid=True,
            gridcolor='#f0f0f0',
            zeroline=True,
            zerolinecolor='#e0e0e0',
            zerolinewidth=1
        ),
        bargap=0.3,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Ajouter des annotations pour la source et la note
    annotations = []
    
    if source:
        annotations.append(dict(
            text=f"Source : {source}",
            x=0,
            y=-0.15,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(
                family="Marianne, sans-serif",
                size=11,
                color="gray"
            ),
            align='left',
            xanchor='left'
        ))
    
    if note:
        note_y_pos = -0.20 if source else -0.15
        annotations.append(dict(
            text=f"Note : {note}",
            x=0,
            y=note_y_pos,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(
                family="Marianne, sans-serif",
                size=11,
                color="gray"
            ),
            align='left',
            xanchor='left'
        ))
    
    fig.update_layout(annotations=annotations)
    
    return fig

def create_qualitative_dashboard(data_series, var_name):
    """
    Crée un dashboard récapitulatif pour une variable qualitative.
    
    Args:
        data_series (Series): Série pandas contenant les données
        var_name (str): Nom de la variable
    """
    # Nettoyage des données
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'NA', 'nr', 'NR', 'Non réponse', 'Non-réponse']
    clean_data = data_series.replace(missing_values, np.nan).dropna()
    
    # Calculs principaux
    total_obs = len(data_series)
    valid_obs = len(clean_data)
    missing_obs = total_obs - valid_obs
    response_rate = (valid_obs / total_obs * 100) if total_obs > 0 else 0
    nb_modalities = len(clean_data.unique())
    
    # Création des colonnes
    col1, col2, col3 = st.columns(3)
    
    # Colonne 1: Résumé général
    with col1:
        st.markdown(f"##### Résumé de la variable")
        st.metric("Observations totales", total_obs)
        st.metric("Observations valides", valid_obs)
        st.metric("Observations manquantes", missing_obs)
        st.metric("Taux de réponse", f"{response_rate:.1f}%")
    
    # Colonne 2: Top 3 et bottom 3 modalités
    with col2:
        st.markdown(f"##### Modalités principales")
        value_counts = clean_data.value_counts()
        percent_counts = (value_counts / valid_obs * 100).round(1)
        
        st.markdown("**Top 3 modalités:**")
        for i, (idx, count) in enumerate(value_counts.head(3).items()):
            st.markdown(f"{i+1}. **{idx}**: {count} ({percent_counts.loc[idx]}%)")
        
        if len(value_counts) > 6:
            st.markdown("**Modalités moins fréquentes:**")
            for i, (idx, count) in enumerate(value_counts.tail(3).items()):
                st.markdown(f"{i+1}. **{idx}**: {count} ({percent_counts.loc[idx]}%)")
    
    # Colonne 3: Informations supplémentaires
    with col3:
        st.markdown(f"##### Caractéristiques")
        st.metric("Nombre de modalités", nb_modalities)
        
        # Modalité la plus fréquente
        mode_value = clean_data.mode()[0]
        mode_count = value_counts.loc[mode_value]
        mode_percent = percent_counts.loc[mode_value]
        st.markdown(f"**Modalité dominante:** {mode_value}")
        st.markdown(f"Représente {mode_count} observations ({mode_percent}%)")
        
        # Équilibre des modalités (indice de Gini simplifié)
        if nb_modalities > 1:
            proportions = percent_counts / 100
            gini = 1 - sum(proportions**2)
            normalized_gini = gini * (nb_modalities / (nb_modalities - 1)) if nb_modalities > 1 else 0
            
            balance_text = "Très déséquilibrée"
            if normalized_gini > 0.8:
                balance_text = "Très équilibrée"
            elif normalized_gini > 0.6:
                balance_text = "Équilibrée"
            elif normalized_gini > 0.4:
                balance_text = "Moyennement équilibrée"
            elif normalized_gini > 0.2:
                balance_text = "Déséquilibrée"
            
            st.metric("Distribution", balance_text)
            st.progress(normalized_gini)

def create_quantitative_dashboard(data_series, var_name):
    """
    Crée un dashboard récapitulatif pour une variable quantitative.
    
    Args:
        data_series (Series): Série pandas contenant les données
        var_name (str): Nom de la variable
    """
    # Nettoyage des données
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'NA', 'nr', 'NR', 'Non réponse', 'Non-réponse']
    clean_data = data_series.replace(missing_values, np.nan).dropna()
    
    # Calculs principaux
    total_obs = len(data_series)
    valid_obs = len(clean_data)
    missing_obs = total_obs - valid_obs
    response_rate = (valid_obs / total_obs * 100) if total_obs > 0 else 0
    
    # Vérifier si les données sont vides
    if len(clean_data) == 0:
        st.warning(f"Aucune donnée valide pour la variable {var_name}")
        return
    
    # Vérifier si c'est une variable entière ou décimale
    try:
        is_integer = all(float(x).is_integer() for x in clean_data)
    except (ValueError, TypeError):
        is_integer = False
    
    # Création des colonnes
    col1, col2, col3 = st.columns(3)
    
    # Colonne 1: Résumé général
    with col1:
        st.markdown(f"##### Résumé de la variable")
        st.metric("Observations totales", total_obs)
        st.metric("Observations valides", valid_obs)
        st.metric("Observations manquantes", missing_obs)
        st.metric("Taux de réponse", f"{response_rate:.1f}%")
    
    # Colonne 2: Statistiques descriptives
    with col2:
        st.markdown(f"##### Statistiques descriptives")
        
        try:
            mean_val = clean_data.mean()
            median_val = clean_data.median()
            std_val = clean_data.std()
            min_val = clean_data.min()
            max_val = clean_data.max()
            
            # Utiliser une gestion plus robuste pour l'affichage
            if pd.isna(mean_val):
                st.metric("Moyenne", "N/A")
            elif is_integer:
                st.metric("Moyenne", f"{int(mean_val)}")
            else:
                st.metric("Moyenne", f"{mean_val:.2f}")
                
            if pd.isna(median_val):
                st.metric("Médiane", "N/A")
            elif is_integer:
                st.metric("Médiane", f"{int(median_val)}")
            else:
                st.metric("Médiane", f"{median_val:.2f}")
                
            if pd.isna(std_val):
                st.metric("Écart-type", "N/A")
            elif is_integer:
                st.metric("Écart-type", f"{int(std_val)}")
            else:
                st.metric("Écart-type", f"{std_val:.2f}")
                
            if pd.isna(min_val) or pd.isna(max_val):
                st.metric("Étendue", "N/A")
            elif is_integer:
                st.metric("Étendue", f"{int(min_val)} - {int(max_val)}")
            else:
                st.metric("Étendue", f"{min_val:.2f} - {max_val:.2f}")
        except Exception as e:
            st.error(f"Erreur lors du calcul des statistiques: {str(e)}")
            st.metric("Moyenne", "Erreur")
            st.metric("Médiane", "Erreur")
            st.metric("Écart-type", "Erreur")
            st.metric("Étendue", "Erreur")
    
    # Colonne 3: Quartiles et distribution
    with col3:
        st.markdown(f"##### Distribution")
        
        q1 = clean_data.quantile(0.25)
        q3 = clean_data.quantile(0.75)
        iqr = q3 - q1
        
        if is_integer:
            st.metric("Q1 (25%)", f"{int(q1)}")
            st.metric("Q3 (75%)", f"{int(q3)}")
            st.metric("Écart interquartile", f"{int(iqr)}")
        else:
            st.metric("Q1 (25%)", f"{q1:.2f}")
            st.metric("Q3 (75%)", f"{q3:.2f}")
            st.metric("Écart interquartile", f"{iqr:.2f}")
        
        # Coefficient de variation
        cv = (std_val / mean_val) * 100 if mean_val != 0 else float('inf')
        
        cv_text = "Très homogène"
        if cv > 100:
            cv_text = "Très hétérogène"
        elif cv > 50:
            cv_text = "Hétérogène"
        elif cv > 30:
            cv_text = "Moyennement homogène"
        elif cv > 15:
            cv_text = "Homogène"
        
        st.metric("Dispersion", cv_text)
        
        # Normaliser le CV pour l'affichage de la barre de progression
        normalized_cv = min(1.0, cv / 100)
        st.progress(normalized_cv)

def export_visualization(fig, export_type, var_name, source="", note="", data_to_plot=None, is_plotly=True, graph_type="bar"):
    """
    Fonction d'export simplifiée avec une approche robuste pour les annotations.
    """
    try:
        buf = BytesIO()
        
        if export_type == 'graph' and is_plotly:
            # Créer une copie de la figure pour l'export
            export_fig = go.Figure(fig)
            
            # Définir des dimensions fixes généreuses pour l'export
            export_width = 1600
            export_height = 900
            
            # Marges très généreuses pour tous les types de graphiques
            export_fig.update_layout(
                width=export_width,
                height=export_height,
                margin=dict(
                    l=400,  # Marge gauche généreuse pour les labels
                    r=150,  # Marge droite
                    t=150,  # Marge haute pour le titre
                    b=250   # Marge basse généreuse pour source/note
                ),
                title=dict(
                    text=f"<b>{fig.layout.title.text}</b>",  # ✅ Ajout du gras dans le titre
                    font=dict(size=22, family="Marianne, sans-serif", color="black"),  # ✅ Taille + Police
                    x=0.5,
                    xanchor='center'
                )
            )
            
            # Supprimer toutes les annotations existantes de source et note
            annotations = []
            for ann in export_fig.layout.annotations:
                if "Source" not in str(ann.text) and "Note" not in str(ann.text):
                    annotations.append(ann)
            
            # Ajouter les annotations source et note de manière fixe et robuste
            if source:
                annotations.append(dict(
                    text=f"<b>Source :</b> {source}",
                    x=0,
                    y=-0.25,  # Position basse fixe
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(family="Marianne, sans-serif", size=14, color="black"),
                    align='left',
                    xanchor='left'
                ))
            
            if note:
                annotations.append(dict(
                    text=f"<b>Note :</b> {note}",
                    x=0,
                    y=-0.30,  # Position encore plus basse
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(family="Marianne, sans-serif", size=14, color="black"),
                    align='left',
                    xanchor='left'
                ))
            
            # Mettre à jour les annotations
            export_fig.update_layout(annotations=annotations)
            
            # Pour les graphiques horizontaux, étendre la plage x pour éviter la troncature
            if hasattr(fig, '_is_horizontal_bar') or graph_type in ["horizontal", "Horizontal Bar"]:
                # Calculer la valeur maximale et ajouter un padding
                if data_to_plot is not None:
                    col = "Taux (%)" if "Taux (%)" in data_to_plot.columns else "Effectif"
                    max_val = data_to_plot[col].max()
                    padding = max_val * 0.2  # 20% de padding
                    
                    export_fig.update_xaxes(range=[0, max_val + padding])
            
            # Exporter en haute résolution
            export_fig.write_image(
                buf,
                format="png",
                scale=2.0,
                validate=False
            )
            
        elif export_type == 'table':
            plt.savefig(
                buf, 
                format='png',
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

        buf.seek(0)
        image_data = buf.getvalue()
        
        file_suffix = "tableau" if export_type == 'table' else "graphique"
        file_name = f"{file_suffix}_{var_name.lower().replace(' ', '_')}.png"

        st.download_button(
            label=f"💾 Télécharger le {file_suffix} (HD)",
            data=image_data,
            file_name=file_name,
            mime="image/png",
            key=f"download_{export_type}_{var_name}"
        )
        return True

    except Exception as e:
        st.error(f"Erreur lors de l'export : {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# Fonctions de manipulation des données
def merge_multiple_tables(dataframes, merge_configs):
    """
    Fusionne plusieurs DataFrames selon les configurations spécifiées.
    
    Args:
        dataframes (list): Liste de DataFrames à fusionner
        merge_configs (list): Liste de configurations pour les fusions
        
    Returns:
        DataFrame: DataFrame fusionné
    """
    # Si un seul DataFrame est fourni, le retourner directement
    if len(dataframes) == 1:
        return dataframes[0]
    
    merged_df = dataframes[0]
    for i in range(1, len(dataframes)):
        merge_config = merge_configs[i - 1]
        
        # Vérifier que les colonnes de fusion existent
        if merge_config['left'] not in merged_df.columns:
            st.error(f"La colonne '{merge_config['left']}' n'existe pas dans le tableau de gauche.")
            return None
        
        if merge_config['right'] not in dataframes[i].columns:
            st.error(f"La colonne '{merge_config['right']}' n'existe pas dans le tableau de droite.")
            return None
        
        # Convertir les types si nécessaire pour assurer la compatibilité
        if merged_df[merge_config['left']].dtype != dataframes[i][merge_config['right']].dtype:
            st.info(f"Conversion automatique des types pour la fusion sur '{merge_config['left']}' et '{merge_config['right']}'")
            
            if pd.api.types.is_numeric_dtype(merged_df[merge_config['left']]) and pd.api.types.is_numeric_dtype(dataframes[i][merge_config['right']]):
                merged_df[merge_config['left']] = merged_df[merge_config['left']].astype(float)
                dataframes[i][merge_config['right']] = dataframes[i][merge_config['right']].astype(float)
            else:
                merged_df[merge_config['left']] = merged_df[merge_config['left']].astype(str)
                dataframes[i][merge_config['right']] = dataframes[i][merge_config['right']].astype(str)
        
        # Effectuer la fusion
        merged_df = merged_df.merge(
            dataframes[i], 
            left_on=merge_config['left'], 
            right_on=merge_config['right'], 
            how='outer',
            suffixes=('', f'_{i}')  # Éviter les conflits de noms de colonnes
        )
    
    return merged_df

def is_numeric_column(df, column):
    """
    Vérifie si une colonne est numérique de manière sûre.
    
    Args:
        df (DataFrame): DataFrame contenant la colonne
        column (str): Nom de la colonne à vérifier
        
    Returns:
        bool: True si la colonne est numérique, False sinon
    """
    try:
        if column not in df.columns:
            return False
        
        # Vérifier si la colonne est déjà de type numérique
        if pd.api.types.is_numeric_dtype(df[column]):
            return True
        
        # Essayer de convertir en numérique pour les colonnes avec des valeurs mixtes
        sample = df[column].dropna().head(100)  # Prendre un échantillon pour tester
        if len(sample) == 0:
            return False
        
        # Essayer de convertir avec pd.to_numeric
        try:
            pd.to_numeric(sample)
            return True
        except (ValueError, TypeError):
            return False
            
    except Exception as e:
        st.error(f"Erreur lors de la vérification du type de la colonne {column}: {str(e)}")
        return False

def check_normality(data, var, sample_size=5000):
    """
    Vérifie la normalité d'une variable avec adaptation pour les grands échantillons.
    
    Args:
        data (DataFrame): DataFrame contenant la variable
        var (str): Nom de la variable à tester
        sample_size (int): Taille de l'échantillon pour les grands jeux de données
        
    Returns:
        bool: True si la variable suit une distribution normale, False sinon
    """
    # Extraire les données non manquantes
    values = data[var].dropna()
    n = len(values)
    
    # Si trop peu de données, pas de test fiable
    if n < 8:
        return False
    
    # Pour les grands échantillons, prendre un échantillon aléatoire
    if n > sample_size:
        values = values.sample(sample_size, random_state=42)
        _, p_value = stats.normaltest(values)
    else:
        # Pour les petits échantillons, utiliser le test de Shapiro-Wilk
        if n <= 5000:
            _, p_value = stats.shapiro(values)
        else:
            # Pour les échantillons moyens, utiliser le test d'Agostino
            _, p_value = stats.normaltest(values)
    
    return p_value > 0.05

def check_duplicates(df, var_x, var_y=None):
    """
    Vérifie la présence de doublons dans une ou deux variables.
    
    Args:
        df (DataFrame): DataFrame contenant les variables
        var_x (str): Nom de la première variable
        var_y (str, optional): Nom de la deuxième variable
        
    Returns:
        bool: True si des doublons sont présents, False sinon
    """
    if var_y is None:
        # Cas d'une seule variable
        duplicates_x = df[var_x].duplicated().any()
        return duplicates_x
    else:
        # Cas de deux variables
        duplicates_x = df[var_x].duplicated().any()
        duplicates_y = df[var_y].duplicated().any()
        return duplicates_x or duplicates_y

def calculate_grouped_stats(data, var, groupby_col, agg_method='mean'):
    """
    Calcule les statistiques avec agrégation.
    
    Args:
        data (DataFrame): DataFrame contenant les données
        var (str): Nom de la variable à analyser
        groupby_col (str): Nom de la colonne pour le regroupement
        agg_method (str): Méthode d'agrégation ('mean', 'sum', 'median')
        
    Returns:
        tuple: Statistiques détaillées, statistiques agrégées, données agrégées
    """
    # Nettoyer les données
    clean_data = data.dropna(subset=[var, groupby_col])
    
    # Calculer les statistiques détaillées sur les données complètes
    detailed_stats = {
        'sum': clean_data[var].sum(),
        'mean': clean_data[var].mean(),
        'median': clean_data[var].median(),
        'std': clean_data[var].std(),
        'count': len(clean_data),
        'min': clean_data[var].min(),
        'max': clean_data[var].max()
    }
    
    # Agréger les données par la colonne de regroupement
    agg_data = clean_data.groupby(groupby_col).agg({var: agg_method}).reset_index()
    
    # Calculer les statistiques sur les données agrégées
    agg_stats = {
        'sum': agg_data[var].sum(),
        'mean': agg_data[var].mean(),
        'median': agg_data[var].median(),
        'std': agg_data[var].std(),
        'count': len(agg_data),
        'min': agg_data[var].min(),
        'max': agg_data[var].max()
    }
    
    return detailed_stats, agg_stats, agg_data

def create_interactive_stats_table(stats_df):
    """
    Crée un tableau de statistiques interactif avec un style amélioré.
    
    Args:
        stats_df (DataFrame): DataFrame contenant les statistiques
        
    Returns:
        None: Affiche le tableau directement
    """
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
    
    # Formatter les nombres
    if stats_df.select_dtypes(include=['number']).shape[1] > 0:
        # Détecter les colonnes numériques qui pourraient être des entiers
        for col in stats_df.select_dtypes(include=['number']).columns:
            # Vérifier si toutes les valeurs sont des entiers
            if all(val.is_integer() for val in stats_df[col].dropna() if hasattr(val, 'is_integer')):
                # Formater comme entier
                styled_df = styled_df.format({col: '{:.0f}'})
            else:
                # Formater avec décimales
                styled_df = styled_df.format({col: '{:.2f}'})
    
    # Afficher le tableau
    return st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

def calculate_regression(x, y):
    """
    Calcule la régression linéaire de manière robuste avec gestion des erreurs.
    
    Args:
        x (array): Variable indépendante
        y (array): Variable dépendante
        
    Returns:
        tuple: Coefficients de régression et statut de succès
    """
    try:
        # Éliminer les valeurs NaN ou infinies
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = np.array(x[mask])
        y_clean = np.array(y[mask])
        
        # Vérifier qu'il reste suffisamment de points
        if len(x_clean) < 2:
            return None, False
        
        # Essayer la méthode simple avec numpy
        try:
            z = np.polyfit(x_clean, y_clean, 1)
            return z, True
        except np.linalg.LinAlgError:
            # En cas d'erreur, essayer avec statsmodels qui est plus robuste
            try:
                import statsmodels.api as sm
                X = sm.add_constant(x_clean)
                model = sm.OLS(y_clean, X)
                results = model.fit()
                return [results.params[1], results.params[0]], True
            except Exception:
                return None, False
    except Exception as e:
        st.error(f"Erreur lors du calcul de la régression : {str(e)}")
        return None, False

def wrap_text(text, width):
    """
    Découpe un texte en lignes en fonction d'une largeur maximale.
    
    Args:
        text (str): Le texte à découper
        width (int): Le nombre maximum de caractères par ligne
        
    Returns:
        str: Le texte avec des retours à la ligne HTML (<br>)
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        if current_length + word_length + 1 <= width:
            current_line.append(word)
            current_length += word_length + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '<br>'.join(lines)

# Fonctions d'analyse bivariée
def analyze_qualitative_bivariate(df, var_x, var_y, exclude_missing=True):
    """
    Analyse bivariée pour deux variables qualitatives avec des statistiques améliorées.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        var_x (str): Nom de la première variable
        var_y (str): Nom de la deuxième variable
        exclude_missing (bool): Exclure les valeurs manquantes
        
    Returns:
        tuple: Tableau croisé et statistiques de réponse
    """
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
    
    # Tableau croisé de base
    crosstab_n = pd.crosstab(data[var_x], data[var_y])
    crosstab_pct = pd.crosstab(data[var_x], data[var_y], normalize='index') * 100
    col_means = crosstab_pct.mean()
    
    # Tableau combiné avec pourcentages et effectifs
    combined_table = pd.DataFrame(index=crosstab_n.index, columns=crosstab_n.columns)
    
    for idx in crosstab_n.index:
        for col in crosstab_n.columns:
            n = crosstab_n.loc[idx, col]
            pct = crosstab_pct.loc[idx, col]
            combined_table.loc[idx, col] = f"{pct:.1f}% ({n})"
    
    # Ajout des totaux
    row_totals = crosstab_n.sum(axis=1)
    combined_table['Total'] = [f"100% ({n})" for n in row_totals]
    
    # Calcul des moyennes
    mean_row = []
    for col in crosstab_n.columns:
        mean_val = col_means[col]
        total_n = crosstab_n[col].sum()
        mean_row.append(f"{mean_val:.1f}% ({total_n})")
    mean_row.append(f"100% ({crosstab_n.values.sum()})")
    
    combined_table.loc['Moyenne'] = mean_row
    
    # Calcul des statistiques de test
    chi2, p, dof, expected = stats.chi2_contingency(crosstab_n)
    cramer_v = np.sqrt(chi2 / (crosstab_n.values.sum() * min(crosstab_n.shape[0]-1, crosstab_n.shape[1]-1)))
    
    # Ajout des statistiques de test au tableau des réponses
    response_stats.update({
        'Test du Chi²': f"{chi2:.2f}",
        'p-value': f"{p:.4f}",
        'V de Cramer': f"{cramer_v:.4f}",
        'Degré de liaison': 'Fort' if cramer_v > 0.3 else 'Moyen' if cramer_v > 0.1 else 'Faible'
    })
    
    return (combined_table, response_stats) if exclude_missing else combined_table

def improved_qualitative_analysis(data_series, var_name):
    """
    Fonction améliorée pour analyser et regrouper des modalités qualitatives
    avec une meilleure ergonomie et option de renommage.
    
    Args:
        data_series: Variable à analyser (Series ou DataFrame)
        var_name: Nom de la variable
    
    Returns:
        pd.DataFrame: Tableau des fréquences
    """
    import pandas as pd
    import numpy as np
    import streamlit as st
    
    # 1. Convertir en série si nécessaire
    if isinstance(data_series, pd.DataFrame):
        data_series = data_series.iloc[:, 0]
    
    # 2. Initialiser le stockage des regroupements
    if 'qualitative_groups' not in st.session_state:
        st.session_state.qualitative_groups = {}
    
    # Créer une clé spécifique pour cette variable
    var_key = f"groups_{var_name}"
    if var_key not in st.session_state.qualitative_groups:
        st.session_state.qualitative_groups[var_key] = []
    
    # Clé pour le renommage des modalités
    rename_key = f"rename_{var_name}"
    if rename_key not in st.session_state.qualitative_groups:
        st.session_state.qualitative_groups[rename_key] = {}
    
    # 3. Obtenir une copie propre des données
    clean_data = data_series.copy()
    
    # Remplacer les valeurs manquantes
    clean_data = clean_data.fillna("Non réponse")
    
    # 4. Interface pour la gestion des modalités
    with st.expander(f"Gérer les modalités de '{var_name}'"):
        # Créer 2 onglets pour séparer regroupement et renommage
        tab1, tab2 = st.tabs(["Regrouper des modalités", "Renommer une modalité"])
        
        with tab1:
            st.subheader("Regrouper plusieurs modalités")
            # Liste des modalités disponibles
            available_modalities = sorted(clean_data.unique())
            
            # Sélection des modalités à regrouper
            selected_modalities = st.multiselect(
                "Sélectionner les modalités à regrouper:",
                options=available_modalities,
                key=f"select_{var_key}"
            )
            
            # Définir le nom du nouveau groupe
            if selected_modalities:
                new_group_name = st.text_input(
                    "Nom du nouveau groupe:",
                    value="Nouveau groupe",
                    key=f"name_{var_key}"
                )
                
                # Appliquer le regroupement
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("✅ Ajouter ce groupe", key=f"apply_{var_key}"):
                        st.session_state.qualitative_groups[var_key].append({
                            "modalites": selected_modalities,
                            "nom": new_group_name
                        })
                        # Effacer les sélections pour un prochain groupe
                        st.session_state[f"select_{var_key}"] = []
                        st.session_state[f"name_{var_key}"] = "Nouveau groupe"
                        st.success(f"Groupe '{new_group_name}' créé avec {len(selected_modalities)} modalités")
                        # Utiliser st.rerun() à la place de st.experimental_rerun()
                        try:
                            st.rerun()
                        except:
                            st.warning("Veuillez actualiser la page pour voir les changements.")
                
                with col2:
                    st.info("Après avoir ajouté ce groupe, la sélection sera effacée pour vous permettre de créer un nouveau groupe.")
        
        with tab2:
            st.subheader("Renommer une modalité")
            # Sélection d'une modalité à renommer
            # Exclure les modalités déjà dans des groupes pour éviter les conflits
            grouped_modalities = []
            for group in st.session_state.qualitative_groups[var_key]:
                grouped_modalities.extend(group["modalites"])
            
            renamable_modalities = [m for m in available_modalities if m not in grouped_modalities]
            
            if not renamable_modalities:
                st.warning("Toutes les modalités sont déjà dans des groupes. Supprimez un groupe si vous souhaitez renommer une modalité.")
            else:
                to_rename = st.selectbox(
                    "Sélectionner une modalité à renommer:",
                    options=renamable_modalities,
                    key=f"select_rename_{var_key}"
                )
                
                # Obtenir le nom actuel, ou l'ancien renommage s'il existe
                current_name = st.session_state.qualitative_groups[rename_key].get(to_rename, to_rename)
                
                # Champ de saisie pour le nouveau nom
                new_name = st.text_input(
                    "Nouveau nom:",
                    value=current_name,
                    key=f"new_name_{var_key}"
                )
                
                # Appliquer le renommage
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("✅ Renommer", key=f"apply_rename_{var_key}"):
                        st.session_state.qualitative_groups[rename_key][to_rename] = new_name
                        st.success(f"Modalité '{to_rename}' renommée en '{new_name}'")
                
                with col2:
                    # Afficher un message d'info pour le renommage
                    if to_rename != new_name:
                        st.info(f"La modalité '{to_rename}' sera renommée en '{new_name}'")
            
        # Afficher les groupes existants
        if st.session_state.qualitative_groups[var_key] or st.session_state.qualitative_groups[rename_key]:
            st.markdown("---")
            st.subheader("Modifications actuelles")
            
            # Afficher les groupes
            if st.session_state.qualitative_groups[var_key]:
                st.write("**Groupes créés:**")
                for i, group in enumerate(st.session_state.qualitative_groups[var_key]):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        modalites_text = ", ".join(group['modalites'])
                        if len(modalites_text) > 80:
                            modalites_text = modalites_text[:77] + "..."
                        st.info(f"**Groupe '{group['nom']}'**: {modalites_text}")
                        if st.checkbox("Voir toutes les modalités", key=f"view_all_{var_key}_{i}"):
                            st.write(", ".join(group['modalites']))
                    with col2:
                        if st.button("🗑️", key=f"del_{var_key}_{i}"):
                            st.session_state.qualitative_groups[var_key].pop(i)
                            # Utiliser st.rerun() à la place de st.experimental_rerun()
                            try:
                                st.rerun()
                            except:
                                st.warning("Veuillez actualiser la page pour voir les changements.")
            
            # Afficher les renommages
            if st.session_state.qualitative_groups[rename_key]:
                st.write("**Modalités renommées:**")
                rename_items = list(st.session_state.qualitative_groups[rename_key].items())
                for i, (old_name, new_name) in enumerate(rename_items):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.info(f"**'{old_name}'** → **'{new_name}'**")
                    with col2:
                        if st.button("🗑️", key=f"del_rename_{var_key}_{i}"):
                            del st.session_state.qualitative_groups[rename_key][old_name]
                            # Utiliser st.rerun() à la place de st.experimental_rerun()
                            try:
                                st.rerun()
                            except:
                                st.warning("Veuillez actualiser la page pour voir les changements.")
            
            # Option pour réinitialiser toutes les modifications
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 Réinitialiser tous les groupes", key=f"reset_groups_{var_key}"):
                    st.session_state.qualitative_groups[var_key] = []
                    # Utiliser st.rerun() à la place de st.experimental_rerun()
                    try:
                        st.rerun()
                    except:
                        st.warning("Veuillez actualiser la page pour voir les changements.")
            with col2:
                if st.button("🔄 Réinitialiser tous les renommages", key=f"reset_renames_{var_key}"):
                    st.session_state.qualitative_groups[rename_key] = {}
                    # Utiliser st.rerun() à la place de st.experimental_rerun()
                    try:
                        st.rerun()
                    except:
                        st.warning("Veuillez actualiser la page pour voir les changements.")
    
    # 5. Appliquer les renommages et regroupements
    grouped_data = clean_data.copy()
    
    # Appliquer d'abord les renommages individuels
    for old_name, new_name in st.session_state.qualitative_groups[rename_key].items():
        grouped_data = grouped_data.replace(old_name, new_name)
    
    # Puis appliquer les regroupements
    for group in st.session_state.qualitative_groups[var_key]:
        grouped_data = grouped_data.replace(group["modalites"], group["nom"])
    
    # 6. Calculer la distribution
    counts = grouped_data.value_counts().reset_index()
    counts.columns = ["Modalités", "Effectif"]
    
    # Calculer les pourcentages
    total = counts["Effectif"].sum()
    counts["Pourcentage"] = (counts["Effectif"] / total * 100).round(1)
    
    return counts

def export_table_as_image(value_counts, title, source, note):
    """ Exporte un tableau en image (PNG) avec titre, source et note. """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Ajout du titre
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    # Conversion du DataFrame en texte pour affichage
    cell_text = []
    for i, row in value_counts.iterrows():
        cell_text.append([
            str(row['Modalité']),
            f"{row['Effectif']:,}",
            f"{row['Taux (%)']:.1f}%"
        ])

    # Création du tableau matplotlib
    table = ax.table(
        cellText=cell_text,
        colLabels=['Modalité', 'Effectif', 'Taux (%)'],
        loc='center',
        cellLoc='center'
    )

    # Ajustements de style
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    # Ajout de la source et de la note en bas
    text_y = -0.15 - (0.02 * len(value_counts))
    if source:
        plt.figtext(0.1, text_y, f"📌 Source : {source}", fontsize=10, ha="left", style="italic")
    if note:
        plt.figtext(0.1, text_y - 0.04, f"📌 Note : {note}", fontsize=10, ha="left", style="italic")

    # Sauvegarde de l’image en mémoire
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
    buffer.seek(0)
    
    return buffer

def analyze_mixed_bivariate(df, qual_var, quant_var):
    """
    Analyse bivariée pour une variable qualitative et une quantitative avec des statistiques améliorées.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        qual_var (str): Nom de la variable qualitative
        quant_var (str): Nom de la variable quantitative
        
    Returns:
        tuple: DataFrame de statistiques et taux de réponse
    """
    data = df.copy()
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    data[qual_var] = data[qual_var].replace(missing_values, np.nan)
    data[quant_var] = data[quant_var].replace(missing_values, np.nan)
    data = data.dropna(subset=[qual_var, quant_var])
    
    # Statistiques par modalité
    stats_df = data.groupby(qual_var)[quant_var].agg([
        ('Effectif', 'count'),
        ('Total', lambda x: x.sum()),
        ('Moyenne', 'mean'),
        ('Médiane', 'median'),
        ('Écart-type', 'std'),
        ('Minimum', 'min'),
        ('Maximum', 'max')
    ]).round(2)
    
    # Ajout des percentiles (Q1 et Q3)
    stats_df['Q1 (25%)'] = data.groupby(qual_var)[quant_var].quantile(0.25).round(2)
    stats_df['Q3 (75%)'] = data.groupby(qual_var)[quant_var].quantile(0.75).round(2)
    
    # Statistiques globales
    total_stats = pd.DataFrame({
        'Effectif': data[quant_var].count(),
        'Total': data[quant_var].sum(),
        'Moyenne': data[quant_var].mean(),
        'Médiane': data[quant_var].median(),
        'Écart-type': data[quant_var].std(),
        'Minimum': data[quant_var].min(),
        'Maximum': data[quant_var].max(),
        'Q1 (25%)': data[quant_var].quantile(0.25),
        'Q3 (75%)': data[quant_var].quantile(0.75)
    }, index=['Total']).round(2)
    
    # Combinaison des statistiques
    stats_df = pd.concat([stats_df, total_stats])
    
    # Calcul du taux de réponse
    response_rate = (data[qual_var].count() / len(df)) * 100
    
    # Test ANOVA pour déterminer si les différences sont significatives
    try:
        groups = [data[data[qual_var] == group][quant_var].dropna() for group in data[qual_var].unique()]
        f_val, p_val = stats.f_oneway(*groups)
        
        # Ajouter ces informations aux statistiques
        anova_stats = pd.DataFrame({
            'Test ANOVA': ['F-value', 'p-value', 'Significatif'],
            'Valeur': [f_val, p_val, 'Oui' if p_val < 0.05 else 'Non']
        })
    except:
        anova_stats = pd.DataFrame({
            'Test ANOVA': ['Erreur'],
            'Valeur': ['Test ANOVA non réalisable']
        })
    
    return stats_df, response_rate, anova_stats

def analyze_quantitative_bivariate(df, var_x, var_y, groupby_col=None, agg_method='sum'):
    """
    Analyse bivariée pour deux variables quantitatives avec des statistiques améliorées.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        var_x (str): Nom de la première variable quantitative
        var_y (str): Nom de la deuxième variable quantitative
        groupby_col (str, optional): Colonne pour l'agrégation
        agg_method (str): Méthode d'agrégation ('sum', 'mean', 'median')
        
    Returns:
        tuple: DataFrame de résultats et taux de réponse
    """
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
    
    # Test de normalité
    is_normal_x = check_normality(data, var_x)
    is_normal_y = check_normality(data, var_y)
    
    # Calcul de la corrélation appropriée
    if is_normal_x and is_normal_y:
        correlation_method = "Pearson"
        correlation, p_value = stats.pearsonr(data[var_x], data[var_y])
    else:
        correlation_method = "Spearman"
        correlation, p_value = stats.spearmanr(data[var_x], data[var_y])
    
    # Interprétation de la corrélation
    correlation_strength = "Très forte"
    if abs(correlation) < 0.19:
        correlation_strength = "Très faible"
    elif abs(correlation) < 0.39:
        correlation_strength = "Faible"
    elif abs(correlation) < 0.59:
        correlation_strength = "Modérée"
    elif abs(correlation) < 0.79:
        correlation_strength = "Forte"
    
    correlation_direction = "positive" if correlation > 0 else "négative"
    
    # Régression linéaire
    regression_coeffs, regression_success = calculate_regression(data[var_x], data[var_y])
    
    # Format des résultats
    results_dict = {
        "Test de corrélation": [correlation_method],
        "Coefficient": [round(correlation, 3)],
        "P-value": [round(p_value, 5)],
        "Significativité": ["Significatif" if p_value < 0.05 else "Non significatif"],
        "Intensité": [correlation_strength],
        "Sens": [correlation_direction],
        "Nombre d'observations": [len(data)]
    }
    
    if regression_success:
        results_dict["Equation de régression"] = [f"y = {regression_coeffs[0]:.4f}x + {regression_coeffs[1]:.4f}"]
    
    if groupby_col is not None:
        results_dict["Note"] = [f"Données agrégées par {groupby_col} ({agg_method})"]
    
    results_df = pd.DataFrame(results_dict)
    
    # Calcul des taux de réponse
    response_rate_x = (df[var_x].notna().sum() / len(df)) * 100
    response_rate_y = (df[var_y].notna().sum() / len(df)) * 100
    
    # Statistiques descriptives bivarées
    descriptive_stats = pd.DataFrame({
        "Statistique": ["Minimum", "Maximum", "Moyenne", "Médiane", "Écart-type"],
        f"{var_x}": [
            data[var_x].min(),
            data[var_x].max(),
            data[var_x].mean().round(2),
            data[var_x].median().round(2),
            data[var_x].std().round(2)
        ],
        f"{var_y}": [
            data[var_y].min(),
            data[var_y].max(),
            data[var_y].mean().round(2),
            data[var_y].median().round(2),
            data[var_y].std().round(2)
        ]
    })
    
    return results_df, response_rate_x, response_rate_y, descriptive_stats, data

def create_enhanced_variable_selector(df, title="Sélectionnez une variable"):
    """
    Crée un sélecteur de variables amélioré avec filtrage et aperçu.

    Args:
        df (DataFrame): DataFrame contenant les variables
        title (str): Titre du sélecteur

    Returns:
        str | None: Nom de la variable sélectionnée ou None si erreur
    """
    st.markdown(f"### {title}")

    # ✅ Vérification : si le DataFrame est vide
    if df.empty:
        st.error("🚨 Le DataFrame est vide. Vérifiez les données chargées !")
        return None

    # ✅ Vérification si le DataFrame a des colonnes
    if df.columns.empty:
        st.error("⚠️ Le DataFrame ne contient aucune colonne.")
        return None

    # Liste des colonnes disponibles
    variables = list(df.columns)

    # Sélecteur de variable
    selected_var = st.selectbox("Variable sélectionnée", variables)

    # ✅ Vérification AVANT d'accéder à df[selected_var]
    if selected_var is None:
        st.error("❌ Aucune variable sélectionnée.")
        return None
    
    if selected_var not in df.columns:
        st.error(f"❌ La variable '{selected_var}' n'existe pas dans le DataFrame.")
        return None

    try:
        dtype = df[selected_var].dtype  # ✅ Vérification d'accès
        return selected_var
    except Exception as e:
        st.error(f"🚨 Erreur lors de la récupération du type de '{selected_var}': {e}")
        return None

def create_tabbed_interface():
    """
    Crée une interface de type radio pour la navigation.
    
    Returns:
        str: Nom de l'onglet sélectionné
    """
    selected_tab = st.radio(
        "Sélectionner une analyse",
        ["📊 Analyse univariée", "🔄 Analyse bivariée", 
         "📈 Séries temporelles", "🔍 Filtrage & exploration", 
         "💾 Export des résultats"],
        horizontal=True
    )
    
    tab_map = {
        "📊 Analyse univariée": "univariate",
        "🔄 Analyse bivariée": "bivariate", 
        "📈 Séries temporelles": "time_series",
        "🔍 Filtrage & exploration": "exploration",
        "💾 Export des résultats": "export"
    }
    
    return tab_map[selected_tab]

def setup_sidebar_filters(df):
    """
    Configure des filtres dans la barre latérale pour filtrer les données.
    
    Args:
        df (DataFrame): DataFrame à filtrer
        
    Returns:
        DataFrame: DataFrame filtré
    """
    st.sidebar.header("Filtres de données")
    
    filtered_df = df.copy()
    filters_applied = False
    
    # Séparation des colonnes par type
    numeric_cols = []
    categorical_cols = []
    
    # Identification plus précise des types de colonnes
    for col in df.columns:
        try:
            # Vérifier si la colonne est réellement numérique
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                # Vérification supplémentaire pour s'assurer que c'est vraiment numérique
                test_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if test_val is not None and isinstance(test_val, (int, float)):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                # Si la colonne a moins de 20 valeurs uniques, on la considère catégorielle
                if df[col].nunique() <= 20:
                    categorical_cols.append(col)
        except:
            # En cas d'erreur, on ne crée pas de filtre
            st.sidebar.warning(f"Impossible de créer un filtre pour {col}")
    
    # Filtres numériques
    if numeric_cols:
        with st.sidebar.expander("Filtres numériques", expanded=False):
            for col in numeric_cols[:5]:  # Limiter à 5 colonnes pour éviter l'encombrement
                try:
                    # Extraction sécurisée des valeurs min/max
                    valid_values = df[col].dropna()
                    if len(valid_values) > 0:  # S'assurer qu'il y a des valeurs valides
                        min_val = float(valid_values.min())
                        max_val = float(valid_values.max())
                        
                        # Vérifier si les valeurs sont trop proches
                        if abs(max_val - min_val) < 1e-10:
                            st.sidebar.info(f"Toutes les valeurs de {col} sont identiques ({min_val})")
                            continue
                        
                        # Calculer le pas en fonction de la plage et du type
                        try:
                            is_integer = all(float(x).is_integer() for x in valid_values if pd.notna(x))
                        except:
                            is_integer = False
                            
                        # Définir un pas approprié
                        value_range = max_val - min_val
                        step = 1 if is_integer else value_range / 100
                        step = max(step, 1e-6)  # Éviter les pas trop petits
                        
                        # Créer le slider
                        filter_vals = st.sidebar.slider(
                            f"Filtre sur {col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            step=step
                        )
                        
                        if filter_vals != (min_val, max_val):
                            filtered_df = filtered_df[(filtered_df[col] >= filter_vals[0]) & 
                                                    (filtered_df[col] <= filter_vals[1])]
                            filters_applied = True
                except Exception as e:
                    st.sidebar.warning(f"Impossible de créer un filtre pour {col}")
    
    # Filtres catégoriels
    if categorical_cols:
        with st.sidebar.expander("Filtres catégoriels", expanded=False):
            for col in categorical_cols[:5]:  # Limiter à 5 colonnes
                try:
                    # Limiter à 100 valeurs uniques max pour éviter les problèmes de performance
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) > 100:
                        st.sidebar.info(f"{col} a trop de valeurs uniques ({len(unique_vals)}) pour un filtre")
                        continue
                        
                    # Convertir en string pour éviter les problèmes avec certains types
                    unique_vals = [str(val) for val in unique_vals]
                    
                    selected_vals = st.sidebar.multiselect(
                        f"Filtre sur {col}",
                        options=unique_vals,
                        default=unique_vals
                    )
                    
                    if len(selected_vals) < len(unique_vals):
                        # Convertir les valeurs de la colonne en string pour la comparaison
                        filtered_df = filtered_df[filtered_df[col].astype(str).isin(selected_vals)]
                        filters_applied = True
                except Exception as e:
                    st.sidebar.warning(f"Impossible de créer un filtre pour {col}: {str(e)}")
    
    # Filtre sur les valeurs manquantes
    with st.sidebar.expander("Filtre de valeurs manquantes", expanded=False):
        handle_missing = st.sidebar.radio(
            "Gestion des valeurs manquantes",
            ["Conserver toutes les lignes", "Exclure les lignes avec valeurs manquantes"]
        )
        
        if handle_missing == "Exclure les lignes avec valeurs manquantes":
            filtered_df = filtered_df.dropna()
            filters_applied = True
    
    # Afficher un résumé des filtres
    if filters_applied:
        st.sidebar.success(f"{len(filtered_df)} lignes sur {len(df)} après filtrage ({(len(filtered_df)/len(df)*100):.1f}%)")
    
    return filtered_df

def plot_density(data, var_name, title, x_label, y_label, color_palette=None, kde_bandwidth=None):
    """
    Crée un graphique de densité amélioré pour les variables quantitatives.
    
    Args:
        data (Series): Série contenant les données
        var_name (str): Nom de la variable
        title (str): Titre du graphique
        x_label (str): Étiquette de l'axe X
        y_label (str): Étiquette de l'axe Y
        color_palette (list): Liste de couleurs
        kde_bandwidth (float): Largeur de bande pour l'estimation par noyau
        
    Returns:
        go.Figure: Figure Plotly
    """
    if color_palette is None:
        color_palette = ['#000091']  # Bleu Marianne par défaut
    
    # Nettoyage des données
    clean_data = data.dropna()
    
    # Créer un histogramme avec courbe KDE
    fig = go.Figure()
    
    # Ajouter l'histogramme
    fig.add_trace(go.Histogram(
        x=clean_data,
        histnorm='probability density',
        name='Histogramme',
        marker=dict(
            color=color_palette[0],
            line=dict(color='white', width=1)
        ),
        opacity=0.7,
        nbinsx=30
    ))
    
    # Ajouter la courbe KDE
    try:
        from scipy import stats
        
        # Déterminer la largeur de bande automatiquement si non spécifiée
        if kde_bandwidth is None:
            kde_bandwidth = 'scott'  # Méthode de Scott pour la détermination automatique
        
        # Calculer la KDE
        kde = stats.gaussian_kde(clean_data, bw_method=kde_bandwidth)
        
        # Générer les points pour la courbe
        x_range = np.linspace(min(clean_data), max(clean_data), 1000)
        y_kde = kde(x_range)
        
        # Ajouter la courbe
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_kde,
            mode='lines',
            name='Densité (KDE)',
            line=dict(color='#E1000F', width=2)  # Rouge Marianne
        ))
        
        # Ajouter les quartiles
        quartiles = [clean_data.quantile(q) for q in [0.25, 0.5, 0.75]]
        
        for i, q in enumerate(quartiles):
            fig.add_shape(
                type='line',
                x0=q,
                y0=0,
                x1=q,
                y1=max(y_kde) * 0.9,
                line=dict(
                    color='#53657D',  # Gris Marianne
                    width=1.5,
                    dash='dash'
                )
            )
        
        # Ajouter les annotations pour les quartiles
        fig.add_annotation(
            x=quartiles[0],
            y=max(y_kde) * 0.95,
            text="Q1",
            showarrow=False,
            font=dict(size=12, color='#53657D')
        )
        
        fig.add_annotation(
            x=quartiles[1],
            y=max(y_kde) * 0.95,
            text="Médiane",
            showarrow=False,
            font=dict(size=12, color='#53657D')
        )
        
        fig.add_annotation(
            x=quartiles[2],
            y=max(y_kde) * 0.95,
            text="Q3",
            showarrow=False,
            font=dict(size=12, color='#53657D')
        )
        
    except Exception as e:
        st.warning(f"Impossible de calculer la courbe de densité : {str(e)}")
    
    # Configuration du layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, family="Marianne, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor='white',
        bargap=0.1,
        height=500,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_time_series(df, time_col, value_col, group_col=None, agg_method='mean', title=None, color_palette=None):
    """
    Crée une visualisation de série temporelle.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        time_col (str): Nom de la colonne de temps
        value_col (str): Nom de la colonne de valeur
        group_col (str, optional): Nom de la colonne pour grouper les données
        agg_method (str): Méthode d'agrégation ('mean', 'sum', 'median')
        title (str, optional): Titre du graphique
        color_palette (list, optional): Liste de couleurs
        
    Returns:
        go.Figure: Figure Plotly
    """
    if color_palette is None:
        color_palette = ['#000091', '#E1000F', '#169B62', '#53657D', '#FFC800']
    
    # Vérifier et convertir la colonne de temps si nécessaire
    df = df.copy()
    
    try:
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
    except:
        st.error(f"Impossible de convertir la colonne {time_col} en date/heure.")
        return None
    
    # Créer le graphique
    fig = go.Figure()
    
    if group_col:
        # Agréger les données par groupe et par temps
        grouped_df = df.groupby([time_col, group_col])[value_col].agg(agg_method).reset_index()
        
        # Trier par date
        grouped_df = grouped_df.sort_values(time_col)
        
        # Ajouter une trace pour chaque groupe
        for i, group in enumerate(grouped_df[group_col].unique()):
            group_data = grouped_df[grouped_df[group_col] == group]
            fig.add_trace(go.Scatter(
                x=group_data[time_col],
                y=group_data[value_col],
                mode='lines+markers',
                name=str(group),
                line=dict(color=color_palette[i % len(color_palette)]),
                marker=dict(size=6)
            ))
    else:
        # Agréger les données par temps seulement
        grouped_df = df.groupby(time_col)[value_col].agg(agg_method).reset_index()
        
        # Trier par date
        grouped_df = grouped_df.sort_values(time_col)
        
        # Ajouter la trace unique
        fig.add_trace(go.Scatter(
            x=grouped_df[time_col],
            y=grouped_df[value_col],
            mode='lines+markers',
            name=value_col,
            line=dict(color=color_palette[0]),
            marker=dict(size=6)
        ))
    
    # Configuration du layout
    fig.update_layout(
        title=dict(
            text=title or f"Série temporelle de {value_col}",
            font=dict(size=20, family="Marianne, sans-serif"),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Date",
        yaxis_title=value_col,
        plot_bgcolor='white',
        height=500,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    # Ajout de grilles pour une meilleure lisibilité
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        tickangle=-45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0'
    )
    
    return fig

def analyze_time_series(df, time_col, value_col, group_col=None, freq='MS'):
    """
    Effectue une analyse de série temporelle.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        time_col (str): Nom de la colonne de temps
        value_col (str): Nom de la colonne de valeur
        group_col (str, optional): Nom de la colonne pour grouper les données
        freq (str): Fréquence de ré-échantillonnage ('MS': début de mois, 'D': jour, etc.)
        
    Returns:
        tuple: DataFrames d'analyse
    """
    # Vérifier et convertir la colonne de temps si nécessaire
    df = df.copy()
    
    try:
        if not pd.api.types.is_datetime64_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
    except:
        st.error(f"Impossible de convertir la colonne {time_col} en date/heure.")
        return None, None, None
    
    # Trier par date
    df = df.sort_values(time_col)
    
    # Préparer le DataFrame pour l'analyse
    if group_col:
        # Pivoter pour avoir une colonne par groupe
        pivot_df = df.pivot_table(
            index=time_col,
            columns=group_col,
            values=value_col,
            aggfunc='mean'
        ).reset_index()
        
        # Réechantillonner à la fréquence spécifiée
        resampled_df = pivot_df.set_index(time_col).resample(freq).mean().reset_index()
        
        # Calculer les statistiques par groupe
        group_stats = {}
        for group in pivot_df.columns[1:]:  # Ignorer la colonne de temps
            group_data = pivot_df[group].dropna()
            group_stats[group] = {
                'Moyenne': group_data.mean(),
                'Médiane': group_data.median(),
                'Écart-type': group_data.std(),
                'Min': group_data.min(),
                'Max': group_data.max(),
                'Tendance (% variation)': ((group_data.iloc[-1] / group_data.iloc[0]) - 1) * 100 if len(group_data) > 1 else 0
            }
        
        group_stats_df = pd.DataFrame(group_stats).T
        
        # Calculer la corrélation entre les groupes
        corr_df = pivot_df.set_index(time_col).corr()
        
        return resampled_df, group_stats_df, corr_df
    
    else:
        # Agréger les données par temps
        agg_df = df.groupby(time_col)[value_col].mean().reset_index()
        
        # Réechantillonner à la fréquence spécifiée
        resampled_df = agg_df.set_index(time_col).resample(freq).mean().reset_index()
        
        # Calculer des statistiques descriptives
        data = agg_df[value_col].dropna()
        stats = {
            'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', 'Tendance (% variation)'],
            'Valeur': [
                data.mean(),
                data.median(),
                data.std(),
                data.min(),
                data.max(),
                ((data.iloc[-1] / data.iloc[0]) - 1) * 100 if len(data) > 1 else 0
            ]
        }
        stats_df = pd.DataFrame(stats)
        
        # Calculer l'autocorrélation
        try:
            from statsmodels.tsa.stattools import acf
            lag_acf = acf(data, nlags=min(10, len(data) - 1))
            acf_df = pd.DataFrame({
                'Lag': range(len(lag_acf)),
                'Autocorrélation': lag_acf
            })
        except:
            acf_df = pd.DataFrame({'Erreur': ['Impossible de calculer l\'autocorrélation']})
        
        return resampled_df, stats_df, acf_df

def detect_variable_to_aggregate(df, var_x, var_y, groupby_col):
    """
    Détecte quelles variables doivent être agrégées et lesquelles doivent être conservées telles quelles.
    
    Args:
        df (DataFrame): DataFrame contenant les données
        var_x (str): Nom de la première variable
        var_y (str): Nom de la deuxième variable
        groupby_col (str): Nom de la colonne de regroupement
        
    Returns:
        tuple: Listes des variables à agréger et à conserver
    """
    vars_to_aggregate = []
    vars_to_keep_raw = []
    
    for var in [var_x, var_y]:
        if df.groupby(groupby_col)[var].nunique().max() > 1:
            vars_to_aggregate.append(var)
        else:
            vars_to_keep_raw.append(var)
    
    return vars_to_aggregate, vars_to_keep_raw

def show_indicator_form(statistics, analysis_type, variables_info):
    """
    Interface de création d'indicateur.
    
    Args:
        statistics (dict): Statistiques à enregistrer
        analysis_type (str): Type d'analyse ('univariate', 'bivariate', etc.)
        variables_info (dict): Informations sur les variables
    """
    st.write("### Création d'indicateur")
    
    with st.form("indicator_form"):
        name = st.text_input("Nom de l'indicateur")
        description = st.text_area("Description")
        category = st.selectbox("Catégorie", ["Éducation", "Recherche", "Réussite", "Économie", "Autre"])
        source = st.text_input("Source", 
                              value=variables_info.get('source', '') if 'source' in variables_info else '')
        creation_date = datetime.now().strftime("%Y-%m-%d")
        
        # Champs spécifiques selon le type d'analyse
        if analysis_type == 'univariate':
            variable_field = st.text_input("Variable", 
                                         value=variables_info.get('var_name', '') if 'var_name' in variables_info else '')
        elif analysis_type == 'bivariate':
            if 'var_qual' in variables_info and 'var_quant' in variables_info:
                variable_field = st.text_input("Variable qualitative", 
                                             value=variables_info.get('var_qual', ''))
                variable_field2 = st.text_input("Variable quantitative", 
                                              value=variables_info.get('var_quant', ''))
            else:
                variable_field = st.text_input("Variable X", 
                                             value=variables_info.get('var_x', '') if 'var_x' in variables_info else '')
                variable_field2 = st.text_input("Variable Y", 
                                              value=variables_info.get('var_y', '') if 'var_y' in variables_info else '')
        
        # Statistiques à enregistrer
        st.write("### Statistiques")
        stats_to_include = {}
        
        for stat_name, stat_value in statistics.items() if isinstance(statistics, dict) else statistics[0].items():
            include = st.checkbox(f"Inclure '{stat_name}'", value=True)
            if include:
                stats_to_include[stat_name] = stat_value
        
        if st.form_submit_button("Enregistrer l'indicateur"):
            try:
                # Préparation des données pour l'API
                indicator_data = {
                    "name": name,
                    "description": description,
                    "category": category,
                    "creation_date": creation_date,
                    "source": source,
                    "statistics": json.dumps(stats_to_include),
                    "analysis_type": analysis_type
                }
                
                if analysis_type == 'univariate':
                    indicator_data["variable"] = variable_field
                elif analysis_type == 'bivariate':
                    indicator_data["variable_x"] = variable_field
                    indicator_data["variable_y"] = variable_field2
                
                # Appel à la fonction d'enregistrement
                save_indicator(indicator_data)
                st.success("✅ Indicateur enregistré avec succès!")
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement : {str(e)}")

def save_indicator(indicator_data):
    """
    Sauvegarde un indicateur dans Grist.
    
    Args:
        indicator_data (dict): Données de l'indicateur à sauvegarder
    """
    try:
        # Table des indicateurs (à adapter selon votre structure)
        grist_data = {
            "records": [
                {
                    "fields": indicator_data
                }
            ]
        }
        
        # Remplacer "indicators" par l'ID de votre table d'indicateurs
        response = grist_api_request("indicators", method="POST", data=grist_data)
        return response
    except Exception as e:
        raise Exception(f"Erreur lors de l'enregistrement de l'indicateur : {str(e)}")

# Structure principale de l'application
def main():
    st.title("Analyse des données ESR 2025")

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

    # Filtres dans la barre latérale
    filtered_data = setup_sidebar_filters(st.session_state.merged_data)
    
    # Interface à onglets
    active_tab = create_tabbed_interface()
    
    if active_tab == "univariate":
        # Sélection de la variable pour l'analyse univariée
        var = create_enhanced_variable_selector(filtered_data, "Sélectionnez la variable pour l'analyse univariée")
        
# Cette section doit être placée dans la partie active_tab == "univariate"
if var is not None:
    # Préparation des données
    plot_data = filtered_data[var].copy()
    
    if plot_data is not None and not plot_data.empty:
        # Détection du type de variable (déplacée ici pour être définie avant utilisation)
        is_numeric = is_numeric_column(filtered_data, var)
        
        # Affichage du dashboard de résumé approprié
        st.write(f"### Résumé de la variable {var}")
        
        if is_numeric:
            # Pour les variables numériques
            # Essayer de convertir en numérique explicitement
            try:
                numeric_plot_data = pd.to_numeric(plot_data, errors='coerce')
                
                # Vérifier si nous avons des données valides après conversion
                if numeric_plot_data.notna().any():
                    # Statistiques descriptives avec les données converties
                    stats_df = pd.DataFrame({
                        'Statistique': ['Effectif total', 'Somme', 'Moyenne', 'Médiane', 'Écart-type', 'Minimum', 'Maximum'],
                        'Valeur': [
                            len(numeric_plot_data.dropna()),
                            numeric_plot_data.sum().round(2),
                            numeric_plot_data.mean().round(2),
                            numeric_plot_data.median().round(2),
                            numeric_plot_data.std().round(2),
                            numeric_plot_data.min(),
                            numeric_plot_data.max()
                        ]
                    })
                    create_interactive_stats_table(stats_df)
                    
                    # Utiliser numeric_plot_data pour le reste de l'analyse
                    plot_data = numeric_plot_data.dropna()
                    
                    # Gestion des doublons si nécessaire
                    has_duplicates = check_duplicates(filtered_data, var)
                    if has_duplicates:
                        st.warning("⚠️ Certaines observations sont répétées dans le jeu de données. "
                                  "Vous pouvez choisir d'agréger les données avant l'analyse.")
                        do_aggregate = st.checkbox("Agréger les données avant l'analyse")
                        
                        if do_aggregate:
                            groupby_cols = [col for col in filtered_data.columns if col != var]
                            groupby_col = st.selectbox("Sélectionner la colonne d'agrégation", groupby_cols)
                            agg_method = st.radio(
                                "Méthode d'agrégation", 
                                ['sum', 'mean', 'median'],
                                format_func=lambda x: {'sum': 'Somme', 'mean': 'Moyenne', 'median': 'Médiane'}[x]
                            )
                            clean_data = filtered_data.dropna(subset=[var, groupby_col])
                            agg_data = clean_data.groupby(groupby_col).agg({var: agg_method}).reset_index()
                            plot_data = agg_data[var]
                            
                            st.info("Note : Les statistiques sont calculées à l'échelle de la variable d'agrégation sélectionnée.")
                    
                    # Options de regroupement
                    st.write("### Options de regroupement")
                    grouping_method = st.selectbox("Méthode de regroupement", ["Aucune", "Quantile", "Manuelle"], key="grouping_method_data")
                    
                    # Vérification si les valeurs sont des entiers
                    is_integer_variable = all(float(x).is_integer() for x in plot_data.dropna() if pd.notna(x))
                    
                    if grouping_method == "Quantile":
                        quantile_type = st.selectbox(
                            "Type de regroupement",
                            ["Quartile (4 groupes)", "Quintile (5 groupes)", "Décile (10 groupes)"]
                        )
                        n_groups = {"Quartile (4 groupes)": 4, "Quintile (5 groupes)": 5, "Décile (10 groupes)": 10}[quantile_type]
                        
                        # Création des labels et de l'ordre de tri
                        if quantile_type == "Quartile (4 groupes)":
                            labels = [f"{i}er quartile" if i == 1 else f"{i}ème quartile" for i in range(1, 5)]
                            sort_order = list(range(1, 5))
                        elif quantile_type == "Quintile (5 groupes)":
                            labels = [f"{i}er quintile" if i == 1 else f"{i}ème quintile" for i in range(1, 6)]
                            sort_order = list(range(1, 6))
                        else:  # Déciles
                            # Ajout du zéro de padding pour un tri correct
                            labels = [f"{'01' if i == 1 else f'{i:02d}'}{'er' if i == 1 else 'ème'} décile" for i in range(1, 11)]
                            sort_order = list(range(1, 11))
                        
                        try:
                            # Vérifier le nombre de valeurs uniques
                            unique_values = plot_data.nunique()
                            
                            if unique_values < n_groups:
                                st.warning(f"Attention : La variable contient seulement {unique_values} valeurs uniques, ce qui est inférieur aux {n_groups} groupes demandés.")
                            
                            # Créer les groupes avec gestion des duplicates
                            grouped_data = pd.qcut(plot_data, q=n_groups, labels=labels, duplicates='drop')
                            
                            # Compter les groupes réels
                            actual_groups = grouped_data.nunique()
                            if actual_groups < n_groups:
                                st.warning(f"Seulement {actual_groups} groupes distincts ont pu être créés.")
                            
                            # Créer un DataFrame avec l'ordre explicite pour le tri
                            value_counts = pd.DataFrame()
                            
                            # Ajout d'une colonne pour l'ordre de tri
                            group_order_map = {label: order for label, order in zip(labels, sort_order)}
                            
                            # Compter les valeurs par groupe
                            grouped_counts = grouped_data.value_counts().reset_index()
                            grouped_counts.columns = ['Groupe', 'Effectif']
                            
                            # Ajouter une colonne d'ordre pour le tri
                            grouped_counts['Ordre'] = grouped_counts['Groupe'].map(lambda x: group_order_map.get(x, 999))
                            
                            # Trier par l'ordre numérique (pas alphabétique)
                            value_counts = grouped_counts.sort_values('Ordre')
                            
                            # Calculer les pourcentages
                            value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100).round(1)
                            
                            # Statistiques par groupe
                            group_stats = plot_data.groupby(grouped_data).agg(['sum', 'mean', 'max'])
                            
                            # Formatage si entiers
                            if is_integer_variable:
                                group_stats = group_stats.applymap(
                                    lambda x: int(x) if pd.notna(x) and float(x).is_integer() else round(x, 2) if pd.notna(x) else x
                                )
                            else:
                                group_stats = group_stats.round(2)
                            
                            group_stats.columns = ['Somme', 'Moyenne', 'Maximum']
                            
                            # Jointure des DataFrames
                            st.write("### Statistiques par groupe")
                            merged_stats = value_counts.set_index('Groupe')
                            merged_stats = merged_stats.join(group_stats)
                            
                            # Affichage sans la colonne Ordre
                            display_stats = merged_stats.drop(columns=['Ordre'])
                            st.dataframe(display_stats)
                            
                        except ValueError as e:
                            st.error(f"Impossible de créer les groupes: {e}")
                            if st.button("Essayer un autre regroupement"):
                                # Regroupement par intervalle
                                min_val = plot_data.min()
                                max_val = plot_data.max()
                                interval = (max_val - min_val) / n_groups
                                bins = [min_val + i * interval for i in range(n_groups + 1)]
                                grouped_data_alt = pd.cut(plot_data, bins=bins, labels=labels)
                                value_counts_alt = pd.DataFrame({
                                    'Groupe': grouped_data_alt.value_counts().index,
                                    'Effectif': grouped_data_alt.value_counts().values
                                })
                                value_counts_alt['Taux (%)'] = (value_counts_alt['Effectif'] / len(plot_data) * 100).round(1)
                                st.dataframe(value_counts_alt)
                                
                    elif grouping_method == "Manuelle":
                        # Code pour le regroupement manuel
                        n_groups = st.number_input("Nombre de groupes", min_value=2, value=3)
                        breaks = []
                        
                        # Déterminer les seuils
                        min_val = float(plot_data.min())
                        max_val = float(plot_data.max())
                        
                        for i in range(n_groups + 1):
                            if i == 0:
                                val = min_val
                            elif i == n_groups:
                                val = max_val
                            else:
                                suggested_val = min_val + (i/n_groups)*(max_val-min_val)
                                val = st.number_input(
                                    f"Seuil {i}", 
                                    value=int(suggested_val) if is_integer_variable else float(suggested_val),
                                    min_value=int(min_val) if is_integer_variable else float(min_val),
                                    max_value=int(max_val) if is_integer_variable else float(max_val),
                                    step=1 if is_integer_variable else 0.1,
                                    key=f"seuil_{i}_{var}"  # Clé unique
                                )
                            breaks.append(val)
                        
                        # Créer les groupes
                        try:
                            if all(breaks[i] <= breaks[i+1] for i in range(len(breaks)-1)):
                                grouped_data = pd.cut(plot_data, bins=breaks)
                                value_counts = grouped_data.value_counts().reset_index()
                                value_counts.columns = ['Groupe', 'Effectif']
                                
                                # Tri par la borne inférieure de l'intervalle
                                value_counts = value_counts.sort_values('Groupe', key=lambda x: x.map(lambda y: y.left))
                                value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100).round(1)
                                
                                # Statistiques par groupe
                                group_stats = plot_data.groupby(grouped_data).agg(['sum', 'mean', 'max'])
                                
                                if is_integer_variable:
                                    group_stats = group_stats.applymap(
                                        lambda x: int(x) if pd.notna(x) and float(x).is_integer() else round(x, 2) if pd.notna(x) else x
                                    )
                                else:
                                    group_stats = group_stats.round(2)
                                    
                                group_stats.columns = ['Somme', 'Moyenne', 'Maximum']
                                
                                st.write("### Statistiques par groupe")
                                merged_stats = value_counts.set_index('Groupe')
                                merged_stats = merged_stats.join(group_stats)
                                
                                st.dataframe(merged_stats)
                            else:
                                st.error("Les seuils doivent être en ordre croissant.")
                        except Exception as e:
                            st.error(f"Erreur lors du regroupement: {str(e)}")
                    
                    # Options de visualisation pour les variables numériques
                    st.write("### Options de visualisation")
                    viz_col1, viz_col2 = st.columns([1, 2])
                    
                    with viz_col1:
                        if grouping_method == "Aucune":
                            graph_type = st.selectbox("Type de graphique", ["Histogramme", "Density plot"], key="graph_type_viz")
                        elif grouping_method == "Quantile":
                            graph_type = st.selectbox("Type de graphique", ["Boîte à moustaches", "Violin plot"], key="graph_type_viz")
                        else:
                            graph_type = st.selectbox("Type de graphique", ["Bar plot", "Lollipop plot"], key="graph_type_viz")
                    
                    with viz_col2:
                        color_scheme = st.selectbox("Palette de couleurs", list(COLOR_PALETTES.keys()), key="color_scheme_viz")
                    
                    # Options avancées pour la visualisation
                    with st.expander("Options avancées"):
                        title = st.text_input("Titre du graphique", f"Distribution de {var}", key="title_viz")
                        x_axis = st.text_input("Titre de l'axe X", var, key="x_axis_viz")
                        y_axis = st.text_input("Titre de l'axe Y", "Valeur", key="y_axis_viz")
                        source = st.text_input("Source des données", "", key="source_viz")
                        note = st.text_input("Note de lecture", "", key="note_viz")
                        show_values = st.checkbox("Afficher les valeurs", True, key="show_values_viz")
                    
                    # Génération du graphique
                    if st.button("Générer la visualisation", key="generate_viz"):
                        try:
                            if graph_type == "Histogramme":
                                fig = px.histogram(
                                    plot_data, 
                                    title=title, 
                                    color_discrete_sequence=COLOR_PALETTES[color_scheme]
                                )
                                if show_values:
                                    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                            elif graph_type == "Density plot":
                                fig = plot_density(
                                    plot_data, 
                                    var, 
                                    title, 
                                    x_axis, 
                                    y_axis,
                                    COLOR_PALETTES[color_scheme]
                                )
                            # Ajouter d'autres types de graphiques ici
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Export du graphique
                            export_visualization(fig, 'graph', var_name=var, source=source, note=note, is_plotly=True)
                        except Exception as e:
                            st.error(f"Erreur lors de la génération du graphique: {str(e)}")
                else:
                    st.error("La variable sélectionnée ne contient pas de données numériques valides.")
                    create_qualitative_dashboard(plot_data, var)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse numérique: {str(e)}")
                create_qualitative_dashboard(plot_data, var)
                                   
                # Options de visualisation (à garder séparé des options de regroupement)
                if is_numeric:
                    st.write("### Options de visualisation")
                    
                    # Utiliser une clé différente pour ce selectbox pour éviter les doublons
                    graph_type_selection = st.selectbox(
                        "Type de graphique",
                        ["Histogramme", "Density plot", "Boîte à moustaches", "Violin plot"],
                        key="graph_type_selection"
                    )
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        color_scheme = st.selectbox("Palette de couleurs", list(COLOR_PALETTES.keys()), key="color_scheme")
                    
                    # Options avancées
                    with st.expander("Options avancées"):
                        title = st.text_input("Titre du graphique", f"Distribution de {var}", key="title_adv")
                        x_axis = st.text_input("Titre de l'axe X", var, key="x_axis_adv")
                        y_axis = st.text_input("Titre de l'axe Y", "Valeur", key="y_axis_adv")
                        source = st.text_input("Source des données", "", key="source_adv")
                        note = st.text_input("Note de lecture", "", key="note_adv")
                        show_values = st.checkbox("Afficher les valeurs", True, key="show_values_adv")
                    
                    if st.button("Générer la visualisation", key="generate_num"):
                        try:
                            if graph_type_selection == "Histogramme":
                                fig = px.histogram(
                                    plot_data, 
                                    title=title, 
                                    color_discrete_sequence=COLOR_PALETTES[color_scheme]
                                )
                                if show_values:
                                    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                            elif graph_type_selection == "Density plot":
                                fig = plot_density(
                                    plot_data, 
                                    var, 
                                    title, 
                                    x_axis, 
                                    y_axis,
                                    COLOR_PALETTES[color_scheme]
                                )
                            elif graph_type_selection == "Boîte à moustaches":
                                fig = px.box(
                                    plot_data,
                                    title=title,
                                    color_discrete_sequence=COLOR_PALETTES[color_scheme]
                                )
                            else:  # Violin plot
                                fig = px.violin(
                                    plot_data,
                                    title=title,
                                    color_discrete_sequence=COLOR_PALETTES[color_scheme],
                                    box=True
                                )
                        
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Bouton d'export
                            export_visualization(
                                fig, 
                                'graph', 
                                var_name=var, 
                                source=source, 
                                note=note, 
                                is_plotly=True
                            )
                            
                        except Exception as e:
                            st.error(f"Erreur lors de la génération du graphique : {str(e)}")
                                               
                    # Configuration de la visualisation pour les variables numériques
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if grouping_method == "Aucune":
                            graph_type = st.selectbox("Type de graphique", ["Histogramme", "Density plot"], key="graph_type_no_group")
                        elif grouping_method == "Quantile":
                            graph_type = st.selectbox(
                                "Type de graphique",
                                ["Boîte à moustaches", "Violin plot", "Box plot avec points"],
                                key="graph_type_quantile"
                            )
                        else:
                            graph_type = st.selectbox("Type de graphique", ["Bar plot", "Lollipop plot", "Treemap"], key="graph_type_group")
                        
                    with col2:
                        color_scheme = st.selectbox("Palette de couleurs", list(COLOR_PALETTES.keys()), key="color_scheme")
                    
                    # Options avancées
                    with st.expander("Options avancées"):
                        title = st.text_input("Titre du graphique", f"Distribution de {var}", key="title_adv")
                        x_axis = st.text_input("Titre de l'axe X", var, key="x_axis_adv")
                        y_axis = st.text_input("Titre de l'axe Y", "Valeur", key="y_axis_adv")
                        source = st.text_input("Source des données", "", key="source_adv")
                        note = st.text_input("Note de lecture", "", key="note_adv")
                        show_values = st.checkbox("Afficher les valeurs", True, key="show_values_adv")
                    
                    if st.button("Générer la visualisation", key="generate_num"):
                        try:
                            if grouping_method == "Aucune":
                                if graph_type == "Histogramme":
                                    fig = px.histogram(
                                        plot_data, 
                                        title=title, 
                                        color_discrete_sequence=COLOR_PALETTES[color_scheme]
                                    )
                                    if show_values:
                                        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                                else:  # Density plot
                                    fig = plot_density(
                                        plot_data, 
                                        var, 
                                        title, 
                                        x_axis, 
                                        y_axis,
                                        COLOR_PALETTES[color_scheme]
                                    )
                        
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Bouton d'export
                                export_visualization(
                                    fig, 
                                    'graph', 
                                    var_name=var, 
                                    source=source, 
                                    note=note, 
                                    is_plotly=True
                                )
                                
                            # Ajouter ici les autres types de graphiques pour les variables numériques
                            
                        except Exception as e:
                            st.error(f"Erreur lors de la génération du graphique : {str(e)}")
                                
                else:  # Pour les variables qualitatives
                    # ✅ Définir les variables de contrôle pour les non-réponses
                    exclude_missing = st.checkbox("Exclure les non-réponses", key="exclude_missing_checkbox")
                    missing_label = "Non réponse"
                    
                    if not exclude_missing:
                        missing_label = st.text_input(
                            "Libellé pour les non-réponses",
                            value="Non réponse",
                            key="missing_label_input"
                        )

                    # ✅ Appliquer le filtrage des non-réponses
                    filtered_data[var] = filtered_data[var].astype(str).str.strip()  # ✅ Nettoyage des valeurs
                    if exclude_missing:
                        filtered_data = filtered_data[
                            (filtered_data[var].notna()) &  # ✅ Supprime les NaN
                            (filtered_data[var] != "") &  # ✅ Supprime les cases vides
                            (filtered_data[var] != missing_label)  # ✅ Supprime les réponses définies comme "Non réponse"
                        ]

                    # ✅ Utiliser la nouvelle fonction améliorée pour le regroupement et l'analyse
                    value_counts = improved_qualitative_analysis(filtered_data[var], var)
                    
                    # ✅ Affichage du tableau
                    table_title = st.text_input("Titre du tableau", f"Distribution de {var}", key="table_title_simple")
                    table_source = st.text_input("Source", "", key="table_source_simple")
                    table_note = st.text_area("Note de lecture", "", key="table_note_simple")
                    
                    # ✅ Affichage du tableau avec titre
                    st.subheader(f"📊 {table_title}")
                    st.dataframe(value_counts, use_container_width=True)
                    
                    # ✅ Affichage de la source et la note
                    if table_source:
                        st.caption(f"Source : {table_source}")
                    if table_note:
                        st.caption(f"Note : {table_note}")
                    
                    # ✅ Export Excel simplifié
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        value_counts.to_excel(writer, sheet_name="Tableau", index=False)
                        writer.close()
                    
                    st.download_button(
                        label="📥 Télécharger le tableau en Excel",
                        data=buffer.getvalue(),
                        file_name=f"tableau_{var.replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="excel_download_simple"
                    )
                    
                    # ✅ Ajout d'une fonction d'export en image simplifiée
                    if st.button("🖼️ Créer une image du tableau", key="create_image_simple"):
                        # Fonction d'export simplifiée intégrée directement
                        fig, ax = plt.subplots(figsize=(10, len(value_counts) + 2))
                        ax.axis('off')
                        
                        # Préparation des données à afficher
                        cell_text = []
                        for _, row in value_counts.iterrows():
                            cell_text.append([str(row[col]) for col in value_counts.columns])
                        
                        # Création du tableau
                        table = ax.table(
                            cellText=cell_text,
                            colLabels=value_counts.columns,
                            loc='center',
                            cellLoc='center'
                        )
                        
                        # Stylisation du tableau
                        table.auto_set_font_size(False)
                        table.set_fontsize(12)
                        table.scale(1, 1.5)
                        
                        # Titre et notes
                        if table_title:
                            plt.title(table_title, fontsize=14, fontweight='bold')
                        
                        # Sauvegarder l'image
                        img_buffer = BytesIO()
                        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
                        plt.close()
                        img_buffer.seek(0)
                        
                        # Bouton de téléchargement
                        st.download_button(
                            label="🖼️ Télécharger l'image",
                            data=img_buffer,
                            file_name=f"tableau_{var.replace(' ', '_')}.png",
                            mime="image/png",
                            key="image_download_simple"
                        )
                    
                    # ✅ Préparation des données pour la visualisation
                    data_to_plot = value_counts.copy()

                    # Configuration de la visualisation
                    st.write("### Configuration de la visualisation")
                    viz_col1, viz_col2 = st.columns([1, 2])

                    with viz_col1:
                        graph_type = st.selectbox(
                            "Type de graphique",
                            ["Bar plot", "Horizontal Bar", "Dot Plot", "Lollipop plot", "Treemap", "Radar"],
                            key="graph_type_qual_viz"
                        )

                    with viz_col2:
                        color_scheme = st.selectbox(
                            "Palette de couleurs",
                            list(COLOR_PALETTES.keys()),
                            key="color_scheme_qual_viz"
                        )

                    with st.expander("Options avancées de visualisation"):
                        adv_col1, adv_col2 = st.columns(2)

                        with adv_col1:
                            viz_title = st.text_input(
                                "Titre du graphique", 
                                value=table_title,  # ✅ Utilisation directe de la variable
                                key="viz_title_simple"
                            )

                            # ✅ Définition de y_axis par défaut pour éviter l'erreur
                            y_axis = None 
                            
                            if graph_type not in ["Treemap", "Radar"]:
                                x_axis = st.text_input(
                                    "Titre de l'axe Y", 
                                    value=var,  # ✅ Utilisation directe du nom de variable
                                    key="x_axis_qual_simple"
                                )
                                y_axis = st.text_input(
                                    "Titre de l'axe X", 
                                    "Valeur", 
                                    key="y_axis_qual_simple"
                                )
                            
                            show_values = st.checkbox("Afficher les valeurs", True, key="show_values_qual_simple")

                        with adv_col2:
                            viz_source = st.text_input(
                                "Source", 
                                value=table_source,  # ✅ Utilisation directe de la variable
                                key="viz_source_simple"
                            )
                            viz_note = st.text_input(
                                "Note de lecture", 
                                value=table_note,  # ✅ Utilisation directe de la variable
                                key="viz_note_simple"
                            )
                            value_type = st.radio("Type de valeur à afficher", ["Effectif", "Taux (%)"], key="value_type_qual_simple")
                            width = st.slider("Largeur du graphique", min_value=600, max_value=1200, value=800, step=50, key="graph_width_simple")

                    # Génération du graphique
                    if st.button("Générer la visualisation", key="generate_qual_viz_simple"):
                        try:
                            # Préparation des données pour le graphique
                            # Renommer les colonnes pour correspondre à ce qu'attendent vos fonctions de graphique
                            data_for_viz = data_to_plot.copy()
                            
                            # S'assurer que les noms des colonnes correspondent à ce qu'attendent vos fonctions
                            if "Modalités" in data_for_viz.columns and "Modalité" not in data_for_viz.columns:
                                data_for_viz = data_for_viz.rename(columns={"Modalités": "Modalité"})
                            
                            # ✅ Calcul des taux si l'utilisateur sélectionne "Taux (%)"
                            if value_type == "Taux (%)":
                                if "Taux (%)" not in data_for_viz.columns:
                                    total = data_for_viz["Effectif"].sum()
                                    data_for_viz["Taux (%)"] = (data_for_viz["Effectif"] / total * 100).round(1)
                                y_axis = "Taux (%)" if y_axis == "Valeur" or y_axis is None else y_axis

                            # Création du graphique selon le type choisi
                            if graph_type == "Bar plot":
                                fig = plot_qualitative_bar(
                                    data_for_viz, 
                                    viz_title, 
                                    x_axis, 
                                    y_axis,
                                    COLOR_PALETTES[color_scheme], 
                                    show_values,
                                    source=viz_source, 
                                    note=viz_note,
                                    value_type=value_type
                                )

                            elif graph_type == "Horizontal Bar":
                                fig = plot_modern_horizontal_bars(
                                    data=data_for_viz,
                                    title=viz_title,
                                    x_label=x_axis,
                                    value_type=value_type,
                                    color_palette=COLOR_PALETTES[color_scheme],
                                    source=viz_source,
                                    note=viz_note
                                )

                            elif graph_type == "Dot Plot":
                                fig = plot_dotplot(
                                    data_for_viz, 
                                    viz_title, 
                                    x_axis, 
                                    y_axis,
                                    COLOR_PALETTES[color_scheme], 
                                    show_values,
                                    source=viz_source, 
                                    note=viz_note, 
                                    width=width,
                                    value_type=value_type
                                )

                            elif graph_type == "Lollipop plot":
                                fig = plot_qualitative_lollipop(
                                    data_for_viz, 
                                    viz_title, 
                                    x_axis, 
                                    y_axis,
                                    COLOR_PALETTES[color_scheme], 
                                    show_values,
                                    source=viz_source, 
                                    note=viz_note,
                                    value_type=value_type
                                )

                            elif graph_type == "Treemap":
                                fig = plot_qualitative_treemap(
                                    data_for_viz, 
                                    viz_title,
                                    COLOR_PALETTES[color_scheme],
                                    source=viz_source, 
                                    note=viz_note
                                )

                            elif graph_type == "Radar":
                                fig = plot_radar(
                                    data_for_viz, 
                                    viz_title,
                                    COLOR_PALETTES[color_scheme],
                                    source=viz_source, 
                                    note=viz_note,
                                    value_type=value_type
                                )

                            # Affichage du graphique
                            st.plotly_chart(fig, use_container_width=False, config=config)
                            
                            # Export du graphique
                            export_visualization(
                                fig, 
                                'graph', 
                                var_name=var, 
                                source=viz_source, 
                                note=viz_note, 
                                data_to_plot=data_for_viz,
                                is_plotly=True,
                                graph_type=graph_type.lower()
                            )

                        except Exception as e:
                            st.error(f"Erreur lors de la génération du graphique : {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())  # Affiche la trace complète pour débogage

    elif active_tab == "bivariate":
        # Code pour l'analyse bivariée (similaire à votre code existant)
        try:
            # Sélection des variables
            var_x = create_enhanced_variable_selector(filtered_data, "Sélectionnez la variable X")
            
            if var_x:
                # Filtrer les colonnes pour var_y en excluant var_x
                var_y = create_enhanced_variable_selector(
                    filtered_data.drop(columns=[var_x]), 
                    "Sélectionnez la variable Y"
                )
                
                if var_y:
                    # Détection des types de variables
                    is_x_numeric = is_numeric_column(filtered_data, var_x)
                    is_y_numeric = is_numeric_column(filtered_data, var_y)
                    
                    # Suite de l'analyse bivariée selon les types de variables...
                    # (Intégrer ici votre code existant pour l'analyse bivariée)
                    
        except Exception as e:
            st.error(f"Erreur dans l'analyse bivariée : {str(e)}")
    
    elif active_tab == "time_series":
        # Analyse de séries temporelles
        st.write("### Analyse de séries temporelles")
        
        # Détection automatique des colonnes de date/heure
        date_cols = [col for col in filtered_data.columns if 
                    pd.api.types.is_datetime64_dtype(filtered_data[col]) or 
                    any(dt_str in col.lower() for dt_str in ['date', 'time', 'jour', 'mois', 'année', 'annee', 'year', 'month', 'day'])]
        
        if not date_cols:
            st.warning("Aucune colonne de date/heure détectée automatiquement. "
                      "Veuillez sélectionner manuellement une colonne contenant des dates.")
            date_cols = filtered_data.columns.tolist()
        
        # Sélection des variables
        time_col = st.selectbox("Sélectionnez la variable de temps", date_cols)
        value_col = st.selectbox("Sélectionnez la variable à analyser", 
                                [col for col in filtered_data.columns if is_numeric_column(filtered_data, col)])
        
        # Option pour grouper par une variable
        use_groupby = st.checkbox("Grouper par une variable")
        group_col = None
        
        if use_groupby:
            group_col = st.selectbox("Sélectionnez la variable de groupement", 
                                    [col for col in filtered_data.columns if col not in [time_col, value_col]])
        
        # Sélection de la fréquence d'agrégation
        freq_options = {
            "Année": "A",
            "Trimestre": "Q",
            "Mois": "MS",
            "Semaine": "W",
            "Jour": "D"
        }
        freq_choice = st.selectbox("Fréquence d'agrégation", list(freq_options.keys()))
        freq = freq_options[freq_choice]
        
        # Sélection de la méthode d'agrégation
        agg_method = st.selectbox("Méthode d'agrégation", 
                                ["mean", "sum", "median", "min", "max"],
                                format_func=lambda x: {
                                    "mean": "Moyenne", 
                                    "sum": "Somme", 
                                    "median": "Médiane",
                                    "min": "Minimum",
                                    "max": "Maximum"
                                }[x])
        
        # Analyse
        if st.button("Analyser la série temporelle"):
            try:
                # Convertir la colonne de temps en datetime si nécessaire
                if not pd.api.types.is_datetime64_dtype(filtered_data[time_col]):
                    filtered_data[time_col] = pd.to_datetime(filtered_data[time_col], errors='coerce')
                
                # Vérifier si la conversion a réussi
                if filtered_data[time_col].isna().all():
                    st.error(f"Impossible de convertir la colonne {time_col} en format date/heure.")
                else:
                    # Créer le graphique
                    title = f"Évolution de {value_col} par {freq_choice.lower()}"
                    if group_col:
                        title += f" groupé par {group_col}"
                    
                    fig = plot_time_series(
                        filtered_data, 
                        time_col, 
                        value_col, 
                        group_col=group_col,
                        agg_method=agg_method,
                        title=title,
                        color_palette=COLOR_PALETTES["Bleu"]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Analyse de la série temporelle
                    resampled_df, stats_df, additional_df = analyze_time_series(
                        filtered_data, 
                        time_col, 
                        value_col, 
                        group_col=group_col,
                        freq=freq
                    )
                    
                    # Affichage des statistiques
                    st.write("### Statistiques de la série temporelle")
                    
                    if group_col is None:
                        # Affichage pour une série simple
                        st.write("#### Statistiques globales")
                        st.dataframe(stats_df, use_container_width=True)
                        
                        st.write("#### Autocorrélation")
                        if len(additional_df.columns) > 1:  # Vérifier qu'il ne s'agit pas du DataFrame d'erreur
                            # Créer un graphique d'autocorrélation
                            fig_acf = px.bar(
                                additional_df, 
                                x='Lag', 
                                y='Autocorrélation',
                                title="Fonction d'autocorrélation"
                            )
                            st.plotly_chart(fig_acf, use_container_width=True)
                        else:
                            st.write(additional_df)
                    else:
                        # Affichage pour des séries groupées
                        st.write("#### Statistiques par groupe")
                        st.dataframe(stats_df, use_container_width=True)
                        
                        st.write("#### Matrice de corrélation entre groupes")
                        # Heatmap de la matrice de corrélation
                        fig_corr = px.imshow(
                            additional_df,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            title="Corrélation entre les groupes"
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Exportation des données
                    st.write("### Exportation des données")
                    
                    # Téléchargement des données réagrégées
                    buffer = BytesIO()
                    resampled_df.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Télécharger les données agrégées (Excel)",
                        data=buffer,
                        file_name=f"serie_temporelle_{value_col}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Bouton de création d'indicateur
                    if st.button("Créer un indicateur à partir de cette série temporelle"):
                        variables_info = {
                            'time_col': time_col,
                            'value_col': value_col,
                            'group_col': group_col
                        }
                        
                        # Convertir les statistiques en format adapté
                        if group_col is None:
                            stats_dict = stats_df.set_index('Statistique')['Valeur'].to_dict()
                        else:
                            stats_dict = stats_df.to_dict('index')
                        
                        show_indicator_form(stats_dict, 'time_series', variables_info)
                        
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de la série temporelle : {str(e)}")
    
    elif active_tab == "exploration":
        st.write("### Exploration et filtrage des données")
        
        # Affichage des données filtrées avec pagination
        st.write("#### Aperçu des données")
        
        # Options d'affichage
        col1, col2 = st.columns([1, 2])
        
        with col1:
            page_size = st.selectbox("Nombre de lignes par page", [10, 20, 50, 100], index=1)
        
        with col2:
            column_selection = st.multiselect(
                "Sélectionner les colonnes à afficher",
                options=filtered_data.columns.tolist(),
                default=filtered_data.columns.tolist()[:5]  # Par défaut, afficher les 5 premières colonnes
            )
        
        # Pagination
        total_rows = len(filtered_data)
        total_pages = (total_rows + page_size - 1) // page_size
        
        if total_pages > 0:
            page_number = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1
            )
            
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            
            # Afficher la plage actuellement affichée
            st.write(f"Affichage des lignes {start_idx + 1} à {end_idx} sur {total_rows}")
            
            # Afficher le dataframe avec les colonnes sélectionnées
            if column_selection:
                st.dataframe(
                    filtered_data.loc[start_idx:end_idx-1, column_selection].reset_index(drop=True),
                    use_container_width=True
                )
            else:
                st.warning("Veuillez sélectionner au moins une colonne à afficher.")
        else:
            st.warning("Aucune donnée à afficher.")
        
        # Export des données filtrées
        st.write("#### Export des données")
        export_format = st.radio("Format d'export", ["Excel (.xlsx)", "CSV", "JSON"])
        
        if st.button("Exporter les données filtrées"):
            buffer = BytesIO()
            
            if export_format == "Excel (.xlsx)":
                filtered_data.to_excel(buffer, index=False, engine='openpyxl')
                file_name = "donnees_filtrees.xlsx"
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif export_format == "CSV":
                filtered_data.to_csv(buffer, index=False)
                file_name = "donnees_filtrees.csv"
                mime = "text/csv"
            else:  # JSON
                buffer.write(filtered_data.to_json(orient="records").encode())
                file_name = "donnees_filtrees.json"
                mime = "application/json"
            
            buffer.seek(0)
            
            st.download_button(
                label=f"📥 Télécharger ({export_format})",
                data=buffer,
                file_name=file_name,
                mime=mime
            )
        
        # Analyse exploratoire supplémentaire
        st.write("#### Analyse exploratoire")
        
        # Sélection des variables pour l'analyse exploratoire
        exploration_type = st.selectbox(
            "Type d'analyse exploratoire",
            ["Distribution des valeurs", "Analyse des valeurs manquantes", "Détection d'anomalies"]
        )
        
        if exploration_type == "Distribution des valeurs":
            # Sélection d'une variable
            var_to_explore = st.selectbox(
                "Sélectionnez une variable à explorer",
                filtered_data.columns.tolist()
            )
            
            # Analyser la distribution
            if is_numeric_column(filtered_data, var_to_explore):
                # Pour les variables numériques
                fig = px.histogram(
                    filtered_data, 
                    x=var_to_explore, 
                    marginal="box",
                    histnorm='probability density',
                    title=f"Distribution de {var_to_explore}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques supplémentaires
                stats = filtered_data[var_to_explore].describe()
                st.write("Statistiques descriptives:")
                st.write(stats)
                
                # Test de normalité
                is_normal = check_normality(filtered_data, var_to_explore)
                st.write(f"Test de normalité: {'Distribution normale' if is_normal else 'Distribution non normale'}")
                
            else:
                # Pour les variables catégorielles
                value_counts = filtered_data[var_to_explore].value_counts().reset_index()
                value_counts.columns = [var_to_explore, 'Fréquence']
                
                fig = px.bar(
                    value_counts, 
                    x=var_to_explore, 
                    y='Fréquence',
                    title=f"Distribution de {var_to_explore}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Afficher les valeurs les plus fréquentes
                st.write("Valeurs les plus fréquentes:")
                st.write(value_counts.head(10))
        
        elif exploration_type == "Analyse des valeurs manquantes":
            # Calculer les valeurs manquantes par colonne
            missing_data = pd.DataFrame({
                'Variable': filtered_data.columns,
                'Valeurs manquantes': filtered_data.isna().sum().values,
                'Pourcentage': (filtered_data.isna().sum() / len(filtered_data) * 100).values
            })
            
            missing_data = missing_data.sort_values('Pourcentage', ascending=False)
            
            # Afficher le tableau des valeurs manquantes
            st.write("#### Analyse des valeurs manquantes par variable")
            st.dataframe(missing_data)
            
            # Créer un graphique des valeurs manquantes
            fig = px.bar(
                missing_data, 
                x='Variable', 
                y='Pourcentage',
                title="Pourcentage de valeurs manquantes par variable",
                color='Pourcentage',
                color_continuous_scale=px.colors.sequential.Blues_r
            )
            
            fig.update_layout(
                xaxis_title="Variables",
                yaxis_title="Pourcentage de valeurs manquantes",
                xaxis={'categoryorder':'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations pour la gestion des valeurs manquantes
            st.write("#### Recommandations pour la gestion des valeurs manquantes")
            
            high_missing = missing_data[missing_data['Pourcentage'] > 50]
            moderate_missing = missing_data[(missing_data['Pourcentage'] <= 50) & (missing_data['Pourcentage'] > 10)]
            low_missing = missing_data[(missing_data['Pourcentage'] <= 10) & (missing_data['Pourcentage'] > 0)]
            
            if not high_missing.empty:
                st.warning("⚠️ Variables avec plus de 50% de valeurs manquantes - Envisager de les supprimer:")
                st.write(", ".join(high_missing['Variable'].tolist()))
            
            if not moderate_missing.empty:
                st.info("ℹ️ Variables avec 10-50% de valeurs manquantes - Envisager l'imputation:")
                st.write(", ".join(moderate_missing['Variable'].tolist()))
            
            if not low_missing.empty:
                st.success("✅ Variables avec moins de 10% de valeurs manquantes - Peut être facilement imputé:")
                st.write(", ".join(low_missing['Variable'].tolist()))
        
        elif exploration_type == "Détection d'anomalies":
            # Sélection d'une variable numérique
            numeric_cols = [col for col in filtered_data.columns if is_numeric_column(filtered_data, col)]
            
            if not numeric_cols:
                st.warning("Aucune variable numérique disponible pour la détection d'anomalies.")
            else:
                var_to_analyze = st.selectbox(
                    "Sélectionnez une variable numérique à analyser",
                    numeric_cols
                )
                
                # Méthode de détection d'anomalies
                detection_method = st.selectbox(
                    "Méthode de détection d'anomalies",
                    ["Z-score", "IQR (écart interquartile)"]
                )
                
                # Seuil de détection
                if detection_method == "Z-score":
                    threshold = st.slider("Seuil Z-score", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
                else:  # IQR
                    threshold = st.slider("Facteur IQR", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
                
                # Analyser les anomalies
                if st.button("Détecter les anomalies"):
                    # Nettoyer les données
                    clean_data = filtered_data[var_to_analyze].dropna()
                    
                    if detection_method == "Z-score":
                        # Méthode Z-score
                        mean = clean_data.mean()
                        std = clean_data.std()
                        z_scores = abs((clean_data - mean) / std)
                        anomalies = clean_data[z_scores > threshold]
                        anomaly_indices = z_scores[z_scores > threshold].index
                    else:
                        # Méthode IQR
                        q1 = clean_data.quantile(0.25)
                        q3 = clean_data.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        anomalies = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                        anomaly_indices = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)].index
                    
                    # Afficher les résultats
                    st.write(f"#### Résultats de la détection d'anomalies pour {var_to_analyze}")
                    st.write(f"Nombre d'anomalies détectées: {len(anomalies)}")
                    
                    if not anomalies.empty:
                        # Créer un graphique des anomalies
                        fig = go.Figure()
                        
                        # Ajouter les points normaux
                        fig.add_trace(go.Scatter(
                            x=list(range(len(clean_data))),
                            y=clean_data.values,
                            mode='markers',
                            name='Normaux',
                            marker=dict(
                                color='blue',
                                size=6
                            )
                        ))
                        
                        # Ajouter les anomalies
                        fig.add_trace(go.Scatter(
                            x=[clean_data.index.get_loc(idx) for idx in anomaly_indices],
                            y=anomalies.values,
                            mode='markers',
                            name='Anomalies',
                            marker=dict(
                                color='red',
                                size=10,
                                symbol='circle-open',
                                line=dict(width=2)
                            )
                        ))
                        
                        fig.update_layout(
                            title=f"Détection d'anomalies pour {var_to_analyze}",
                            xaxis_title="Index",
                            yaxis_title=var_to_analyze,
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Afficher les valeurs anomales
                        st.write("Valeurs anomales détectées:")
                        anomaly_df = filtered_data.loc[anomaly_indices].copy()
                        anomaly_df["Valeur"] = anomalies.values
                        
                        if detection_method == "Z-score":
                            anomaly_df["Z-score"] = z_scores[anomaly_indices].values
                        else:
                            anomaly_df["Distance IQR"] = [
                                abs(val - q1) / iqr if val < q1 else abs(val - q3) / iqr 
                                for val in anomalies.values
                            ]
                        
                        st.dataframe(anomaly_df)
                    else:
                        st.success("Aucune anomalie détectée avec les paramètres actuels.")
    
    elif active_tab == "export":
        st.write("### Export des résultats")
        
        # Option d'exportation globale
        st.write("#### Export des données")
        
        # Sélection des colonnes à exporter
        all_columns = filtered_data.columns.tolist()
        selected_columns = st.multiselect(
            "Sélectionnez les colonnes à exporter",
            options=all_columns,
            default=all_columns
        )
        
        # Format d'export
        export_format = st.radio(
            "Format d'export",
            ["Excel (.xlsx)", "CSV", "JSON"]
        )
        
        # Bouton d'export
        if st.button("Exporter les données"):
            if selected_columns:
                export_data = filtered_data[selected_columns].copy()
                
                buffer = BytesIO()
                
                if export_format == "Excel (.xlsx)":
                    export_data.to_excel(buffer, index=False, engine='openpyxl')
                    file_name = "donnees_exportees.xlsx"
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif export_format == "CSV":
                    export_data.to_csv(buffer, index=False)
                    file_name = "donnees_exportees.csv"
                    mime = "text/csv"
                else:  # JSON
                    buffer.write(export_data.to_json(orient="records").encode())
                    file_name = "donnees_exportees.json"
                    mime = "application/json"
                
                buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Télécharger les données ({export_format})",
                    data=buffer,
                    file_name=file_name,
                    mime=mime
                )
            else:
                st.warning("Veuillez sélectionner au moins une colonne à exporter.")
        
        # Création de rapport
        st.write("#### Générer un rapport d'analyse")
        
        # Options du rapport
        include_data_summary = st.checkbox("Inclure le résumé des données", True)
        include_data_preview = st.checkbox("Inclure un aperçu des données", True)
        include_missing_analysis = st.checkbox("Inclure l'analyse des valeurs manquantes", True)
        
        # Sélection des variables pour les visualisations
        st.write("Sélectionnez jusqu'à 3 variables pour les visualisations:")
        var_selections = []
        for i in range(3):
            var = st.selectbox(
                f"Variable {i+1}",
                options=["---"] + all_columns,
                key=f"report_var_{i}"
            )
            if var != "---":
                var_selections.append(var)
        
        # Format du rapport
        report_format = st.radio(
            "Format du rapport",
            ["HTML", "PDF"]
        )
        
        # Bouton de génération
        if st.button("Générer le rapport"):
            try:
                # Créer un buffer pour enregistrer le rapport
                buffer = BytesIO()
                
                if report_format == "HTML":
                    # Générer le rapport HTML
                    html_content = "<html><head><title>Rapport d'analyse ESR</title></head><body>"
                    
                    # En-tête
                    html_content += "<h1>Rapport d'analyse des données ESR</h1>"
                    html_content += f"<p>Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>"
                    
                    # Résumé des données
                    if include_data_summary:
                        html_content += "<h2>Résumé des données</h2>"
                        html_content += f"<p>Nombre de lignes: {len(filtered_data):,}</p>"
                        html_content += f"<p>Nombre de colonnes: {len(filtered_data.columns):,}</p>"
                        
                        # Statistiques des variables numériques
                        numeric_cols = [col for col in filtered_data.columns if is_numeric_column(filtered_data, col)]
                        if numeric_cols:
                            html_content += "<h3>Statistiques des variables numériques</h3>"
                            numeric_df = filtered_data[numeric_cols].describe().round(2)
                            html_content += numeric_df.to_html(classes="table table-striped")
                    
                    # Aperçu des données
                    if include_data_preview:
                        html_content += "<h2>Aperçu des données</h2>"
                        html_content += filtered_data.head(10).to_html(classes="table table-striped")
                    
                    # Analyse des valeurs manquantes
                    if include_missing_analysis:
                        html_content += "<h2>Analyse des valeurs manquantes</h2>"
                        missing_data = pd.DataFrame({
                            'Variable': filtered_data.columns,
                            'Valeurs manquantes': filtered_data.isna().sum().values,
                            'Pourcentage': (filtered_data.isna().sum() / len(filtered_data) * 100).round(2).values
                        })
                        html_content += missing_data.to_html(classes="table table-striped")
                    
                    # Visualisations
                    if var_selections:
                        html_content += "<h2>Visualisations</h2>"
                        
                        for var in var_selections:
                            if is_numeric_column(filtered_data, var):
                                # Pour les variables numériques, intégrer une image d'histogramme
                                fig = px.histogram(
                                    filtered_data, 
                                    x=var,
                                    title=f"Distribution de {var}"
                                )
                                img_bytes = fig.to_image(format="png")
                                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                                html_content += f"<h3>Distribution de {var}</h3>"
                                html_content += f'<img src="data:image/png;base64,{img_b64}" alt="{var}" width="100%">'
                            else:
                                # Pour les variables catégorielles, intégrer un tableau de fréquences
                                html_content += f"<h3>Distribution de {var}</h3>"
                                value_counts = filtered_data[var].value_counts().head(10)
                                value_counts_df = pd.DataFrame({
                                    var: value_counts.index,
                                    'Fréquence': value_counts.values,
                                    'Pourcentage': (value_counts.values / len(filtered_data) * 100).round(2)
                                })
                                html_content += value_counts_df.to_html(classes="table table-striped")
                    
                    # Fermeture du document HTML
                    html_content += "</body></html>"
                    
                    # Écrire dans le buffer
                    buffer.write(html_content.encode())
                    buffer.seek(0)
                    
                    # Télécharger le rapport
                    st.download_button(
                        label="📄 Télécharger le rapport HTML",
                        data=buffer,
                        file_name="rapport_esr.html",
                        mime="text/html"
                    )
                
                elif report_format == "PDF":
                    # Pour le PDF, afficher simplement un message d'information
                    # car la génération de PDF nécessite des librairies supplémentaires
                    st.info("La génération de PDF nécessite l'installation des librairies 'weasyprint' ou 'reportlab'."
                           "Veuillez choisir le format HTML pour l'instant.")
            
            except Exception as e:
                st.error(f"Erreur lors de la génération du rapport : {str(e)}")

# Exécution de l'application
if __name__ == "__main__":
    main()
