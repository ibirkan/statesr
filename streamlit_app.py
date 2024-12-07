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
import numpy as np
from scipy import stats
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
        
def merge_multiple_tables(dataframes, merge_configs):
    """Merge multiple dataframes based on the provided configurations."""
    merged_df = dataframes[0]
    for i in range(1, len(dataframes)):
        merge_config = merge_configs[i - 1]
        # Print data types for debugging
        print(f"Merging on columns: {merge_config['left']} and {merge_config['right']}")
        print(f"Data types before merge: {merged_df[merge_config['left']].dtype}, {dataframes[i][merge_config['right']].dtype}")
        # Convert data types if necessary
        if merged_df[merge_config['left']].dtype != dataframes[i][merge_config['right']].dtype:
            if pd.api.types.is_numeric_dtype(merged_df[merge_config['left']]) and pd.api.types.is_numeric_dtype(dataframes[i][merge_config['right']]):
                merged_df[merge_config['left']] = merged_df[merge_config['left']].astype(float)
                dataframes[i][merge_config['right']] = dataframes[i][merge_config['right']].astype(float)
            else:
                merged_df[merge_config['left']] = merged_df[merge_config['left']].astype(str)
                dataframes[i][merge_config['right']] = dataframes[i][merge_config['right']].astype(str)
        merged_df = merged_df.merge(dataframes[i], left_on=merge_config['left'], right_on=merge_config['right'], how='outer')
        # Print data types after merge for debugging
        print(f"Data types after merge: {merged_df[merge_config['left']].dtype}, {merged_df[merge_config['right']].dtype}")
    return merged_df
    
def is_numeric_column(df, column):
    """Vérifie si une colonne est numérique."""
    return pd.api.types.is_numeric_dtype(df[column])

def analyze_qualitative_bivariate(df, var_x, var_y, exclude_missing=True):
    """
    Analyse bivariée pour deux variables qualitatives.
    Parameters:
        df: DataFrame source
        var_x: Variable en ligne
        var_y: Variable en colonne
        exclude_missing: Si True, exclut les non-réponses
    """
    # Copie du DataFrame pour éviter les modifications sur l'original
    data = df.copy()
    
    # Liste des valeurs considérées comme non-réponses
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    
    # Remplacement des non-réponses par np.nan pour faciliter le filtrage
    data[var_x] = data[var_x].replace(missing_values, np.nan)
    data[var_y] = data[var_y].replace(missing_values, np.nan)
    
    # Filtrage des non-réponses
    if exclude_missing:
        data = data.dropna(subset=[var_x, var_y])
        
        # Calcul du taux de réponse
        response_rate_x = (df[var_x].notna().sum() / len(df)) * 100
        response_rate_y = (df[var_y].notna().sum() / len(df)) * 100
        
        response_stats = {
            f"{var_x}": f"{response_rate_x:.1f}%",
            f"{var_y}": f"{response_rate_y:.1f}%"
        }
    
    # Création du tableau croisé avec effectifs
    crosstab_n = pd.crosstab(data[var_x], data[var_y])
    
    # Calcul des pourcentages en ligne
    crosstab_pct = pd.crosstab(data[var_x], data[var_y], normalize='index') * 100
    
    # Calcul des moyennes par colonne (pour le total)
    col_means = crosstab_pct.mean()
    
    # Création du tableau combiné
    combined_table = pd.DataFrame(index=crosstab_n.index, columns=crosstab_n.columns)
    
    # Remplissage du tableau principal
    for idx in crosstab_n.index:
        for col in crosstab_n.columns:
            n = crosstab_n.loc[idx, col]
            pct = crosstab_pct.loc[idx, col]
            combined_table.loc[idx, col] = f"{pct:.1f}% ({n})"
    
    # Ajout des totaux
    row_totals = crosstab_n.sum(axis=1)
    combined_table['Total'] = [f"100% ({n})" for n in row_totals]
    
    # Ajout de la ligne des moyennes
    mean_row = []
    for col in crosstab_n.columns:
        mean_val = col_means[col]
        total_n = crosstab_n[col].sum()
        mean_row.append(f"{mean_val:.1f}% ({total_n})")
    mean_row.append(f"100% ({crosstab_n.values.sum()})")
    
    combined_table.loc['Moyenne'] = mean_row
    
    if exclude_missing:
        return combined_table, response_stats
    return combined_table

def plot_qualitative_bivariate(df, var_x, var_y, plot_type, color_palette, plot_options):
    """
    Création des visualisations pour l'analyse bivariée qualitative.
    """
    # Copie du DataFrame pour éviter les modifications sur l'original
    data = df.copy()
    
    # Liste des valeurs considérées comme non-réponses
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    
    # Remplacement des non-réponses par np.nan
    data[var_x] = data[var_x].replace(missing_values, np.nan)
    data[var_y] = data[var_y].replace(missing_values, np.nan)
    
    # Filtrage des non-réponses
    data = data.dropna(subset=[var_x, var_y])
    
    # Données de base pour les graphiques
    crosstab_n = pd.crosstab(data[var_x], data[var_y])
    
    if plot_type == "Grouped Bar Chart":
        fig, ax = plt.subplots(figsize=(12, 6))
        crosstab_n.plot(kind='bar', ax=ax, color=color_palette)
        plt.title(plot_options['title'])
        plt.xlabel(plot_options['x_label'])
        plt.ylabel(plot_options['y_label'])
        plt.legend(title=var_y, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        
    elif plot_type == "Stacked Bar Chart":
        fig, ax = plt.subplots(figsize=(12, 6))
        crosstab_pct = pd.crosstab(data[var_x], data[var_y], normalize='index') * 100
        crosstab_pct.plot(kind='bar', stacked=True, ax=ax, color=color_palette)
        plt.title(plot_options['title'])
        plt.xlabel(plot_options['x_label'])
        plt.ylabel("Pourcentage")
        plt.legend(title=var_y, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        
    elif plot_type == "Mosaic Plot":
        # Création d'une figure plus grande pour accommoder les labels
        fig, ax = plt.subplots(figsize=(14, 8))
        data_norm = crosstab_n.div(crosstab_n.sum().sum())
        
        # Calcul des positions pour les rectangles
        widths = data_norm.sum(axis=1)
        x = 0
        x_centers = []  # Pour stocker les centres des rectangles en x
        
        # Premier passage pour créer les rectangles
        for i, (idx, row) in enumerate(data_norm.iterrows()):
            y = 0
            width = widths[idx]
            x_centers.append(x + width/2)  # Stocker le centre pour le label
            
            for j, val in enumerate(row):
                height = val / widths[idx]
                rect = plt.Rectangle((x, y), width, height, 
                                   facecolor=color_palette[j % len(color_palette)],
                                   edgecolor='white',
                                   linewidth=1)
                ax.add_patch(rect)
                
                # Ajout des pourcentages si assez d'espace
                if plot_options['show_values'] and width * height > 0.02:
                    plt.text(x + width/2, y + height/2, 
                            f'{val*100:.1f}%',
                            ha='center', va='center',
                            fontsize=9)
                y += height
            x += width
        
        # Ajout des labels pour var_x sous les rectangles
        ax.set_xticks(x_centers)
        ax.set_xticklabels(crosstab_n.index, rotation=45, ha='right')
        
        # Ajout d'une légende pour var_y
        legend_elements = [plt.Rectangle((0,0), 1, 1, 
                                       facecolor=color_palette[i % len(color_palette)])
                         for i in range(len(crosstab_n.columns))]
        ax.legend(legend_elements, crosstab_n.columns, 
                 title=var_y, bbox_to_anchor=(1.05, 1), 
                 loc='upper left')
        
        # Ajustement des limites et des titres
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1)
        plt.title(plot_options['title'])
        plt.xlabel(plot_options['x_label'])
        
        # Suppression des graduations de l'axe y mais conservation de l'axe
        ax.set_yticks([])
        
        # Ajout de la source et de la note si spécifiées
        if plot_options['source']:
            plt.figtext(0.01, -0.1, f"Source : {plot_options['source']}", 
                       ha='left', fontsize=8)
        
        if plot_options['note']:
            plt.figtext(0.01, -0.15, f"Note : {plot_options['note']}", 
                       ha='left', fontsize=8)
        
        plt.tight_layout()
    
    return fig

def analyze_mixed_bivariate(df, qual_var, quant_var):
    """
    Analyse bivariée pour une variable qualitative et une quantitative.
    Retourne les statistiques descriptives par modalité.
    """
    # Filtrage des non-réponses
    data = df.copy()
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    data[qual_var] = data[qual_var].replace(missing_values, np.nan)
    data[quant_var] = data[quant_var].replace(missing_values, np.nan)
    data = data.dropna(subset=[qual_var, quant_var])
    
    # Calcul des statistiques par modalité
    stats_df = data.groupby(qual_var)[quant_var].agg([
        ('Effectif', 'count'),
        ('Total', lambda x: x.sum()),
        ('Moyenne', 'mean'),
        ('Médiane', 'median'),
        ('Écart-type', 'std'),
        ('Minimum', 'min'),
        ('Maximum', 'max')
    ]).round(2)
    
    # Ajout du total général
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
    
    # Calcul du taux de réponse
    response_rate = (data[qual_var].count() / len(df)) * 100
    
    return stats_df, response_rate

def plot_mixed_bivariate(df, qual_var, quant_var, color_palette, plot_options):
    """
    Création d'un box plot pour l'analyse mixte avec Plotly.
    """
    # Filtrage des non-réponses
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
    
    # Ajout de la source et de la note si spécifiées
    if plot_options['source']:
        fig.add_annotation(
            text=f"Source : {plot_options['source']}",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.15,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
    
    if plot_options['note']:
        fig.add_annotation(
            text=f"Note : {plot_options['note']}",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.2,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
    
    return fig

def check_normality(data, var):
    """
    Vérifie la normalité d'une variable avec adaptation pour les grands échantillons.
    """
    n = len(data)
    if n > 5000:
        # Pour les grands échantillons, utiliser le test d'Anderson-Darling
        # qui est plus adapté aux grands échantillons
        _, p_value = stats.normaltest(data[var])
    else:
        # Pour les petits échantillons, utiliser Shapiro-Wilk
        _, p_value = stats.shapiro(data[var])
    return p_value > 0.05

def check_duplicates(df, var_x, var_y):
    """
    Vérifie si certaines valeurs de var_x ou var_y sont répétées
    dans le jeu de données.
    """
    duplicates_x = df[var_x].duplicated().any()
    duplicates_y = df[var_y].duplicated().any()
    return duplicates_x or duplicates_y

def analyze_quantitative_bivariate(df, var_x, var_y, groupby_col=None, agg_method='sum'):
    """
    Analyse bivariée pour deux variables quantitatives avec option d'agrégation.
    """
    data = df.copy()
    missing_values = [None, np.nan, '', 'nan', 'NaN', 'Non réponse', 'NA', 'nr', 'NR', 'Non-réponse']
    data[var_x] = data[var_x].replace(missing_values, np.nan)
    data[var_y] = data[var_y].replace(missing_values, np.nan)
    
    # Si une colonne de groupement est spécifiée, agrégeons d'abord les données
    if groupby_col is not None:
        data = data.groupby(groupby_col).agg({
            var_x: agg_method,
            var_y: agg_method
        }).reset_index()
    
    # Suppression des valeurs manquantes
    data = data.dropna(subset=[var_x, var_y])
    
    # Test de normalité
    is_normal_x = check_normality(data, var_x)
    is_normal_y = check_normality(data, var_y)
    is_normal = is_normal_x and is_normal_y
    
    if is_normal:
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
    
    # Calcul des taux de réponse sur données originales
    response_rate_x = (df[var_x].count() / len(df)) * 100
    response_rate_y = (df[var_y].count() / len(df)) * 100
    
    return results_df, response_rate_x, response_rate_y
    
def detect_variable_to_aggregate(df, var_x, var_y, groupby_col):
    """
    Détecte automatiquement quelle variable doit être agrégée en comptant
    le nombre de valeurs uniques pour chaque modalité du groupby_col.
    
    Returns:
        tuple: (var_to_aggregate, var_to_keep_raw)
    """
    # Pour chaque variable, on compte le nombre de valeurs différentes par groupby_col
    x_values_per_group = df.groupby(groupby_col)[var_x].nunique()
    y_values_per_group = df.groupby(groupby_col)[var_y].nunique()
    
    # Si une variable a plusieurs valeurs pour certains groupes, elle doit être agrégée
    x_has_duplicates = (x_values_per_group > 1).any()
    y_has_duplicates = (y_values_per_group > 1).any()
    
    if x_has_duplicates and not y_has_duplicates:
        return var_x, var_y
    elif y_has_duplicates and not x_has_duplicates:
        return var_y, var_x
    elif x_has_duplicates and y_has_duplicates:
        # Si les deux variables ont des doublons, on regarde laquelle en a le plus
        x_total_duplicates = x_values_per_group.sum()
        y_total_duplicates = y_values_per_group.sum()
        return (var_x, var_y) if x_total_duplicates > y_total_duplicates else (var_y, var_x)
    else:
        # Si aucune variable n'a de doublons, on peut garder l'ordre original
        return None, None

def detect_repeated_variable(df, var_x, var_y, groupby_col):
    """
    Détecte quelle variable contient des observations répétées pour chaque modalité du groupby_col.
    """
    x_duplicates = df.groupby(groupby_col)[var_x].nunique() > 1
    y_duplicates = df.groupby(groupby_col)[var_y].nunique() > 1
    
    if x_duplicates.any() and not y_duplicates.any():
        return var_x
    elif y_duplicates.any() and not x_duplicates.any():
        return var_y
    return None

def plot_quantitative_bivariate_interactive(df, var_x, var_y, color_scheme, plot_options, groupby_col=None, agg_method=None):
    """
    Création d'un scatter plot interactif avec Plotly pour l'analyse quantitative.
    Inclut des info-bulles et une ligne de régression robuste.
    """
    # Nettoyage des données pour la régression
    df_clean = df.dropna(subset=[var_x, var_y])
    x = df_clean[var_x].values
    y = df_clean[var_y].values
    
    # Calcul de la régression de manière robuste
    regression_coeffs, regression_success = calculate_regression(x, y)
    
    # Création du scatter plot
    fig = go.Figure()
    
    # Ajout du nuage de points avec info-bulles personnalisées
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
    
    # Ajout du nuage de points
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
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Arial"
        )
    ))
    
    # Ajout de la ligne de régression si le calcul a réussi
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
    else:
        st.warning("La ligne de régression n'a pas pu être calculée en raison de la distribution des données.")
    
    # Configuration du layout
    title = plot_options['title']
    if groupby_col and agg_method:
        title += f"<br><sup>Données agrégées par {groupby_col} ({agg_method})</sup>"
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
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
    
    # Ajout de la grille
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Ajout de la source et de la note si spécifiées
    annotations = []
    
    current_y = -0.15
    if plot_options['source']:
        annotations.append(dict(
            text=f"Source : {plot_options['source']}",
            x=0,
            y=current_y,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10)
        ))
        current_y -= 0.05
    
    if plot_options['note']:
        annotations.append(dict(
            text=f"Note : {plot_options['note']}",
            x=0,
            y=current_y,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10)
        ))
    
    if annotations:
        fig.update_layout(annotations=annotations)
    
    return fig
    
def calculate_regression(x, y):
    """
    Calcule la régression linéaire de manière robuste.
    Retourne les coefficients et un booléen indiquant si la régression a réussi.
    """
    try:
        # Première tentative avec numpy polyfit
        z = np.polyfit(x, y, 1)
        return z, True
    except np.linalg.LinAlgError:
        try:
            # Deuxième tentative avec statsmodels (plus robuste)
            import statsmodels.api as sm
            X = sm.add_constant(x)
            model = sm.OLS(y, X)
            results = model.fit()
            return [results.params[1], results.params[0]], True
        except:
            # Si les deux méthodes échouent
            return None, False

def plot_quantitative_bivariate_interactive(df, var_x, var_y, color_scheme, plot_options, groupby_col=None, agg_method=None):
    """
    Création d'un scatter plot interactif avec Plotly pour l'analyse quantitative.
    Inclut des info-bulles et une ligne de régression robuste.
    """
    # Nettoyage des données pour la régression
    df_clean = df.dropna(subset=[var_x, var_y])
    x = df_clean[var_x].values
    y = df_clean[var_y].values
    
    # Calcul de la régression de manière robuste
    regression_coeffs, regression_success = calculate_regression(x, y)
    
    # Création du scatter plot
    fig = go.Figure()
    
    # Ajout du nuage de points avec info-bulles personnalisées
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
    
    # Ajout du nuage de points
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
    
    # Ajout de la ligne de régression si le calcul a réussi
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
    else:
        st.warning("La ligne de régression n'a pas pu être calculée en raison de la distribution des données.")
    
    # Configuration du layout
    title = plot_options['title']
    if groupby_col and agg_method:
        title += f"<br><sup>Données agrégées par {groupby_col} ({agg_method})</sup>"
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
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
    
    # Ajout de la grille
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Ajout de la source et de la note si spécifiées
    annotations = []
    
    current_y = -0.15
    if plot_options['source']:
        annotations.append(dict(
            text=f"Source : {plot_options['source']}",
            x=0,
            y=current_y,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10)
        ))
        current_y -= 0.05
    
    if plot_options['note']:
        annotations.append(dict(
            text=f"Note : {plot_options['note']}",
            x=0,
            y=current_y,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=10)
        ))
    
    if annotations:
        fig.update_layout(annotations=annotations)
    
    return fig

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
        # Sélection de la variable avec une option vide
        var = st.selectbox("Sélectionnez la variable:", 
                          options=["---"] + list(st.session_state.merged_data.columns))
        
        # Ne continuer que si une variable réelle est sélectionnée
        if var != "---":
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
                        st.dataframe(pd.concat([value_counts.set_index('Groupe'), group_stats], axis=1))
                        
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
        else:
            st.info("Veuillez sélectionner une variable à analyser")

    # Analyse bivariée
    elif analysis_type == "Analyse bivariée":
        try:
            # Initialisation de var_y dans session_state s'il n'existe pas
            if 'previous_var_y' not in st.session_state:
                st.session_state.previous_var_y = None
    
            # Ajout de l'option vide pour var_x
            var_x = st.selectbox("Variable X", 
                                 ["---"] + list(st.session_state.merged_data.columns), 
                                 key='var_x_select')
    
            if var_x != "---":
                # Filtrer les colonnes pour var_y en excluant var_x
                available_columns_y = [col for col in st.session_state.merged_data.columns if col != var_x]
    
                # Si la valeur précédente de var_y est valide, la mettre en premier dans la liste
                if st.session_state.previous_var_y in available_columns_y:
                    available_columns_y.remove(st.session_state.previous_var_y)
                    available_columns_y.insert(0, st.session_state.previous_var_y)
    
                # Ajout de l'option vide pour var_y
                var_y = st.selectbox("Variable Y", 
                                     ["---"] + available_columns_y, 
                                     key='var_y_select')
    
                # Ne continuer que si les deux variables sont sélectionnées
                if var_y != "---":
                    # Sauvegarder la valeur de var_y pour la prochaine itération
                    st.session_state.previous_var_y = var_y
    
                    # Détection des types de variables
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
                        st.dataframe(stats_df)
    
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
    
                        # Option d'agrégation avec variable de référence d'abord
                        st.write("Si certaines observations sont répétées dans votre jeu de données, vous pouvez choisir une variable de référence pour l'agrégation.")
    
                        do_aggregate = st.checkbox("Vérifier les observations répétées avec une variable de référence", key="do_aggregate_quant")
    
                        has_duplicates = False
                        agg_method = None
                        reference_var = None
    
                        if do_aggregate:
                            # Sélection de la variable de référence parmi toutes les variables
                            reference_var = st.selectbox(
                                "Sélectionner la variable de référence", 
                                st.session_state.merged_data.columns,
                                key="ref_var_quant"
                            )
                            
                            # Vérification des répétitions
                            has_duplicates = st.session_state.merged_data[reference_var].duplicated().any()
                            
                            if has_duplicates:
                                st.warning(f"⚠️ La variable {reference_var} contient des observations répétées. Une agrégation sera effectuée.")
                                # Méthode d'agrégation qui sera appliquée aux deux variables
                                agg_method = st.radio(
                                    "Méthode d'agrégation", 
                                    ['sum', 'mean', 'median'],
                                    format_func=lambda x: {'sum': 'Somme', 'mean': 'Moyenne', 'median': 'Médiane'}[x],
                                    key="agg_method_quant"
                                )
                            else:
                                st.info(f"La variable {reference_var} ne contient pas d'observations répétées. L'agrégation n'est pas nécessaire.")
    
                        # Sélection des variables X et Y après la configuration de l'agrégation
                        numeric_cols = [col for col in st.session_state.merged_data.columns 
                                       if is_numeric_column(st.session_state.merged_data, col)]
    
                        var_x = st.selectbox("Variable X", numeric_cols, key='var_x_quant')
                        var_y = st.selectbox(
                            "Variable Y", 
                            [col for col in numeric_cols if col != var_x],
                            key='var_y_quant'
                        )
    
                        # Traitement des données selon la configuration d'agrégation
                        if do_aggregate and has_duplicates and reference_var and agg_method:
                            # Agrégation des deux variables avec la même méthode
                            agg_data = st.session_state.merged_data.groupby(reference_var).agg({
                                var_x: agg_method,
                                var_y: agg_method
                            }).reset_index()
                            
                            results_df, response_rate_x, response_rate_y = analyze_quantitative_bivariate(
                                st.session_state.merged_data,
                                var_x,
                                var_y,
                                groupby_col=reference_var,
                                agg_method=agg_method
                            )
                        else:
                            agg_data = st.session_state.merged_data
                            results_df, response_rate_x, response_rate_y = analyze_quantitative_bivariate(
                                st.session_state.merged_data,
                                var_x,
                                var_y
                            )
    
                        # Affichage des taux de réponse
                        st.write("Taux de réponse :")
                        st.write(f"- {var_x} : {response_rate_x:.1f}%")
                        st.write(f"- {var_y} : {response_rate_y:.1f}%")
    
                        st.write("Statistiques de corrélation")
                        st.dataframe(results_df)
    
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
    
                else:
                    st.info("Veuillez sélectionner une variable Y")
            else:
                st.info("Veuillez sélectionner une variable X")
                    
        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")

# Exécution de l'application
if __name__ == "__main__":
    main()

