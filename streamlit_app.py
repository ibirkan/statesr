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
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.grid_options_builder import GridOptionsBuilder
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
    
    # Remplacement des non-réponses par np.nan pour les colonnes sélectionnées
    data[var_x] = data[var_x].replace(missing_values, np.nan)
    data[var_y] = data[var_y].replace(missing_values, np.nan)
    if groupby_col:
        data[groupby_col] = data[groupby_col].replace(missing_values, np.nan)
    
    # Si une colonne de groupement est spécifiée, agrégeons d'abord les données
    if groupby_col is not None:
        # Suppression des non-réponses pour les variables quantitatives et la variable d'agrégation
        data = data.dropna(subset=[var_x, var_y, groupby_col])
        data = data.groupby(groupby_col).agg({
            var_x: agg_method,
            var_y: agg_method
        }).reset_index()
    else:
        # Suppression des non-réponses pour les variables quantitatives seulement
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
    Détecte automatiquement quelles variables doivent être agrégées en comptant
    le nombre de valeurs uniques pour chaque modalité du groupby_col.
    
    Returns:
        tuple: (vars_to_aggregate, vars_to_keep_raw)
    """
    # Pour chaque variable, on compte le nombre de valeurs différentes par groupby_col
    x_values_per_group = df.groupby(groupby_col)[var_x].nunique()
    y_values_per_group = df.groupby(groupby_col)[var_y].nunique()
    
    # Initialisation des variables à agréger et à garder
    vars_to_aggregate = []
    vars_to_keep_raw = []
    
    # Si une variable a plusieurs valeurs pour certains groupes, elle doit être agrégée
    if (x_values_per_group > 1).any():
        vars_to_aggregate.append(var_x)
    else:
        vars_to_keep_raw.append(var_x)
    
    if (y_values_per_group > 1).any():
        vars_to_aggregate.append(var_y)
    else:
        vars_to_keep_raw.append(var_y)
    
    return vars_to_aggregate, vars_to_keep_raw

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

def plot_density(plot_data, var, title, x_axis, y_axis):
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

def plot_qualitative_bar(data, title, x_label, y_label, color_palette, show_values=True):
    """
    Crée un graphique en barres pour une variable qualitative.
    """
    fig = go.Figure(data=[
        go.Bar(
            x=data['Modalité'],
            y=data['Effectif'],
            marker_color=color_palette[0]
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        margin=dict(t=100, b=100),
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray'
        )
    )

    if show_values:
        fig.update_traces(
            text=data['Effectif'].round(1),
            textposition='outside',
            texttemplate='%{text:.1f}'
        )

    return fig

def plot_qualitative_lollipop(data, title, x_label, y_label, color_palette, show_values=True):
    """
    Crée un graphique lollipop pour une variable qualitative avec des bâtons verticaux.
    """
    fig = go.Figure()
    
    # Pour chaque point, créer une ligne verticale partant de zéro
    for i in range(len(data)):
        fig.add_trace(go.Scatter(
            x=[data['Modalité'].iloc[i], data['Modalité'].iloc[i]],
            y=[0, data['Effectif'].iloc[i]],
            mode='lines',
            line=dict(color=color_palette[0], width=2),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Ajout des points (sucettes)
    fig.add_trace(go.Scatter(
        x=data['Modalité'],
        y=data['Effectif'],
        mode='markers',
        marker=dict(color=color_palette[0], size=10),
        name='Valeurs',
        showlegend=False
    ))

    # Ajout des valeurs au-dessus des points
    if show_values:
        # Calcul de la position Y pour le texte (20% plus haut que les points)
        text_y = data['Effectif'] + (data['Effectif'].max() * 0.1)
        
        fig.add_trace(go.Scatter(
            x=data['Modalité'],
            y=text_y,  # Position Y ajustée
            mode='text',
            text=data['Effectif'].round(0).astype(str),
            textposition='middle center',
            textfont=dict(size=12),
            showlegend=False
        ))

    # Mise à jour du layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        margin=dict(t=100, b=100),
        plot_bgcolor='white',
        yaxis=dict(
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='lightgray',
            gridcolor='lightgray',
            range=[0, max(data['Effectif']) * 1.3]  # Plus d'espace pour le texte
        ),
        xaxis=dict(
            gridcolor='lightgray'
        )
    )

    return fig
    
def plot_qualitative_treemap(data, title, color_palette):
    """
    Crée un treemap pour une variable qualitative.
    """
    fig = px.treemap(
        data,
        path=['Modalité'],
        values='Effectif',
        title=title,
        color='Effectif',
        color_continuous_scale=[color_palette[0]] * 2
    )

    fig.update_layout(
        height=600,
        margin=dict(t=100, b=100),
        # Amélioration de la lisibilité des labels
        uniformtext=dict(minsize=10, mode='hide'),
        # Ajustement des marges pour éviter la superposition
        margin_pad=5
    )

    return fig

def calculate_grouped_stats(data, var, groupby_col, agg_method='mean'):
    """
    Calcule les statistiques avec agrégation.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Données source
    var : str
        Variable à analyser
    groupby_col : str
        Colonne d'agrégation
    agg_method : str
        Méthode d'agrégation ('sum', 'mean', 'median')
    """
    # Nettoyage des données pour les deux variables
    clean_data = data.dropna(subset=[var, groupby_col])
    
    # Calcul des statistiques sans agrégation (niveau détaillé)
    detailed_stats = {
        'sum': clean_data[var].sum(),
        'mean': clean_data[var].mean(),
        'median': clean_data[var].median(),
        'std': clean_data[var].std(),
        'count': len(clean_data)
    }
    
    # Agrégation des données
    agg_data = clean_data.groupby(groupby_col).agg({
        var: agg_method
    }).reset_index()
    
    # Calcul des statistiques sur données agrégées
    agg_stats = {
        'sum': agg_data[var].sum(),
        'mean': agg_data[var].mean(),
        'median': agg_data[var].median(),
        'std': agg_data[var].std(),
        'count': len(agg_data)  # nombre de groupes
    }
    
    return detailed_stats, agg_stats, agg_data

# Interface Streamlit
def display_comparison_stats(data, var, groupby_col):
    agg_method = st.radio(
        "Méthode d'agrégation",
        ['sum', 'mean', 'median'],
        format_func=lambda x: {
            'sum': 'Somme',
            'mean': 'Moyenne',
            'median': 'Médiane'
        }[x],
        key="agg_method_univ"
    )
    
    # Calcul des statistiques
    detailed_stats, agg_stats, agg_data = calculate_grouped_stats(
        data, var, groupby_col, agg_method
    )
    
    # Affichage des résultats
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Statistiques au niveau détaillé")
        st.write(f"Somme: {detailed_stats['sum']:,.2f}")
        st.write(f"Moyenne: {detailed_stats['mean']:,.2f}")
        st.write(f"Médiane: {detailed_stats['median']:,.2f}")
        st.write(f"Écart-type: {detailed_stats['std']:,.2f}")
        st.write(f"Nombre d'observations: {detailed_stats['count']}")
    
    with col2:
        st.write("Statistiques après agrégation")
        st.write(f"Somme: {agg_stats['sum']:,.2f}")
        st.write(f"Moyenne: {agg_stats['mean']:,.2f}")
        st.write(f"Médiane: {agg_stats['median']:,.2f}")
        st.write(f"Écart-type: {agg_stats['std']:,.2f}")
        st.write(f"Nombre de groupes: {agg_stats['count']}")
        
    return agg_data

def create_interactive_stats_table(stats_df):
    """
    Crée un tableau de statistiques interactif avec des fonctionnalités avancées.
    
    Parameters:
    -----------
    stats_df : pandas.DataFrame
        DataFrame contenant les statistiques
    """
    gb = GridOptionsBuilder.from_dataframe(stats_df)
    
    # Fonctionnalités de base
    gb.configure_default_column(
        sorteable=True,              # Tri
        filterable=True,             # Filtres
        resizable=True,              # Redimensionnement des colonnes
        draggable=True              # Réorganisation des colonnes
    )
    
    # Fonctionnalités de groupe
    gb.configure_grid_options(
        enableRangeSelection=True,   # Sélection de plages
        groupable=True,             # Groupement
        groupDefaultExpanded=1      # Groupes développés par défaut
    )
    
    # Configuration des agrégations par groupe
    gb.configure_columns(
        stats_df.columns.tolist(),
        groupable=True,
        value=True,
        aggFunc='sum',              # Fonction d'agrégation par défaut
        enableValue=True
    )
    
    gridOptions = gb.build()
    
    return AgGrid(
        stats_df,
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        update_mode='VALUE_CHANGED',
        fit_columns_on_grid_load=True,
        theme='streamlit'           # Thème correspondant à Streamlit
    )

def show_indicator_form(statistics, analysis_type, variables_info):
    """
    Version test simplifiée avec 3 champs.
    """
    st.write("### Test création d'indicateur")
    
    with st.form("test_indicator_form"):
        name = st.text_input("Nom de l'indicateur")
        description = st.text_input("Description")
        creation_date = datetime.now().strftime("%Y-%m-%d")
        
        submit = st.form_submit_button("Enregistrer")
        
        if submit:
            test_data = {
                "name": name,
                "description": description,
                "creation_date": creation_date
            }
            
            try:
                save_test_indicator(test_data)
                st.success("✅ Test d'enregistrement réussi!")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")

def save_test_indicator(test_data):
    """
    Version test simplifiée de la sauvegarde.
    """
    try:
        # Préparation des données
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
        
        # Debug : afficher les données avant envoi
        st.write("Données à envoyer:", grist_data)
        
        # Appel API
        response = grist_api_request(
            "4",
            method="POST",
            data=grist_data
        )
        
        # Debug : afficher la réponse
        st.write("Réponse reçue:", response)
        
        return response
        
    except Exception as e:
        raise Exception(f"Erreur test : {str(e)}")

# Fonctions pour les différentes pages
def page_analyse():
    st.title("Analyse des données ESR")
    
def main():
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
        var = st.selectbox(
            "Sélectionnez la variable:", 
            options=["---"] + list(st.session_state.merged_data.columns)
        )
        
        # Ne continuer que si une variable réelle est sélectionnée
        if var != "---":
            # Nettoyage des données pour la variable sélectionnée
            plot_data = st.session_state.merged_data[var].copy()
            plot_data = plot_data.dropna()
            
            if plot_data is not None and not plot_data.empty:
                # Détecter le type de variable
                is_numeric = pd.api.types.is_numeric_dtype(plot_data)
                
                # Affichage des statistiques de base
                st.write(f"### Statistiques principales de la variable {var}")
                
                if is_numeric:
                    # Gestion des variables numériques
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
                    
                    # Statistiques pour variable quantitative
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
                    
                    grid_response = create_interactive_stats_table(stats_df)
                    
                    if do_aggregate:
                        st.info("Note : Les statistiques sont calculées à l'échelle de la variable d'agrégation sélectionnée.")
    
                    # Options de regroupement
                    st.write("### Options de regroupement")
                    grouping_method = st.selectbox(
                        "Méthode de regroupement",
                        ["Aucune", "Quantile", "Manuelle"]
                    )
                    
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
                    value_counts = grouped_data.value_counts().reset_index()
                    value_counts.columns = ['Groupe', 'Effectif']
                    value_counts['Taux (%)'] = (value_counts['Effectif'] / len(plot_data) * 100).round(2)
                    
                    # Tri des valeurs pour avoir les groupes dans le bon ordre
                    value_counts = value_counts.sort_values('Groupe')
                    
                    group_stats = plot_data.groupby(grouped_data).agg(['sum', 'mean', 'max']).round(2)
                    group_stats.columns = ['Somme', 'Moyenne', 'Maximum']
                        
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
                
                # Configuration de la visualisation
                st.write("### Configuration de la visualisation")
                
                # Sélection du type de graphique et de la palette de couleurs
                viz_col1, viz_col2 = st.columns([1, 2])
                with viz_col1:
                    if is_numeric:
                        if grouping_method == "Aucune":
                            graph_type = st.selectbox(
                                "Type de graphique",
                                ["Histogramme", "Density plot"]
                            )
                        else:
                            graph_type = st.selectbox(
                                "Type de graphique",
                                ["Bar plot", "Lollipop plot", "Treemap"]
                            )
                    else:
                        graph_type = st.selectbox(
                            "Type de graphique",
                            ["Bar plot", "Lollipop plot", "Treemap"]
                        )
                
                with viz_col2:
                    color_scheme = st.selectbox(
                        "Palette de couleurs",
                        list(COLOR_PALETTES.keys())
                    )
                
                # Options avancées
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
                        if not is_numeric:
                            value_type = st.radio("Type de valeur à afficher", ["Effectif", "Taux (%)"], key="value_type_adv")
    
                # Génération du graphique
                if st.button("Générer la visualisation"):
                    try:
                        # Préparation des annotations
                        annotations = []
                        current_y = -0.15
                
                        if source:
                            annotations.append(dict(
                                text=f"Source : {source}",
                                xref="paper",
                                yref="paper",
                                x=0,
                                y=current_y,
                                showarrow=False,
                                font=dict(size=10),
                                align="left"
                            ))
                            current_y -= 0.05
                
                        if note:
                            annotations.append(dict(
                                text=f"Note : {note}",
                                xref="paper",
                                yref="paper",
                                x=0,
                                y=current_y,
                                showarrow=False,
                                font=dict(size=10),
                                align="left"
                            ))
                
                        # Création du graphique selon le type de variable
                        if not is_numeric:  # Pour les variables qualitatives
                            data_to_plot = value_counts.copy()
                            if value_type == "Taux (%)":
                                data_to_plot['Effectif'] = data_to_plot['Taux (%)']
                                y_axis = "Taux (%)" if y_axis == "Valeur" else y_axis

                            # Convertir les objets Interval en chaînes de caractères
                            data_to_plot['Modalité'] = data_to_plot['Modalité'].astype(str)
                
                            if graph_type == "Bar plot":
                                fig = plot_qualitative_bar(
                                    data_to_plot,
                                    title,
                                    x_axis,
                                    y_axis,
                                    COLOR_PALETTES[color_scheme],
                                    show_values
                                )
                            elif graph_type == "Lollipop plot":
                                fig = plot_qualitative_lollipop(
                                    data_to_plot,
                                    title,
                                    x_axis,
                                    y_axis,
                                    COLOR_PALETTES[color_scheme],
                                    show_values
                                )
                            elif graph_type == "Treemap":
                                fig = plot_qualitative_treemap(
                                    data_to_plot,
                                    title,
                                    COLOR_PALETTES[color_scheme]
                                )
                
                        else:  # Pour les variables numériques
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
                                    fig = plot_density(plot_data, var, title, x_axis, y_axis)
                            else:  # Pour les données groupées
                                data_to_plot = pd.DataFrame({
                                    'Modalité': value_counts['Groupe'].astype(str),  # Conversion en chaîne de caractères ici
                                    'Effectif': value_counts['Effectif']
                                })
                
                                if graph_type == "Bar plot":
                                    fig = plot_qualitative_bar(
                                        data_to_plot,
                                        title,
                                        x_axis,
                                        y_axis,
                                        COLOR_PALETTES[color_scheme],
                                        show_values
                                    )
                                elif graph_type == "Lollipop plot":
                                    fig = plot_qualitative_lollipop(
                                        data_to_plot,
                                        title,
                                        x_axis,
                                        y_axis,
                                        COLOR_PALETTES[color_scheme],
                                        show_values
                                    )
                                else:  # Treemap
                                    fig = plot_qualitative_treemap(
                                        data_to_plot,
                                        title,
                                        COLOR_PALETTES[color_scheme]
                                    )
                
                        # Ajout des annotations au graphique
                        if annotations and isinstance(fig, go.Figure):
                            fig.update_layout(annotations=annotations)

                        # Configuration des axes et du titre
                        fig.update_layout(
                            title=title,
                            xaxis_title=x_axis,
                            yaxis_title=y_axis,
                        )
                
                        # Affichage du graphique
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
      
                # Détection des types de variables avec gestion d'erreur
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
            st.error(f"Une erreur s'est produite : {str(e)}")

# Exécution de l'application
if __name__ == "__main__":
    main()
