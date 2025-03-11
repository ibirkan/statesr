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

    # ✅ Préparation des labels avec retour à la ligne
    wrapped_labels = [
        "<br>".join(textwrap.wrap(str(label), width=20)) if isinstance(label, str) and len(label) > 20 else str(label)
        for label in data['Modalités']
    ]

    # ✅ Création du graphique
    fig = go.Figure()

    # ✅ Configuration de la taille et des marges
    num_bars = len(data)
    base_height = 100 + (num_bars * 60)
    bottom_margin = 100 + (30 * (bool(source) + bool(note)))  # ✅ Ajustement dynamique de la marge

    fig.update_layout(
        autosize=False,
        width=900,
        height=base_height,
        margin=dict(
            l=250,  # Large marge gauche pour éviter la troncature des labels
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

    # ✅ Ajout des barres
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
    fig.update_yaxes(
        title=x_label,
        autorange="reversed",
        tickfont=dict(family="Marianne, sans-serif", size=14)
    )

    # ✅ Ajouter un peu d'espace à droite pour les valeurs
    max_value = data[y_column].max()
    padding = max_value * 0.15

    fig.update_xaxes(
        title=value_type,
        range=[0, max_value + padding],
        tickfont=dict(family="Marianne, sans-serif", size=14),
        gridcolor='lightgray'
    )

    # ✅ Annotations pour source et note
    annotations = []
    annotation_y = -0.25 - (0.05 * num_bars)  # ✅ Dynamisation de la position

    if source:
        annotations.append(dict(
            text=f"📌 Source : {source}",
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
            text=f"📝 Note : {note}",
            x=0,
            y=annotation_y - 0.05,  # ✅ Espacement entre source et note
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(family="Marianne, sans-serif", size=12, color="gray"),
            align='left',
            xanchor='left'
        ))

    fig.update_layout(annotations=annotations)

    # ✅ Marquer pour l'export
    fig._is_horizontal_bar = True

    return fig
