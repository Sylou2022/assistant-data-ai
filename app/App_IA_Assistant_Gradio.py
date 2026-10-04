# code : App_IA_Assistant_Gradio.py

"""Assistant Data AI : interface Gradio.

Lancement (depuis la racine du projet) :
    python -m app.App_IA_Assistant_Gradio
"""

from __future__ import annotations
import plotly.graph_objects as go
import plotly.graph_objects as go

import html
import logging
import re
import tempfile
import time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from datetime import datetime

from app.llm.openai_client import OpenAIClient
from app.pipeline.data_pipeline import DataPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Constantes
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "image"

# Icône seule en priorité, sinon l'ancien logo complet
LOGO_PATH = next(
    (
        path
        for path in (
            IMAGE_DIR / "logo-icon.png",
            IMAGE_DIR / "generated-image.png",
        )
        if path.is_file()
    ),
    IMAGE_DIR / "logo-icon.png",
)
HAS_LOGO = LOGO_PATH.is_file()

EXAMPLES = [
    "Chiffre d'affaires par région",
    "Chiffre d'affaires par mois",
    "Marge par produit",
    "Quantité vendue par produit",
    "Prévois le chiffre d'affaires pour les 6 prochains mois",
]

WELCOME = (
    "Bonjour 👋 Je suis votre assistant Data. "
    "Posez-moi simplement votre question en langage naturel "
    "et je vous aiderai à trouver l'information recherchée."
)

# Couleurs
INDIGO = "#4f46e5"
TEXT_DARK = "#1e1b4b"
ZEBRA = ("#ffffff", "#f5f3ff")      # lignes alternées du tableau
HEAT_LOW = (224, 231, 255)          # valeur la plus basse (indigo clair)
HEAT_HIGH = (79, 70, 229)           # valeur la plus haute (indigo foncé)
MAX_STYLED_ROWS = 5000              # au-delà, pas de coloration (performance)

# Colonnes à ne pas traiter comme des indicateurs chiffrés
TIME_WORDS = (
    "annee", "année", "year", "mois", "month", "jour", "day",
    "trimestre", "quarter", "semaine", "week", "date",
)
# Indicateurs pour lesquels somme et part du total n'ont pas de sens
NON_ADDITIVE_WORDS = ("taux", "moyenne", "ratio", "pourcent", "%", "prix", "pct", "avg")

# Nettoyage des noms de colonnes techniques
LABEL_REPLACEMENTS = (
    (r"\bchiffre affaires\b", "chiffre d'affaires"),
    (r"\bquantite\b", "quantité"),
    (r"\bregion\b", "région"),
    (r"\bannee\b", "année"),
    (r"\bcategorie\b", "catégorie"),
    (r"\bcout\b", "coût"),
    (r"\bbenefice\b", "bénéfice"),
    (r"\bca\b", "CA"),
)

MONTHS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)

# Colonnes contenant un numéro de mois (1 à 12) à afficher en toutes lettres
MONTH_COLUMN_NAMES = ("mois", "month")

# Dimensions proposées pour les questions de suivi
FOLLOW_UP_DIMENSIONS = (
    ("region", "région"),
    ("produit", "produit"),
    ("mois", "mois"),
    ("annee", "année"),
)

# Force le mode clair, quel que soit le réglage du navigateur
FORCE_LIGHT_JS = """
() => {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

CSS = """
/* =========================================================
   POLICE GLOBALE
   ========================================================= */

* {
    font-family: Arial, Helvetica, sans-serif !important;
}


/* =========================================================
   CONTENEUR PRINCIPAL
   ========================================================= */

.gradio-container {
    max-width: 1500px !important;
    background: #ffffff !important;
    padding-top: 6px !important;
}

#sql-box {
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px;
    padding: 10px;
}

/* =========================================================
   HEADER — VERSION PROPRE
   ========================================================= */

#entete {
    height: 72px !important;
    min-height: 72px !important;
    margin: 0 !important;
    padding: 0 !important;

    align-items: flex-start !important;
}


/* =========================================================
   GAUCHE : LOGO + TITRE
   ========================================================= */

#entete-gauche {
    height: 72px !important;
    min-height: 72px !important;

    margin: 0 !important;
    padding: 0 !important;

    display: flex !important;
    flex-direction: row !important;
    align-items: flex-start !important;
}


/* =========================================================
   LOGO
   ========================================================= */

#logo {
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;

    margin: 0 12px 0 0 !important;
    padding: 0 !important;
}


/* =========================================================
   TITRE + SOUS-TITRE
   ========================================================= */

#titre {
    height: auto !important;
    min-height: 48px !important;

    margin: 0 !important;
    padding: 1px 0 0 0 !important;

    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}


/* Titre principal */
#titre h1 {
    display: block !important;

    margin: 0 0 4px 0 !important;
    padding: 0 !important;

    font-size: 1.30rem !important;
    line-height: 1.15 !important;
}


/* Sous-titre */
#titre p {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;

    height: auto !important;
    max-height: none !important;

    margin: 0 !important;
    padding: 0 !important;

    font-size: 0.78rem !important;
    line-height: 1.2 !important;

    overflow: visible !important;
}




/* =========================================================
   KPI — HEADER, À DROITE
   ========================================================= */

#kpi {
    width: auto !important;
    margin: 0 0 0 auto !important;
    padding: 0 !important;
    align-self: flex-start !important;

    position: relative !important;
    top: -22px !important;
}

#kpi .kpi-section {
    width: auto !important;
    margin: 0 !important;
    padding: 0 !important;

    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
}

#kpi .metrics {
    width: auto !important;

    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;

    justify-content: flex-end !important;
    align-items: stretch !important;

    gap: 8px !important;

    margin: 0 !important;
    padding: 0 !important;
}

#kpi .metrics > * {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 130px !important;
}

#kpi .kpi-status-row {
    width: 100% !important;

    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;

    justify-content: flex-end !important;
    align-items: center !important;

    gap: 6px !important;

    margin-top: 4px !important;
    padding: 0 !important;
}

#kpi .kpi-status-row .kpi-badge {
    margin: 0 !important;
    white-space: nowrap !important;
}

/* =========================================================
   HEADER — COMPACT MAIS SOUS-TITRE VISIBLE
   ========================================================= */

#entete {
    min-height: 68px !important;
    height: 68px !important;
    margin: 0 !important;
    padding: 0 !important;
    align-items: center !important;
}

#entete-gauche {
    height: 68px !important;
    min-height: 68px !important;
    margin: 0 !important;
    padding: 0 !important;
    align-items: center !important;
}

/* Logo */
#logo {
    height: 52px !important;
    width: 52px !important;
    min-width: 52px !important;
}


/* =========================================================
   TITRE
   ========================================================= */

#titre {
    margin: 0 !important;
    padding: 0 !important;
}

#titre h1 {
    margin: 0 0 5px 0 !important;
    padding: 0 !important;
    font-size: 1.35rem !important;
    line-height: 1.1 !important;
}

#titre p {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0.82rem !important;
    line-height: 1.2 !important;
}


/* =========================================================
   KPI
   ========================================================= */

#kpi {
    margin: 0 !important;
    padding: 0 !important;
}

#kpi .kpi-section {
    margin: 0 !important;
    padding: 0 !important;
}

#kpi .metrics {
    margin: 0 !important;
    padding: 0 !important;
    gap: 8px !important;
}


/* =========================================================
   ESPACE SOUS LE HEADER
   ========================================================= */

#workspace {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

#chat-column,
#results-column {
    margin-top: 0 !important;
    padding-top: 0 !important;
}




/* =========================================================
   TITRE + SOUS-TITRE
   ========================================================= */

#titre {
    margin: 0 !important;
    padding: 0 !important;
}

#titre h1 {
    margin: 0 0 4px 0 !important;
    padding: 0 !important;
    font-size: 1.35rem !important;
    line-height: 1.1 !important;
}

#titre p {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0.80rem !important;
    line-height: 1.15 !important;
}




/* =========================================================
   KPI — ALIGNEMENT HORIZONTAL
   ========================================================= */

#kpi {
    width: 100% !important;
    margin-left: auto !important;
}

/* Conteneur global généré par build_kpi_html() */
#kpi .kpi-section {
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
}




/* =========================================================
   REQUÊTE SQL
   ========================================================= */

#titre-sql {
    margin-top: 14px !important;
    margin-bottom: 6px !important;
}

#sql-box {
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}



/* =========================================================
   HEADER COMPACT
   ========================================================= */

/* Hauteur minimale du header */
#entete {
    min-height: 65px !important;
    height: 65px !important;
    margin: 0 !important;
    padding: 0 !important;
    align-items: flex-start !important;
}

/* Zone logo + titre */
#entete-gauche {
    height: 65px !important;
    min-height: 65px !important;
    margin: 0 !important;
    padding: 0 !important;
    align-items: flex-start !important;
}

/* Logo plus compact */
#logo {
    height: 48px !important;
    width: 48px !important;
    min-width: 48px !important;
}

/* Titre principal */
#titre {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 0.88rem !important;
    line-height: 1.05 !important;
}

/* Réduit les marges générées par le Markdown */
#titre h1 {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 1.35rem !important;
    line-height: 1.05 !important;
}

#titre p {
    margin: 2px 0 0 0 !important;
    padding: 0 !important;
    font-size: 0.78rem !important;
    line-height: 1 !important;
}


/* =========================================================
   KPI PLUS COMPACTS
   ========================================================= */

#kpi {
    margin: 0 !important;
    padding: 0 !important;
}

#kpi .kpi-section {
    margin: 0 !important;
    padding: 0 !important;
}

#kpi .metrics {
    gap: 8px !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Cartes KPI */
#kpi .metrics > * {
    padding: 6px 10px !important;
    min-height: 42px !important;
}

/* Réduire les textes à l'intérieur des KPI */
#kpi .metrics h1,
#kpi .metrics h2,
#kpi .metrics h3,
#kpi .metrics p {
    margin: 0 !important;
    line-height: 1.05 !important;
}

/* Badges sous les KPI */
#kpi .kpi-status-row {
    margin-top: 4px !important;
    gap: 4px !important;
}


/* =========================================================
   ESPACE SOUS LE HEADER
   ========================================================= */

#workspace {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

#chat-column,
#results-column {
    margin-top: 0 !important;
    padding-top: 0 !important;
}




/* =========================================================
   RÉDUIRE AU MAXIMUM L'ESPACE SOUS L'EN-TÊTE
   ========================================================= */

#entete {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

#workspace {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Supprime l'espace éventuel de la colonne gauche/droite */
#chat-column,
#results-column {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Le premier élément de chaque colonne ne doit pas
   rajouter d'espace au-dessus */
#chat-column > :first-child,
#results-column > :first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* Réduit également l'espace généré par les Row/Column Gradio */
#workspace > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* Le chatbot commence immédiatement après le header */
#chatbot {
    margin-top: 0 !important;
}

/* Le bloc graphique commence au même niveau */
#graphique-principal {
    margin-top: 0 !important;
}




/* =========================================================
   REQUÊTE SQL — GRIS PLUS FONCÉ
   ========================================================= */

#sql-box {
    background: #e5e7eb !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}




/* =========================================================
   RÉSULTATS
   ========================================================= */

#results-column {
    min-width: 0;
}

#table {
    margin-top: 10px;
}


/* =========================================================
   INFORMATIONS TECHNIQUES
   ========================================================= */

#technical-tabs {
    margin-top: 12px;
}

/* =========================================================
   CARTES KPI
   ========================================================= */

.metrics {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 4px 0 10px;
    align-items: stretch;
}



.metric-card {
    flex: 1 1 180px;
    min-width: 0;

    padding: 14px 18px;
    border-radius: 14px;

    background: linear-gradient(
        135deg,
        #4f46e5,
        #6366f1
    );
    border: 1px solid #4f46e5;

    box-shadow:
        0 4px 14px rgba(79, 70, 229, .25);

    box-sizing: border-box;
}

.metric-card.alt {
    background: #ffffff;
    border: 1px solid #c7d2fe;
    box-shadow: 0 2px 8px rgba(79, 70, 229, .08);
}

.metric-card:not(.alt) * {
    color: #ffffff !important;
}

.metric-card.alt * {
    color: #3730a3 !important;
}

.metric-label {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: .78rem !important;
    line-height: 1.2 !important;
    opacity: .85;

    text-transform: uppercase;
    letter-spacing: .05em;
}

.metric-value {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;

    white-space: nowrap;
}

.metric-sub {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: .9rem !important;
    line-height: 1.3 !important;
    opacity: .85;
}



/* =========================================================
   PASTILLES KPI
   ========================================================= */

.kpi-badge {
    display: inline-block;
    padding: 4px 14px;
    margin: 0 8px 6px 0;
    border-radius: 999px;

    font-family: Arial, Helvetica, sans-serif !important;
    font-weight: 600;
    font-size: .9rem;

    background: #eef2ff;
    color: #3730a3;
    border: 1px solid #c7d2fe;
}

.kpi-badge.teal {
    background: #ccfbf1;
    color: #115e59;
    border-color: #99f6e4;
}

.kpi-badge.green {
    background: #dcfce7;
    color: #166534;
    border-color: #86efac;
}

.kpi-badge.orange {
    background: #ffedd5;
    color: #9a3412;
    border-color: #fdba74;
}

.kpi-badge.red {
    background: #fee2e2;
    color: #991b1b;
    border-color: #fca5a5;
}



/* =========================================================
   ZONE DE SAISIE
   ========================================================= */

#saisie {
    align-items: stretch;
    gap: 10px;
    margin-top: 10px;
}

#saisie textarea {
    min-height: 64px !important;
    padding: 15px 18px !important;

    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 15px !important;
    line-height: 1.45 !important;

    color: #1e1b4b !important;
    background: #ffffff !important;

    border: 1.5px solid #c7d2fe !important;
    border-radius: 14px !important;

    box-shadow:
        0 2px 6px rgba(30, 41, 59, 0.06),
        0 6px 18px rgba(79, 70, 229, 0.08) !important;

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease !important;
}

#saisie textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

#saisie textarea:focus {
    border-color: #6366f1 !important;
    outline: none !important;

    box-shadow:
        0 0 0 3px rgba(99, 102, 241, 0.14),
        0 6px 20px rgba(79, 70, 229, 0.10) !important;
}

#saisie button {
    height: 100%;
    min-height: 64px;

    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;

    border-radius: 12px !important;
}


/* =========================================================
   CHATBOT
   ========================================================= */

#chatbot {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 14px !important;

    background: linear-gradient(
        160deg,
        #eef2ff 0%,
        #e0f2fe 100%
    ) !important;

    border: 1px solid #c7d2fe !important;
    border-radius: 16px !important;
}

#chatbot .message,
#chatbot .message * {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 14px !important;
    line-height: 1.45 !important;
}

#chatbot .message.bot p,
#chatbot .message.bot li,
#chatbot .message.bot ul,
#chatbot .message.bot ol {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

#chatbot .bubble-wrap,
#chatbot .wrapper {
    background: transparent !important;
}

#chatbot .message.user {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
}

#chatbot .message.user * {
    color: #ffffff !important;
}

#chatbot .message.bot {
    background: #ffffff !important;
    color: #1e1b4b !important;
    border: 1px solid #c7d2fe !important;
}


/* =========================================================
   TABLEAU DE RÉSULTATS
   ========================================================= */

#table,
#table * {
    font-family: Arial, Helvetica, sans-serif !important;
}

#table th,
#table thead {
    background: #4f46e5 !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}

#table th span,
#table th button {
    color: #ffffff !important;
    font-family: Arial, Helvetica, sans-serif !important;
}

#table td {
    font-family: Arial, Helvetica, sans-serif !important;
    font-size: 13px !important;
    font-variant-numeric: tabular-nums;
}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#ffffff",
    body_background_fill_dark="#ffffff",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f8fafc",
    background_fill_secondary_dark="#f8fafc",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    body_text_color="#1e1b4b",
    body_text_color_dark="#1e1b4b",
)


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------


@lru_cache(maxsize=1)
def get_pipeline() -> DataPipeline:
    """Instancie le pipeline une seule fois avec OpenAI."""
    return DataPipeline(llm_client=OpenAIClient())


# ---------------------------------------------------------
# Détection des colonnes et formatage
# ---------------------------------------------------------


def _norm(name: Any) -> str:
    return str(name).lower()


def is_id_column(name: Any) -> bool:
    return _norm(name).endswith("id")


def is_time_column(name: Any) -> bool:
    return any(word in _norm(name) for word in TIME_WORDS)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes Decimal/objets numériques en vrais nombres."""
    df = df.copy().reset_index(drop=True)

    for col in df.columns:
        if df[col].dtype != object:
            continue

        values = df[col].dropna()
        if len(values) and all(
            isinstance(v, (Decimal, int, float)) and not isinstance(v, bool)
            for v in values
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return month_names(df)


def month_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace un numéro de mois (8) par son nom (août).

    La colonne devient une catégorie ordonnée : les tris et les graphiques
    gardent l'ordre janvier -> décembre au lieu de l'ordre alphabétique.
    """
    for col in df.columns:
        if _norm(col) not in MONTH_COLUMN_NAMES:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if (
            values.notna().all()
            and values.between(1, 12).all()
            and (values % 1 == 0).all()
        ):
            df[col] = pd.Categorical(
                [MONTHS_FR[int(v) - 1] for v in values],
                categories=MONTHS_FR,
                ordered=True,
            )

    return df


def measure_columns(df: pd.DataFrame) -> list:
    """Colonnes d'indicateurs chiffrés (hors identifiants et dates)."""
    return [
        col
        for col in df.select_dtypes(include="number").columns
        if not is_id_column(col) and not is_time_column(col)
    ]


def label_columns(df: pd.DataFrame) -> list:
    """Colonnes d'axes d'analyse (région, produit, mois...)."""
    measures = set(measure_columns(df))
    return [c for c in df.columns if c not in measures and not is_id_column(c)]


def pretty_column(name: Any) -> str:
    """Chiffre_Affaires_Total -> Chiffre d'affaires total."""
    text = str(name).replace("_", " ").strip().lower()
    text = re.sub(r"^nom ", "", text)

    for pattern, replacement in LABEL_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    return text[:1].upper() + text[1:]


def detect_unit(name: Any) -> str:
    """Devine l'unité d'un indicateur d'après son nom."""
    n = _norm(name)

    if any(word in n for word in ("taux", "pct", "pourcent", "ratio", "%")):
        return "%"

    euro_words = (
        "chiffre", "affaires", "marge", "montant",
        "prix", "cout", "revenu", "benefice",
    )
    if any(word in n for word in euro_words) or re.search(r"(^|_)ca($|_)", n):
        return "€"

    if "quantit" in n:
        return "unités"

    return ""


def format_number(value: int) -> str:
    return f"{value:,}".replace(",", "\u00a0")


def plural(count: int, word: str) -> str:
    """1 résultat / 2 résultats."""
    return f"{format_number(count)} {word}{'s' if count > 1 else ''}"


def format_percent(ratio: float) -> str:
    return f"{ratio * 100:.0f}\u00a0%"


def format_value(value: Any, unit: str = "", compact: bool = False) -> str:
    """Formate un nombre à la française : 13 136 952 € ou 13,1 M€."""
    number = float(value)

    # Un taux compris entre 0 et 1 est considéré comme une proportion
    if unit == "%" and abs(number) <= 1:
        number *= 100

    if compact and unit == "€" and abs(number) >= 1e6:
        scaled, suffix = (
            (number / 1e9, "Md€") if abs(number) >= 1e9 else (number / 1e6, "M€")
        )
        return f"{scaled:.1f}".replace(".", ",") + "\u00a0" + suffix

    if unit == "%":
        decimals = 0 if number.is_integer() else 1
    else:
        decimals = 0 if number.is_integer() or abs(number) >= 1000 else 2

    text = f"{number:,.{decimals}f}".replace(",", "\u00a0").replace(".", ",")
    return f"{text}\u00a0{unit}" if unit else text


def format_label(value: Any) -> str:
    """Affiche une valeur d'axe sans décimales parasites (2024, pas 2024.0)."""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%d/%m/%Y")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def is_additive(col: Any, values: pd.Series, unit: str) -> bool:
    """Une somme ou une part du total a-t-elle un sens pour cet indicateur ?"""
    return (
        values.min() >= 0
        and unit != "%"
        and not any(word in _norm(col) for word in NON_ADDITIVE_WORDS)
    )


# ---------------------------------------------------------
# Tableau : noms lisibles, nombres formatés, couleurs
# ---------------------------------------------------------


def _heat_style(ratio: float) -> str:
    """Style CSS d'une cellule selon sa position entre min et max (0 à 1)."""
    r, g, b = (
        round(low + (high - low) * ratio)
        for low, high in zip(HEAT_LOW, HEAT_HIGH)
    )
    text = "#ffffff" if ratio > 0.55 else TEXT_DARK
    return f"background-color: rgb({r},{g},{b}); color: {text};"


def readable_names(df: pd.DataFrame) -> list[str]:
    names = [pretty_column(c) for c in df.columns]
    return names if len(set(names)) == len(names) else [str(c) for c in df.columns]


def style_dataframe(df: pd.DataFrame):
    """Tableau lisible : noms propres, nombres formatés, lignes alternées et dégradé."""
    try:
        data = df.reset_index(drop=True)
        measures = set(measure_columns(data))
        names = readable_names(data)

        display = pd.DataFrame(index=data.index)
        for col, name in zip(data.columns, names):
            unit = detect_unit(col) if col in measures else ""
            display[name] = [
                ""
                if pd.isna(v)
                else (format_value(v, unit) if col in measures else format_label(v))
                for v in data[col]
            ]

        if len(data) > MAX_STYLED_ROWS:
            return display

        styles = pd.DataFrame("", index=display.index, columns=display.columns)
        for i in range(len(data)):
            styles.iloc[i, :] = (
                f"background-color: {ZEBRA[i % 2]}; color: {TEXT_DARK};"
            )

        # Nombres alignés à droite, dégradé selon la valeur
        for j, col in enumerate(data.columns):
            if col not in measures:
                continue

            series = data[col]
            low, high = series.min(), series.max()
            spread = (high - low) if pd.notna(low) else 0

            for i, value in enumerate(series):
                if pd.isna(value):
                    continue
                heat = _heat_style((value - low) / spread) if spread else ""
                styles.iat[i, j] = (heat or styles.iat[i, j]) + " text-align: right;"

        return display.style.apply(lambda _: styles, axis=None)

    except Exception:
        logger.exception("Mise en forme du tableau impossible")
        return df


# ---------------------------------------------------------
# Exports
# ---------------------------------------------------------


def export_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = readable_names(df)
    return out


def export_csv(df: pd.DataFrame) -> str:
    path = Path(tempfile.mkdtemp()) / "resultats.csv"
    export_frame(df).to_csv(path, index=False, encoding="utf-8-sig", sep=";", decimal=",")
    return str(path)


def export_excel(df: pd.DataFrame) -> str | None:
    """Export Excel mis en forme. Retourne None si openpyxl est absent."""
    try:
        from openpyxl.styles import Font, PatternFill

        path = Path(tempfile.mkdtemp()) / "resultats.xlsx"
        out = export_frame(df)

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            out.to_excel(writer, index=False, sheet_name="Résultats")
            sheet = writer.sheets["Résultats"]

            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="4F46E5")

            for idx, col in enumerate(out.columns, start=1):
                longest = max([len(str(col))] + [len(str(v)) for v in out[col].head(200)])
                letter = sheet.cell(row=1, column=idx).column_letter
                sheet.column_dimensions[letter].width = min(longest + 3, 50)

        return str(path)

    except Exception:
        logger.warning("Export Excel indisponible (pip install openpyxl)", exc_info=True)
        return None


# ---------------------------------------------------------
# Réponse en langage naturel
# ---------------------------------------------------------


def build_answer(question: str, df: pd.DataFrame) -> str:
    """Phrase de réponse directe avec le chiffre clé."""
    measures = measure_columns(df)
    data = df.dropna(subset=measures[:1]) if measures else df
    if not measures or data.empty:
        return ""

    labels = label_columns(df)
    value_col = measures[0]
    analysis_label = analysis_label_column(question, df)
    unit = detect_unit(value_col)
    metric = pretty_column(value_col).lower()

    q = question.lower()
    wants_min = any(w in q for w in ("moins", "minimum", "faible", "pire", "dernier", "plus bas"))
    wants_max = not wants_min and any(
        w in q for w in ("plus", "meilleur", "maximum", "premier", "top", "élevé")
    )

    ranked = data.sort_values(value_col, ascending=False)
    row = ranked.iloc[-1] if wants_min else ranked.iloc[0]
    value = f"**{format_value(row[value_col], unit, compact=True)}**"

    if not labels:
        return f"{pretty_column(value_col)} : {value}."

    if analysis_label is not None:
        label = f"**{format_label(row[analysis_label])}**"

        if is_time_column(analysis_label) and len(data) > 1:
            phrase = "Point bas" if wants_min else "Pic"
            return f"{phrase} en {label} — {metric} : {value}."

        if wants_min:
            return (
                f"{label} affiche la valeur la plus faible — "
                f"{metric} : {value}."
            )

        if wants_max or len(data) > 1:
            return f"{label} affiche la valeur la plus élevée — {metric} : {value}."

        return f"{label} — {metric} : {value}."

    return f"{metric} : {value}."


#...........................................................................................

def analysis_label_column(question: str, df: pd.DataFrame) -> Any | None:
    """
    Détermine la dimension réellement analysée à partir de la question.

    Exemple :
    - "CA par mois en 2024" -> Nom_Mois
    - "CA par région" -> Région
    - "CA par produit" -> Produit
    - "CA par année" -> Année
    """
    labels = label_columns(df)

    if not labels:
        return None

    q = _norm(question)

    # ---------------------------------------------------------
    # Dimension demandée explicitement par l'utilisateur
    # ---------------------------------------------------------

    # Mois : on privilégie Nom_Mois s'il existe,
    # car il est plus lisible que le numéro du mois.
    if "mois" in q or "month" in q:
        for col in labels:
            normalized = _norm(col)
            if "nom_mois" in normalized:
                return col

        for col in labels:
            normalized = _norm(col)
            if normalized in ("mois", "month"):
                return col

    # Région
    if "région" in q or "region" in q:
        for col in labels:
            if "region" in _norm(col):
                return col

    # Produit
    if "produit" in q:
        for col in labels:
            if "produit" in _norm(col):
                return col

    # Année
    if "année" in q or "annee" in q or "year" in q:
        for col in labels:
            normalized = _norm(col)
            if "annee" in normalized or "year" in normalized:
                return col

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    return labels[0]


#...........................................................................................


def build_insights(
    question: str,
    df: pd.DataFrame,
) -> list[str]:
    """
    Génère une synthèse métier concise et professionnelle.

    La dimension analysée est déterminée à partir de la question,
    et non simplement à partir de la première colonne du résultat.
    """
    measures = measure_columns(df)

    if not measures:
        return []

    data = df.dropna(subset=[measures[0]]).copy()

    if len(data) < 2:
        return []

    value_col = measures[0]
    unit = detect_unit(value_col)

    analysis_label = analysis_label_column(question, data)

    if analysis_label is None:
        return []

    ranked = data.sort_values(value_col, ascending=False)

    best = ranked.iloc[0]
    last = ranked.iloc[-1]

    values = ranked[value_col]
    additive = is_additive(value_col, values, unit)

    def fmt(value: Any) -> str:
        return format_value(value, unit, compact=True)

    dimension = pretty_column(analysis_label).lower()
    best_label = format_label(best[analysis_label])
    last_label = format_label(last[analysis_label])

    lines: list[str] = []

    # ---------------------------------------------------------
    # Valeur maximale
    # ---------------------------------------------------------

    lines.append(
        f"**{dimension.capitalize()} le plus élevé** : "
        f"**{best_label}** — **{fmt(best[value_col])}**."
    )

    # ---------------------------------------------------------
    # Valeur minimale
    # ---------------------------------------------------------

    lines.append(
        f"**{dimension.capitalize()} le plus faible** : "
        f"**{last_label}** — **{fmt(last[value_col])}**."
    )

    # ---------------------------------------------------------
    # Total et moyenne
    # ---------------------------------------------------------

    if additive:
        total = values.sum()
        average = values.mean()

        lines.append(
            f"**Total** : **{fmt(total)}** · "
            f"**moyenne** : **{fmt(average)}**."
        )
    else:
        lines.append(
            f"**Moyenne** : **{fmt(values.mean())}**."
        )

    return lines


# ---------------------------------------------------------
# Éléments du panneau de droite
# ---------------------------------------------------------


def metric_card(label: str, value: str, sub: str = "", alt: bool = False) -> str:
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    css = "metric-card alt" if alt else "metric-card"
    return (
        f'<div class="{css}"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{value}</div>{sub_html}</div>'
    )



def build_kpi_html(
    df: pd.DataFrame,
    elapsed: float | None,
    question: str = "",
    last_updated: Any = None,
    ) -> str:
    """Construit le bloc KPI supérieur."""


    cards = ""

    measures = measure_columns(df)
    data = df.dropna(subset=measures[:1]) if measures else df

    if measures and not data.empty:
        labels = label_columns(df)
        value_col = measures[0]
        unit = detect_unit(value_col)
        metric = pretty_column(value_col)
        analysis_label = analysis_label_column(question, df)

        def fmt(v: Any) -> str:
            return format_value(v, unit, compact=True)

        # ---------------------------------------------------------
        # KPI principal
        # ---------------------------------------------------------
        if len(data) == 1 or not labels:
            row = data.iloc[0]

            sub = (
                html.escape(format_label(row[labels[0]]))
                if labels
                else ""
            )

            cards = metric_card(
                metric,
                fmt(row[value_col]),
                sub,
            )

        else:
            # Meilleure valeur
            best = data.loc[data[value_col].idxmax()]

            if analysis_label is not None:
                label_name = pretty_column(analysis_label)

                label_value = html.escape(
                    format_label(best[analysis_label])
                )

                cards = metric_card(
                    label_name,
                    label_value,
                    fmt(best[value_col]),
                )
            else:
                cards = metric_card(
                    metric,
                    fmt(best[value_col]),
                )

            # Total uniquement pour les métriques additives
            if is_additive(
                value_col,
                data[value_col],
                unit,
            ):
                cards += metric_card(
                    "Total",
                    fmt(data[value_col].sum()),
                    metric,
                    alt=True,
                )

            # Moyenne
            cards += metric_card(
                "Moyenne",
                fmt(data[value_col].mean()),
                metric,
                alt=True,
            )

    # ---------------------------------------------------------
    # Badges de statut
    # ---------------------------------------------------------
    badges = []

    # Nombre de résultats
    badges.append(
        f'<span class="kpi-badge">'
        f'{plural(len(df), "résultat")}'
        f'</span>'
    )

    # SQL connecté
    badges.append(
        '<span class="kpi-badge green">'
        '🟢 SQL connecté'
        '</span>'
    )

    # Requête validée / lecture seule
    badges.append(
        '<span class="kpi-badge green">'
        '🔒 Requête validée · lecture seule'
        '</span>'
    )

    # Temps de réponse
    if elapsed is not None:
        seconds = f"{elapsed:.1f}".replace(".", ",")

        badges.append(
            f'<span class="kpi-badge">'
            f'⏱ {seconds} s'
            f'</span>'
        )

    # Date de dernière actualisation
    last_updated_text = format_last_updated(last_updated)

    if last_updated_text:
        badges.append(
            '<span class="kpi-badge">'
            f'↻ Mise à jour : '
            f'{html.escape(last_updated_text)}'
            '</span>'
        )

    # ---------------------------------------------------------
    # HTML final
    # ---------------------------------------------------------
    cards_html = (
        f'<div class="metrics">{cards}</div>'
        if cards
        else ""
    )

    badges_html = (
        f'<div class="kpi-status-row">'
        f'{"".join(badges)}'
        f'</div>'
    )

    return (
        '<div class="kpi-section">'
        f'{cards_html}'
        f'{badges_html}'
        '</div>'
)




def build_trace(result, elapsed: float | None) -> str:
    """Traçabilité : d'où vient le chiffre."""
    parts: list[str] = []

    explanation = getattr(result, "explanation", None)
    if explanation:
        parts.append(f"**Méthode** : {explanation}")

    tables = getattr(result, "tables_used", None)
    if tables:
        parts.append("**Tables utilisées** : " + ", ".join(f"`{t}`" for t in tables))

    columns = getattr(result, "columns_used", None)
    if columns:
        parts.append("**Colonnes utilisées** : " + ", ".join(f"`{c}`" for c in columns))

    assumptions = getattr(result, "assumptions", None)
    if assumptions:
        parts.append("**Hypothèses**\n" + "\n".join(f"- {a}" for a in assumptions))

    parts.append("**Contrôle de sécurité** : requête validée avant exécution (lecture seule).")

    if elapsed is not None:
        parts.append(f"**Temps de réponse** : {f'{elapsed:.1f}'.replace('.', ',')} s")

    return "\n\n".join(parts)


def build_suggestions(df: pd.DataFrame) -> list[list[str]]:
    """Questions de suivi proposées après une réponse."""
    measures = measure_columns(df)
    if not measures:
        return []

    metric = re.sub(r" totale?$", "", pretty_column(measures[0]).lower())
    metric = metric[:1].upper() + metric[1:]

    labels = label_columns(df)
    current = _norm(labels[0]) if labels else ""

    suggestions = [
        f"{metric} par {word}"
        for key, word in FOLLOW_UP_DIMENSIONS
        if key not in current
    ]

    # Après une réponse SQL, on propose aussi une prévision
    suggestions.append("Prévois le chiffre d'affaires pour les 6 prochains mois")

    return [[s] for s in suggestions[:4]]


def make_chart(df: pd.DataFrame) -> str | None:
    """Graphique automatique : courbe pour une évolution, barres pour un classement."""
    measures, labels = measure_columns(df), label_columns(df)
    if not measures or not labels or len(df) < 2:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        return None

    value_col = measures[0]

    # Choisir la meilleure colonne pour l'axe X
    # Priorité : mois > date > année > autre dimension
    normalized_labels = {
        col: _norm(col)
        for col in labels
    }

    month_label = next(
        (
            col
            for col in labels
            if "mois" in normalized_labels[col]
        ),
        None,
    )

    date_label = next(
        (
            col
            for col in labels
            if any(
                word in normalized_labels[col]
                for word in ("date", "jour")
            )
        ),
        None,
    )

    year_label = next(
        (
            col
            for col in labels
            if any(
                word in normalized_labels[col]
                for word in ("annee", "year")
            )
        ),
        None,
    )

    label_col = (
        month_label
        or date_label
        or year_label
        or labels[0]
    )
    
    unit = detect_unit(value_col)
    data = df.dropna(subset=[value_col])
    thousands = FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=150)
    try:
        if is_time_column(label_col):
            data = data.sort_values(label_col)

            names = [
                format_label(v)
                for v in data[label_col]
            ]

            positions = list(range(len(names)))

            ax.plot(
                positions,
                data[value_col],
                marker="o",
                color=INDIGO,
                linewidth=2.5,
            )

            ax.fill_between(
                positions,
                data[value_col],
                alpha=0.12,
                color=INDIGO,
            )

            ax.set_xticks(positions)
            ax.set_xticklabels(
                names,
                rotation=45,
                ha="right",
            )

            # -----------------------------------------------------
            # Année associée aux mois
            # -----------------------------------------------------
            chart_year = None

            if year_label and year_label in data.columns:
                years = data[year_label].dropna().unique()

                if len(years) == 1:
                    chart_year = str(years[0])

            # -----------------------------------------------------
            # Titre naturel
            # -----------------------------------------------------
            if chart_year:
                ax.set_title(
                    f"{pretty_column(value_col)} par {pretty_column(label_col).lower()} en {chart_year}",
                    loc="left",
                    fontsize=12,
                    fontweight="bold",
                    color=TEXT_DARK,
                )

                # Année affichée sous les mois
                ax.set_xlabel(
                    f"Année {chart_year}",
                    fontsize=9,
                )

            ax.yaxis.set_major_formatter(thousands)
            ax.grid(axis="y", alpha=0.2)
        else:
            data = data.sort_values(value_col, ascending=False).head(10)
            names = [format_label(v)[:28] for v in data[label_col]][::-1]
            values = list(data[value_col])[::-1]
            bars = ax.barh(names, values, color="#6366f1")
            ax.bar_label(
                bars,
                labels=[format_value(v, unit, compact=True) for v in values],
                padding=4,
                fontsize=8,
            )
            ax.xaxis.set_major_formatter(thousands)
            ax.margins(x=0.18)
            ax.grid(axis="x", alpha=0.2)

        # ax.set_title(
        #     f"{pretty_column(value_col)} par {pretty_column(label_col).lower()}",
        #     loc="left", fontsize=12, fontweight="bold", color=TEXT_DARK,
        # )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        fig.tight_layout()
        path = Path(tempfile.mkdtemp()) / "graphique.png"
        fig.savefig(path)
        return str(path)

    except Exception:
        logger.exception("Génération du graphique impossible")
        return None
    finally:
        plt.close(fig)

#..........................................................................................

def make_interactive_chart(df: pd.DataFrame):
    """
    Graphique interactif avec :
    - mois sur l'axe X
    - année sous les mois
    - titre naturel
    - valeur CA au survol
    """
    measures, labels = measure_columns(df), label_columns(df)

    if not measures or not labels or len(df) < 2:
        return None

    value_col = measures[0]

    # ---------------------------------------------------------
    # Choix de l'axe X
    # Priorité : mois > date > année > autre
    # ---------------------------------------------------------
    normalized_labels = {
        col: _norm(col)
        for col in labels
    }

    month_label = next(
        (
            col
            for col in labels
            if "mois" in normalized_labels[col]
        ),
        None,
    )

    date_label = next(
        (
            col
            for col in labels
            if any(
                word in normalized_labels[col]
                for word in ("date", "jour")
            )
        ),
        None,
    )

    year_label = next(
        (
            col
            for col in labels
            if any(
                word in normalized_labels[col]
                for word in ("annee", "year")
            )
        ),
        None,
    )

    label_col = (
        month_label
        or date_label
        or year_label
        or labels[0]
    )

    data = df.dropna(
        subset=[value_col]
    ).copy()

    # ---------------------------------------------------------
    # Année
    # ---------------------------------------------------------
    chart_year = None

    if year_label and year_label in data.columns:

        years = (
            data[year_label]
            .dropna()
            .unique()
        )

        if len(years) == 1:
            chart_year = str(years[0])

    # ---------------------------------------------------------
    # Tri chronologique
    # ---------------------------------------------------------
    if month_label:

        data = data.sort_values(
            label_col
        )

    elif date_label:

        data = data.sort_values(
            label_col
        )

    # ---------------------------------------------------------
    # Libellés
    # ---------------------------------------------------------
    names = [
        format_label(v)
        for v in data[label_col]
    ]

    values = data[value_col].tolist()

    # ---------------------------------------------------------
    # Titre
    # ---------------------------------------------------------
    if chart_year and month_label:
        title = (
            f"{pretty_column(value_col)} "
            f"par {pretty_column(label_col).lower()} "
            f"en {chart_year}"
        )

        x_title = f"Année {chart_year}"

    else:
        title = (
            f"{pretty_column(value_col)} "
            f"par {pretty_column(label_col).lower()}"
        )

        x_title = (
            pretty_column(year_label)
            if year_label
            else ""
        )

    # ---------------------------------------------------------
    # Graphique Plotly
    # ---------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=names,
            y=values,
            mode="lines+markers",
            line=dict(
                color="#4F46E5",
                width=3,
            ),
            marker=dict(
                size=8,
                color="#4F46E5",
            ),
            fill="tozeroy",
            fillcolor="rgba(79,70,229,0.12)",
            hovertemplate=(
                "<b>%{x}</b>"
                "<br>"
                "Chiffre d’affaires : "
                "<b>%{y:,.0f} €</b>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(
                size=18,
                color="#27245C",
            ),
        ),

        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(size=12),
            ),
            type="category",
            tickangle=-45,
        ),

        yaxis=dict(
            title=None,
            tickformat=",",
        ),

        hovermode="x unified",

        margin=dict(
            l=60,
            r=30,
            t=70,
            b=90,
        ),

        height=430,

        plot_bgcolor="white",
        paper_bgcolor="white",

        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
    )

    return fig

#..........................................................................................

def make_forecast_chart(fr) -> str | None:
    """
    Graphique de prévision avec :
    - historique
    - prévision
    - zone d'incertitude Lower / Upper
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

    except ImportError:
        return None

    try:
        date_col = fr.date_column
        value_col = fr.value_column

        history = fr.historical[[date_col, value_col]].copy()
        forecast = fr.forecast.copy()

        history[date_col] = pd.to_datetime(history[date_col])
        forecast[date_col] = pd.to_datetime(forecast[date_col])

        unit = detect_unit(value_col)

        fig, ax = plt.subplots(
            figsize=(11, 5.8),
            dpi=150,
        )

        try:
            formatter = FuncFormatter(
                lambda v, _: format_value(v, unit, compact=True)
            )
            ax.yaxis.set_major_formatter(formatter)

            # ---------------------------------------------
            # Historique
            # ---------------------------------------------

            ax.plot(
                history[date_col],
                history[value_col],
                color=INDIGO,
                linewidth=2.4,
                marker="o",
                markersize=4,
                label="Historique",
            )

            # ---------------------------------------------
            # Prévision
            # ---------------------------------------------

            ax.plot(
                forecast[date_col],
                forecast[value_col],
                color="#7c3aed",
                linewidth=2.8,
                marker="o",
                markersize=4,
                linestyle="--",
                label="Prévision",
            )

            # ---------------------------------------------
            # Zone d'incertitude
            # ---------------------------------------------

            if {"Lower", "Upper"} <= set(forecast.columns):

                ax.fill_between(
                    forecast[date_col],
                    forecast["Lower"],
                    forecast["Upper"],
                    color="#8b5cf6",
                    alpha=0.16,
                    label="Zone d'incertitude",
                )

            # ---------------------------------------------
            # Séparation historique / prévision
            # ---------------------------------------------

            if not history.empty and not forecast.empty:

                last_history = history[date_col].max()

                ax.axvline(
                    last_history,
                    color="#94a3b8",
                    linestyle=":",
                    linewidth=1.2,
                )

                ax.text(
                    last_history,
                    ax.get_ylim()[1],
                    "  Début prévision",
                    va="top",
                    fontsize=9,
                    color="#64748b",
                )

            # ---------------------------------------------
            # Style
            # ---------------------------------------------

            ax.set_title(
                f"{pretty_column(value_col)} — historique et prévision",
                loc="left",
                fontsize=14,
                fontweight="bold",
                color=TEXT_DARK,
                pad=14,
            )

            ax.set_xlabel("")
            ax.grid(axis="y", alpha=0.18)

            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            ax.legend(
                loc="upper left",
                frameon=False,
                ncol=3,
            )

            fig.autofmt_xdate()

            fig.tight_layout()

            path = Path(tempfile.mkdtemp()) / "prevision.png"

            fig.savefig(
                path,
                bbox_inches="tight",
                facecolor="white",
            )

            return str(path)

        finally:
            plt.close(fig)

    except Exception:
        logger.exception(
            "Génération du graphique de prévision impossible"
        )
        return None






# ---------------------------------------------------------
# Prévision (forecasting)
# ---------------------------------------------------------

# Fiabilité -> (classe CSS de la pastille, icône)
RELIABILITY_BADGES = {
    "bonne": ("green", "✅"),
    "moyenne": ("orange", "⚠️"),
    "faible": ("red", "⛔"),
}


def month_label(value: Any) -> str:
    """2026-10-01 -> octobre 2026."""
    ts = pd.Timestamp(value)
    return f"{MONTHS_FR[ts.month - 1]} {ts.year}"


def same_period_last_year(fr) -> float | None:
    """Total de l'historique sur les mêmes mois un an plus tôt."""
    history = fr.historical.copy()
    history[fr.date_column] = pd.to_datetime(history[fr.date_column])
    series = history.set_index(fr.date_column)[fr.value_column]

    previous = [
        pd.Timestamp(d) - pd.DateOffset(years=1)
        for d in pd.to_datetime(fr.forecast[fr.date_column])
    ]

    if not all(p in series.index for p in previous):
        return None

    total = float(series.loc[previous].sum())
    return total if total > 0 else None


def forecast_table(fr) -> pd.DataFrame:
    """Tableau métier : historique + prévisions."""
    date_col, value_col = fr.date_column, fr.value_column

    history = fr.historical[[date_col, value_col]].copy()
    history.columns = ["Date", value_col]
    history["Type"] = "Historique"

    future = fr.forecast

    projected = pd.DataFrame(
        {
            "Date": future[date_col],
            value_col: future[value_col],
            "Type": "Prévision",
        }
    )

    table = pd.concat(
        [history, projected],
        ignore_index=True,
    )

    first = ["Date", "Type"]

    return table[
        first + [
            c for c in table.columns
            if c not in first
        ]
    ]


def build_forecast_reply(
    result,
    panel: dict[str, Any],
    elapsed: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Réponse métier + panneau pour un résultat de prévision."""
    fr = result.forecast_result

    date_col, value_col = fr.date_column, fr.value_column
    unit = detect_unit(value_col)
    metric = pretty_column(value_col).lower()

    forecast = fr.forecast
    count = len(forecast)
    total = float(forecast[value_col].sum())

    previous = same_period_last_year(fr)
    variation = (total / previous - 1) * 100 if previous else None

    reliability = getattr(fr, "reliability", "faible")
    css, icon = RELIABILITY_BADGES.get(reliability, ("", "ℹ️"))
    metrics = fr.metrics or {}
    mape = metrics.get("MAPE")
    reason = getattr(fr, "selection_reason", "") or ""
    folds = getattr(fr, "backtest_folds", 0)
    window = getattr(fr, "backtest_horizon", 0)

    # Alertes : qualité de l'historique + limites de la prévision
    notes = list(
        dict.fromkeys(
            [
                *(getattr(fr, "quality_warnings", None) or []),
                *(getattr(result, "warnings", None) or []),
            ]
        )
    )

    # ---- Cartes d'indicateurs ----
    if variation is None:
        variation_text = ""
    elif round(variation) == 0:
        variation_text = "stable"
    else:
        variation_text = f"{variation:+.0f} %"

    sub = (
        f"{variation_text} vs mêmes mois un an plus tôt"
        if variation_text
        else ""
    )
    mape_text = (
        f"MAPE {mape:.1f} %".replace(".", ",")
        if mape is not None
        else "non validée"
    )

    cards = metric_card(
        f"{pretty_column(value_col)} prévu · {count} mois",
        format_value(total, unit, compact=True),
        html.escape(sub),
    )
    cards += metric_card("Fiabilité", reliability.capitalize(), mape_text, alt=True)
    cards += metric_card(
        "Modèle retenu",
        f'<span style="font-size:1.05rem">{html.escape(fr.model_name)}</span>',
        alt=True,
    )

    badges = [
        '<span class="kpi-badge">📈 Prévision</span>',
        f'<span class="kpi-badge {css}">{icon} Fiabilité {reliability}</span>',
    ]
    if folds:
        badges.append(
            f'<span class="kpi-badge">🧪 Validée sur {folds} '
            f"fenêtres de {window} mois</span>"
        )
    badges.append('<span class="kpi-badge green">🔒 Requête validée · lecture seule</span>')
    if elapsed is not None:
        seconds = f"{elapsed:.1f}".replace(".", ",")
        badges.append(f'<span class="kpi-badge">⏱ {seconds} s</span>')

    panel["kpi"] = (
        f'<div class="metrics">{cards}</div><div>{"".join(badges)}</div>'
    )

    # ---- Message métier ----
    dates = pd.to_datetime(forecast[date_col])
    period = (
        f"{month_label(dates.min())} à {month_label(dates.max())}"
        if count > 1
        else month_label(dates.iloc[0])
    )

    headline = (
        f"**{format_value(total, unit, compact=True)}** prévus sur {count} mois"
    )
    if variation_text == "stable":
        headline += ", soit un niveau **stable** par rapport aux mêmes mois un an plus tôt"
    elif variation_text:
        headline += (
            f", soit **{variation_text}** par rapport aux mêmes mois "
            "un an plus tôt"
        )

    peak = forecast.loc[forecast[value_col].idxmax()]
    hist_dates = pd.to_datetime(fr.historical[fr.date_column])

    lines = [
        f"📈 **Prévision — {metric}** · {period}",
        f"{headline}.",
        f"Mois le plus fort : **{month_label(peak[date_col])}** "
        f"({format_value(peak[value_col], unit, compact=True)}).",
        f"{icon} **Fiabilité {reliability}**",
    ]

    if reason:
        lines.append(f"🧠 {reason}")

    if reliability == "faible":
        lines.append("Prenez ces chiffres avec prudence.")

    lines.append(
        f"Historique utilisé : {month_label(hist_dates.min())} → "
        f"{month_label(hist_dates.max())} ({len(hist_dates)} mois)."
    )

    message = "\n\n".join(lines)

    if notes:
        message += "\n\n" + "\n".join(f"- ⚠️ {n}" for n in notes)

    message += (
        "\n\n> Détail mois par mois dans l'onglet **Résultats**, "
        "graphique dans **Visualisation**."
    )

    # ---- Tableau, exports, graphique ----
    table = clean_dataframe(forecast_table(fr))

    panel["table"] = gr.update(value=style_dataframe(table), visible=True)
    panel["download"] = gr.update(value=export_csv(table), visible=True)

    excel_path = export_excel(table)
    panel["excel"] = gr.update(value=excel_path, visible=bool(excel_path))

    chart_path = make_forecast_chart(fr)

    if chart_path:
        panel["chart"] = gr.update(
            value=str(chart_path),
            visible=True,
        )

        panel["chart_download"] = gr.update(
            value=str(chart_path),
            visible=True,
        )

    # ---- Traçabilité : modèles comparés ----
    trace = build_trace(result, elapsed)

    ranking = getattr(fr, "candidates", None) or []
    if ranking:
        rows = "\n".join(
            f"| {r['model']} | {format_value(r['MAE'], unit)} | "
            f"{r['MAPE']:.1f} % |".replace(".", ",")
            for r in ranking
        )
        trace += (
            "\n\n**Modèles comparés (backtest)**\n\n"
            "| Modèle | Erreur moyenne (MAE) | MAPE |\n"
            "|---|---:|---:|\n" + rows
        )

    panel["trace"] = trace

    # ---- Avertissements ----
    if notes:
        panel["warnings"] = "\n".join(f"- ⚠️ {n}" for n in notes)

    # ---- Pour aller plus loin ----
    follow_ups = [
        ["Prévois le chiffre d'affaires pour les 12 prochains mois"],
        ["Chiffre d'affaires par mois"],
        ["Chiffre d'affaires par région"],
    ]
    panel["suggestions"] = gr.update(samples=follow_ups, visible=True)

    return message, panel


# ---------------------------------------------------------
# Construction des sorties
# ---------------------------------------------------------

# Ordre des composants du panneau (doit correspondre à build_app)
PANEL_KEYS = (
    "kpi", "table", "chart", "chart_download",
    "sql", "trace", "warnings", "download",
    "excel", "suggestions",
)


def empty_panel() -> dict[str, Any]:
    """État du panneau lorsqu'il n'y a rien à afficher."""
    hidden = gr.update(value=None, visible=False)
    return {
        "kpi": "",
        "table": hidden,
        "chart": hidden,
        "chart_download": hidden,
        "sql": "",
        "trace": "",
        "warnings": "",
        "download": hidden,
        "excel": hidden,
        "suggestions": gr.update(samples=[], visible=False),
    }


def panel_list(panel: dict[str, Any]) -> list[Any]:
    return [panel[key] for key in PANEL_KEYS]


def extract_requested_year(question: str) -> int | None:
    """Extrait une année explicite présente dans la question."""
    match = re.search(r"\b((?:19|20)\d{2})\b", question)
    return int(match.group(1)) if match else None


def build_no_data_message(result) -> str:
    """Construit un message naturel lorsqu'aucune donnée n'est trouvée."""
    question = getattr(result, "question", "") or ""

    requested_year = extract_requested_year(question)
    max_year = getattr(result, "data_max_year", None)

    if requested_year is not None and max_year is not None and requested_year > max_year:
        return (
            f"Je n’ai pas trouvé d’information concernant votre demande "
            f"pour **{requested_year}**, car les données disponibles "
            f"s’arrêtent en **{max_year}**."
        )

    if requested_year is not None:
        return (
            f"Je n’ai trouvé aucune information correspondant à votre "
            f"demande pour **{requested_year}**."
        )

    return (
        "Je n’ai trouvé aucune information correspondant à votre demande."
        "\n\nVous pouvez essayer avec une autre période ou un autre critère."
    )


def build_reply(
    result,
    elapsed: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
    """Transforme le résultat du pipeline en réponse naturelle + panneau."""


    panel = empty_panel()

    # ---------------------------------------------------------
    # SQL
    # ---------------------------------------------------------
    sql = getattr(result, "sql", None)

    if sql:
        panel["sql"] = sql

    # ---------------------------------------------------------
    # Clarification
    # ---------------------------------------------------------
    if result.status == "clarification_needed":
        question = (
            result.clarification_question
            or "Pouvez-vous préciser votre demande ?"
        )

        return (
            f"🤔 **Pour être sûr de bien vous répondre, "
            f"j’ai besoin d’une précision.**\n\n{question}",
            panel,
        )

    # ---------------------------------------------------------
    # SQL refusé
    # ---------------------------------------------------------
    if result.status == "validation_error":
        return (
            "Je n’ai pas pu traiter cette demande de manière fiable. "
            "Pouvez-vous reformuler votre question ou préciser "
            "les informations que vous recherchez ?",
            panel,
        )

    # ---------------------------------------------------------
    # Erreur métier
    # ---------------------------------------------------------
    if result.status == "error":
        detail = (
            getattr(result, "errors", None)
            or [""]
        )[0]

        message = "Je n’ai pas pu traiter cette demande."

        if detail:
            message += f"\n\n{detail}"

        return message, panel

    # ---------------------------------------------------------
    # État inattendu
    # ---------------------------------------------------------
    if result.status != "ok":
        logger.warning(
            "État inattendu retourné par le pipeline : %s",
            result.status,
        )

        return (
            "Je rencontre un petit problème pour traiter cette demande. "
            "Vous pouvez réessayer dans un instant.",
            panel,
        )

    # ---------------------------------------------------------
    # Avertissements
    # ---------------------------------------------------------
    if result.warnings:
        panel["warnings"] = "\n".join(
            f"- ⚠️ {w}"
            for w in result.warnings
        )

    # ---------------------------------------------------------
    # Prévision
    # ---------------------------------------------------------
    if getattr(result, "forecast_result", None) is not None:
        return build_forecast_reply(
            result,
            panel,
            elapsed,
        )

    # ---------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------
    dataframe = result.dataframe

    if dataframe is not None and not dataframe.empty:
        dataframe = dataframe.dropna(how="all")

    # ---------------------------------------------------------
    # Aucun résultat
    # ---------------------------------------------------------
    if dataframe is None or dataframe.empty:
        return build_no_data_message(result), panel

    # ---------------------------------------------------------
    # Nettoyage
    # ---------------------------------------------------------
    df = clean_dataframe(dataframe)

    rows = len(df)

    question = (
        getattr(result, "question", "")
        or ""
    )

    # ---------------------------------------------------------
    # Réponse principale
    # ---------------------------------------------------------
    answer = build_answer(
        question,
        df,
    )

    message = (
        answer
        or "Voici ce que j’ai trouvé."
    )

    # ---------------------------------------------------------
    # Insights
    # ---------------------------------------------------------
    insights = (
        build_insights(question, df)
        if rows > 1
        else []
    )

    if insights:
        message += (
            "\n\n**📌 L'essentiel**\n"
            + "\n".join(
                f"- {line}"
                for line in insights
            )
        )

    # ---------------------------------------------------------
    # Nombre de résultats
    # ---------------------------------------------------------
    if rows > 1 or not answer:
        message += (
            f"\n\n> {plural(rows, 'résultat')} — "
            "détail dans l'onglet **Résultats**."
        )

    # ---------------------------------------------------------
    # Date de mise à jour
    # ---------------------------------------------------------
    last_updated = getattr(
        result,
        "last_updated",
        None,
    )

    # ---------------------------------------------------------
    # KPI
    # ---------------------------------------------------------
    panel["kpi"] = build_kpi_html(
        df,
        elapsed,
        question,
        last_updated=last_updated,
    )

    # ---------------------------------------------------------
    # Tableau
    # ---------------------------------------------------------

    # Version destinée uniquement à l'affichage utilisateur
    display_df = prepare_dataframe_for_display(df)

    panel["table"] = gr.update(
        value=style_dataframe(display_df),
        visible=True,
    )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------
    panel["download"] = gr.update(
        value=export_csv(display_df),
        visible=True,
    )

    # ---------------------------------------------------------
    # Excel
    # ---------------------------------------------------------
    excel_path = export_excel(display_df)

    panel["excel"] = gr.update(
        value=excel_path,
        visible=bool(excel_path),
    )

    # ---------------------------------------------------------
    # Graphique interactif
    # ---------------------------------------------------------
    chart = make_interactive_chart(df)

    if chart is not None:

        panel["chart"] = gr.update(
            value=chart,
            visible=True,
        )

    else:

        panel["chart"] = gr.update(
            value=None,
            visible=False,
        )

    # Pas de téléchargement pour le moment
    panel["chart_download"] = gr.update(
        value=None,
        visible=False,
    )

    # ---------------------------------------------------------
    # Traçabilité
    # ---------------------------------------------------------
    panel["trace"] = build_trace(
        result,
        elapsed,
    )

    # ---------------------------------------------------------
    # Suggestions
    # ---------------------------------------------------------
    suggestions = build_suggestions(df)

    panel["suggestions"] = gr.update(
        samples=suggestions,
        visible=bool(suggestions),
    )

    return message, panel






# ---------------------------------------------------------


def format_last_updated(value: Any) -> str | None:
    """Formate la date de dernière actualisation des données."""
    if not value:
        return None

    try:
        timestamp = pd.Timestamp(value)

        return (
            f"{timestamp.day:02d}/"
            f"{timestamp.month:02d}/"
            f"{timestamp.year} à "
            f"{timestamp.hour:02d}:"
            f"{timestamp.minute:02d}"
        )

    except Exception:
        return str(value)


#..........................................
# Callbacks
# ---------------------------------------------------------

def contextualize(message: str, chat: list[dict]) -> str:
    """
    Rattache une réponse courte à la question précédente quand
    l'assistant venait de demander une précision.

        user : "Chiffre d'affaires"            (question trop vague)
        bot  : "j'ai besoin d'une précision : par mois ou par région ?"
        user : "par région"   ->  "Chiffre d'affaires (par région)"
    """
    if len(message.split()) > 8:
        return message

    last_assistant = next(
        (m["content"] for m in reversed(chat) if m["role"] == "assistant"),
        "",
    )
    if "besoin d’une précision" not in last_assistant:
        return message

    last_user = next(
        (m["content"] for m in reversed(chat) if m["role"] == "user"),
        None,
    )
    return f"{last_user} ({message})" if last_user else message


def handle_question(message: str, chat: list[dict], questions: list[str]):
    """Traite une question. Générateur : affiche un état d'attente d'abord."""
    message = (message or "").strip()

    def outputs(chat_value, panel, questions_value):
        return (
            chat_value,
            "",
            *panel_list(panel),
            questions_value,
            gr.update(samples=[[q] for q in reversed(questions_value[-8:])]),
        )

    if not message:
        gr.Warning("Saisissez une question avant d'envoyer.")
        yield outputs(chat, empty_panel(), questions)
        return

    # Texte envoyé au pipeline (enrichi si c'est une réponse courte à
    # une demande de précision) ; le chat affiche le message d'origine.
    question_for_pipeline = contextualize(message, chat)

    chat = chat + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "🔎 Je regarde vos données…"},
    ]
    yield outputs(chat, empty_panel(), questions)

    try:
        start = time.perf_counter()
        result = get_pipeline().run(question_for_pipeline)
        elapsed = time.perf_counter() - start
        reply, panel = build_reply(result, elapsed)
    except Exception:
        logger.exception("Erreur pendant l'exécution du pipeline")
        reply = (
            "Je rencontre actuellement un problème pour récupérer "
            "ces informations. Vous pouvez réessayer dans un instant."
        )
        panel = empty_panel()

    chat[-1] = {"role": "assistant", "content": reply}

    if message not in questions:
        questions = questions + [message]

    yield outputs(chat, panel, questions)


def reset_all():
    return (
        [{"role": "assistant", "content": WELCOME}],
        "",
        *panel_list(empty_panel()),
    )


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------


def prepare_dataframe_for_display(df):
    """
    Prépare le DataFrame pour l'affichage utilisateur.
    Les noms techniques sont remplacés par des noms naturels.
    """

    if df is None:
        return df

    df = df.copy()

    # Colonnes techniques inutiles pour l'utilisateur
    columns_to_hide = [
        "Nom_Mois",
    ]

    df = df.drop(
        columns=[
            col
            for col in columns_to_hide
            if col in df.columns
        ],
        errors="ignore",
    )

    # Noms affichés à l'utilisateur
    column_labels = {
        "Annee": "Année",
        "Mois": "Mois",
        "Chiffre_Affaires": "Chiffre d’affaires",
    }

    df = df.rename(
        columns=column_labels
    )

    return df






#.................

def _sidebar():
    """Menu latéral à droite (repli sur la gauche si la version de Gradio est ancienne)."""
    try:
        return gr.Sidebar(position="right", open=True, width=300)
    except TypeError:
        return gr.Sidebar(open=True, width=300)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Assistant Data AI") as app:
        questions_state = gr.State([])

        # =========================================================
        # EN-TÊTE
        # =========================================================
        with gr.Row(
            elem_id="entete",
            equal_height=True,
        ):

            # -----------------------------------------------------
            # GAUCHE : LOGO + TITRE + SOUS-TITRE
            # -----------------------------------------------------
            with gr.Row(
                elem_id="entete-gauche",
                scale=7,
                equal_height=True,
            ):

                if HAS_LOGO:
                    gr.Image(
                        value=str(LOGO_PATH),
                        height=72,
                        width=72,
                        show_label=False,
                        container=False,
                        interactive=False,
                        buttons=[],
                        scale=0,
                        min_width=72,
                        elem_id="logo",
                    )

                gr.Markdown(
                    "# Assistant Data AI\n"
                    "Interrogez vos données en langage naturel.",
                    elem_id="titre",
                )

            # -----------------------------------------------------
            # DROITE : KPI SUR UNE SEULE LIGNE
            # -----------------------------------------------------
            kpi = gr.HTML(
                elem_id="kpi",
                scale=5,
            )

        # =========================================================
        # CONTENU PRINCIPAL
        # =========================================================
        with gr.Row(
            elem_id="workspace",
            equal_height=False,
        ):

            # =====================================================
            # COLONNE GAUCHE : CONVERSATION + SQL
            # =====================================================
            with gr.Column(
                scale=4,
                min_width=320,
                elem_id="chat-column",
            ):

                # -------------------------------------------------
                # CHATBOT
                # -------------------------------------------------
                chatbot = gr.Chatbot(
                    value=[
                        {
                            "role": "assistant",
                            "content": WELCOME,
                        }
                    ],
                    height="52vh",
                    show_label=False,
                    render_markdown=True,
                    avatar_images=(
                        None,
                        str(LOGO_PATH) if HAS_LOGO else None,
                    ),
                    elem_id="chatbot",
                )

                # -------------------------------------------------
                # SAISIE
                # -------------------------------------------------
                with gr.Row(elem_id="saisie"):

                    msg = gr.Textbox(
                        placeholder=(
                            "Ex. : Quel est le chiffre "
                            "d'affaires par région ?"
                        ),
                        show_label=False,
                        container=False,
                        lines=2,
                        max_lines=4,
                        autofocus=True,
                        scale=6,
                    )

                    send_btn = gr.Button(
                        "Analyser",
                        variant="primary",
                        scale=2,
                        min_width=110,
                    )

                    clear_btn = gr.Button(
                        "↺ Nouveau",
                        scale=1,
                        min_width=90,
                    )

                # -------------------------------------------------
                # REQUÊTE SQL
                # Toujours visible sous le chatbot
                # -------------------------------------------------
                gr.Markdown(
                    "### Requête SQL",
                    elem_id="titre-sql",
                )

                sql_box = gr.Code(
                    language="sql",
                    interactive=False,
                    show_label=False,
                    elem_id="sql-box",
                )

            # =====================================================
            # COLONNE DROITE : GRAPHIQUE + RÉSULTATS
            # =====================================================
            with gr.Column(
                scale=8,
                min_width=600,
                elem_id="results-column",
            ):

                # -------------------------------------------------
                # HISTORIQUE
                #
                # Le composant reste géré par _sidebar().
                # -------------------------------------------------

                # -------------------------------------------------
                # GRAPHIQUE
                # -------------------------------------------------
                with gr.Group(
                    elem_id="graphique-principal",
                ):

                    gr.Markdown(
                        "### Visualisation",
                        elem_id="titre-graphique",
                    )

                    with gr.Row(
                        elem_id="chart-row",
                        equal_height=True,
                    ):

                        chart = gr.Plot(
                            visible=False,
                            show_label=False,
                            elem_id="chart",
                            scale=12,
                            min_width=400,
                        )

                        chart_download = gr.DownloadButton(
                            "⬇",
                            variant="secondary",
                            visible=False,
                            scale=0,
                            min_width=42,
                            elem_id="chart-download",
                        )

                # -------------------------------------------------
                # TABLEAU RÉSULTATS
                #
                # Directement sous le graphique
                # -------------------------------------------------
                table = gr.Dataframe(
                    visible=False,
                    interactive=False,
                    wrap=True,
                    max_height="38vh",
                    elem_id="table",
                    show_label=False,
                )

                # -------------------------------------------------
                # TÉLÉCHARGEMENTS
                # -------------------------------------------------
                with gr.Row(
                    elem_id="download-row",
                ):

                    download = gr.DownloadButton(
                        "Télécharger en CSV",
                        variant="primary",
                        visible=False,
                    )

                    excel = gr.DownloadButton(
                        "Télécharger en Excel",
                        variant="secondary",
                        visible=False,
                    )

        # =========================================================
        # INFORMATIONS TECHNIQUES
        #
        # Seuls Traçabilité et Avertissements restent en onglets.
        # Ils sont donc séparés du tableau et de la SQL.
        # =========================================================
        with gr.Tabs(
            elem_id="technical-tabs",
        ):


            with gr.Tab("Avertissements"):

                warnings_md = gr.Markdown(
                visible=True,
                )

            with gr.Tab("Traçabilité"):

                trace_md = gr.Markdown(
                    visible=True,
                )



        # =========================================================
        # MENU LATÉRAL
        # =========================================================
        with _sidebar():

            gr.Markdown("### 💡 Assistant")

            examples_ds = gr.Dataset(
                components=[
                    gr.Textbox(visible=False)
                ],
                samples=[
                    [example]
                    for example in EXAMPLES
                ],
                label="Exemples de questions",
                type="values",
            )

            suggest_ds = gr.Dataset(
                components=[
                    gr.Textbox(visible=False)
                ],
                samples=[],
                label="Pour aller plus loin",
                type="values",
                visible=False,
            )

            history_ds = gr.Dataset(
                components=[
                    gr.Textbox(visible=False)
                ],
                samples=[],
                label="Historique récent",
                type="values",
            )

        # =========================================================
        # COMPOSANTS DU PANEL
        #
        # L'ordre DOIT rester identique à PANEL_KEYS.
        # =========================================================
        panel_components = [
            kpi,
            table,
            chart,
            chart_download,
            sql_box,
            trace_md,
            warnings_md,
            download,
            excel,
            suggest_ds,
        ]

        # =========================================================
        # ÉVÉNEMENTS
        # =========================================================
        submit_inputs = [
            msg,
            chatbot,
            questions_state,
        ]

        submit_outputs = [
            chatbot,
            msg,
            *panel_components,
            questions_state,
            history_ds,
        ]

        # ---------------------------------------------------------
        # ANALYSER
        # ---------------------------------------------------------
        send_btn.click(
            handle_question,
            submit_inputs,
            submit_outputs,
        )

        # ---------------------------------------------------------
        # ENTRÉE CLAVIER
        # ---------------------------------------------------------
        msg.submit(
            handle_question,
            submit_inputs,
            submit_outputs,
        )

        # ---------------------------------------------------------
        # NOUVEAU
        # ---------------------------------------------------------
        clear_btn.click(
            reset_all,
            None,
            [
                chatbot,
                msg,
                *panel_components,
            ],
        )

        # =========================================================
        # EXEMPLES
        # =========================================================
        examples_ds.click(
            lambda row: row[0],
            examples_ds,
            msg,
        )

        # =========================================================
        # HISTORIQUE
        # =========================================================
        history_ds.click(
            lambda row: row[0],
            history_ds,
            msg,
        )

        # =========================================================
        # SUGGESTIONS
        # =========================================================
        suggest_ds.click(
            lambda row: row[0],
            suggest_ds,
            msg,
        ).then(
            handle_question,
            submit_inputs,
            submit_outputs,
        )

    return app



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Préchauffage : connexion SQL, client OpenAI et chargement du catalogue
    # se font au démarrage plutôt qu'à la première question.
    try:
        get_pipeline()
    except Exception:
        logger.exception("Préchauffage du pipeline impossible")

    build_app().queue().launch(
        favicon_path=str(LOGO_PATH) if HAS_LOGO else None,
        allowed_paths=[str(IMAGE_DIR)],
        theme=THEME,
        css=CSS,
        js=FORCE_LIGHT_JS,
    )