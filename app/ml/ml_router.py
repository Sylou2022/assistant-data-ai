# code : ml/ml_router.py

"""
ML Router
---------

Décide si une question doit être traitée :

    - par SQL seul         -> AnalysisMode.SQL
    - par SQL + ML         -> AnalysisMode.SQL_ML

Exemples :

    "Quel est le chiffre d'affaires par mois en 2024 ?"
        -> SQL   (on regarde le passé)

    "Prévois le chiffre d'affaires pour les 6 prochains mois."
        -> SQL_ML / FORECASTING / horizon=6 / metric=chiffre_affaires

    "Quelle sera la marge l'année prochaine ?"
        -> SQL_ML / FORECASTING / horizon=12 / metric=marge

Principe : le routage est déterministe (aucun appel LLM), donc instantané
et testable. La question est d'abord normalisée (minuscules, sans accents,
apostrophes unifiées) : « Prévois », « prevois » et « PRÉVOIS » donnent le
même résultat.

Une question est une PRÉVISION seulement si elle contient un signal tourné
vers le futur (prévois, prévision, quel sera, prochains mois, à venir...).
Les mots « par mois », « évolution », « 6 derniers mois » ne déclenchent
jamais une prévision.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------


class AnalysisMode(str, Enum):
    """Mode d'analyse sélectionné par le routeur."""

    SQL = "sql"
    ML = "ml"
    SQL_ML = "sql_ml"


class MLTask(str, Enum):
    """Type de tâche Machine Learning."""

    NONE = "none"
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"


# ---------------------------------------------------------------------
# Routing Decision
# ---------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """
    Résultat du routage.

    mode            SQL ou SQL + ML.
    ml_task         Tâche ML détectée (NONE si question SQL simple).
    confidence      Confiance du routeur entre 0 et 1.
    reason          Explication du choix.
    keywords        Expressions ayant déclenché la détection.
    horizon_months  Horizon demandé pour une prévision (en mois).
    metric          Indicateur à prévoir : "chiffre_affaires", "marge"
                    ou "quantite" (prévisions uniquement).
    """

    mode: AnalysisMode
    ml_task: MLTask = MLTask.NONE
    confidence: float = 1.0
    reason: str = ""
    keywords: list[str] | None = None
    horizon_months: int | None = None
    metric: str | None = None

    def __post_init__(self) -> None:
        if self.keywords is None:
            self.keywords = []


# ---------------------------------------------------------------------
# Motifs de détection (appliqués sur le texte normalisé : sans accents)
# ---------------------------------------------------------------------

FORECASTING_PATTERNS = [
    r"\bprevo(?:i|y)\w*",                 # prevoir, prevois, prevoit, prevoyez
    r"\bprevu(?:e|s|es)?\b",              # prevu, prevue
    r"\bprevision\w*",                    # prevision, previsions, previsionnel
    r"\bpredi\w*",                        # predire, predis, prediction
    r"\bforecast\w*",                     # forecast, forecasting
    r"\bprojet(?:er|te|ons|ez)\b",        # projeter, projette
    r"\bprojections?\b",
    r"\bfutur\w*",                        # futur, future, futurs
    r"\ba venir\b",
    r"\bprochains? mois\b",
    r"\bmois prochains?\b",
    r"\bprochains? (?:trimestres?|semestres?)\b",
    r"\b(?:trimestre|semestre) prochains?\b",
    r"\bannee prochaine\b",
    r"\bprochaine annee\b",
    r"\ban prochain\b",
    r"\bprochain an\b",
    r"\bannees suivantes\b",
    r"\bque(?:l|lle)s? ser(?:a|ont)\b",   # quel sera, quelles seront
    r"\bd'ici\b",
]

ANOMALY_PATTERNS = [
    r"\banomalies?\b",
    r"\banormal\w*",
    r"\binhabituel\w*",
    r"\baberrant\w*",
    r"\boutliers?\b",
]

CLUSTERING_PATTERNS = [
    r"\bsegmentation\b",
    r"\bsegmenter\b",
    r"\bclusters?\b",
    r"\bclustering\b",
    r"\bprofils? clients?\b",
    r"\bregrouper (?:les )?clients\b",
    r"\bidentifier des groupes\b",
]

CLASSIFICATION_PATTERNS = [
    r"\brisque client\b",
    r"\bclients? a risque\b",
    r"\brisquent? de ne plus acheter\b",
    r"\bchurn\b",
    r"\battrition\b",
    r"\bclassification\b",
    r"\bclasser les clients\b",
    r"\bprobabilite de depart\b",
]

METRIC_PATTERNS = (
    ("marge", r"\bmarges?\b"),
    ("quantite", r"\b(?:quantites?|volumes?|unites?)\b"),
)

DEFAULT_METRIC = "chiffre_affaires"

NUMBER_WORDS = {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "six": 6, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11,
    "douze": 12, "quinze": 15, "dix-huit": 18, "dix huit": 18,
    "vingt-quatre": 24, "vingt quatre": 24,
    "trente-six": 36, "trente six": 36,
}

_NUMBER = (
    r"(\d+|"
    + "|".join(
        re.escape(word)
        for word in sorted(NUMBER_WORDS, key=len, reverse=True)
    )
    + r")"
)

# (motif, nombre de mois par unité) : essayés dans cet ordre
HORIZON_NUMERIC_PATTERNS = (
    (rf"\b{_NUMBER}\s+(?:prochains?\s+|suivants?\s+)?mois\b", 1),
    (rf"\b{_NUMBER}\s+(?:prochaines?\s+|suivantes?\s+)?(?:annees?|ans?)\b", 12),
    (rf"\b{_NUMBER}\s+(?:prochains?\s+|suivants?\s+)?trimestres?\b", 3),
    (rf"\b{_NUMBER}\s+(?:prochains?\s+|suivants?\s+)?semestres?\b", 6),
)

HORIZON_SINGLE_PATTERNS = (
    (r"\bmois (?:prochain|suivant)\b|\bprochain mois\b", 1),
    (r"\btrimestre prochain\b|\bprochain trimestre\b", 3),
    (r"\bsemestre prochain\b|\bprochain semestre\b", 6),
    (r"\bannee prochaine\b|\bprochaine annee\b|\ban prochain\b|\bprochain an\b", 12),
)


# ---------------------------------------------------------------------
# ML Router
# ---------------------------------------------------------------------


class MLRouter:
    """Routeur déterministe SQL / SQL + ML."""

    DEFAULT_FORECAST_HORIZON_MONTHS = 6

    # -----------------------------------------------------------------
    # Routage principal
    # -----------------------------------------------------------------

    def route(self, question: str) -> RoutingDecision:

        if not question or not question.strip():
            return RoutingDecision(
                mode=AnalysisMode.SQL,
                ml_task=MLTask.NONE,
                confidence=1.0,
                reason="Question vide : routage SQL par défaut.",
            )

        text = self._normalize(question)

        # 1. Prévision ------------------------------------------------

        keywords = self._find(text, FORECASTING_PATTERNS)

        if keywords:
            return RoutingDecision(
                mode=AnalysisMode.SQL_ML,
                ml_task=MLTask.FORECASTING,
                confidence=0.95,
                reason=(
                    "La question demande une prévision : les données "
                    "historiques sont récupérées en SQL puis transmises "
                    "au modèle de forecasting."
                ),
                keywords=keywords,
                horizon_months=self._extract_horizon(text),
                metric=self._extract_metric(text),
            )

        # 2. Anomalies ------------------------------------------------

        keywords = self._find(text, ANOMALY_PATTERNS)

        if keywords:
            return RoutingDecision(
                mode=AnalysisMode.SQL_ML,
                ml_task=MLTask.ANOMALY_DETECTION,
                confidence=0.95,
                reason=(
                    "La question recherche des valeurs ou comportements "
                    "inhabituels."
                ),
                keywords=keywords,
            )

        # 3. Segmentation ---------------------------------------------

        keywords = self._find(text, CLUSTERING_PATTERNS)

        if keywords:
            return RoutingDecision(
                mode=AnalysisMode.SQL_ML,
                ml_task=MLTask.CLUSTERING,
                confidence=0.90,
                reason=(
                    "La question demande une segmentation ou une "
                    "identification automatique de groupes."
                ),
                keywords=keywords,
            )

        # 4. Classification / risque ----------------------------------

        keywords = self._find(text, CLASSIFICATION_PATTERNS)

        if keywords:
            return RoutingDecision(
                mode=AnalysisMode.SQL_ML,
                ml_task=MLTask.CLASSIFICATION,
                confidence=0.90,
                reason=(
                    "La question demande une prédiction de classe ou "
                    "l'identification d'un risque."
                ),
                keywords=keywords,
            )

        # 5. Question SQL simple --------------------------------------

        return RoutingDecision(
            mode=AnalysisMode.SQL,
            ml_task=MLTask.NONE,
            confidence=0.85,
            reason=(
                "Aucun signal de prévision ou d'analyse ML : la question "
                "porte sur des données existantes, traitée en SQL."
            ),
        )

    # -----------------------------------------------------------------
    # Horizon
    # -----------------------------------------------------------------

    @classmethod
    def _extract_horizon(cls, text: str) -> int:
        """
        Horizon de prévision en mois, à partir du texte normalisé.

            "6 prochains mois" / "six mois" / "sur 6 mois"   -> 6
            "les 2 prochaines années" / "sur 2 ans"          -> 24
            "le trimestre prochain"                           -> 3
            "le mois prochain"                                -> 1
            "l'année prochaine"                               -> 12

        Sans durée explicite : 6 mois.
        """

        for pattern, months_per_unit in HORIZON_NUMERIC_PATTERNS:
            match = re.search(pattern, text)

            if match:
                value = cls._to_int(match.group(1))

                if value > 0:
                    return value * months_per_unit

        for pattern, months in HORIZON_SINGLE_PATTERNS:
            if re.search(pattern, text):
                return months

        return cls.DEFAULT_FORECAST_HORIZON_MONTHS

    @staticmethod
    def _to_int(token: str) -> int:
        if token.isdigit():
            return int(token)

        return NUMBER_WORDS.get(token, 0)

    # -----------------------------------------------------------------
    # Indicateur à prévoir
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_metric(text: str) -> str:
        """marge, quantité ou chiffre d'affaires (par défaut)."""

        for metric, pattern in METRIC_PATTERNS:
            if re.search(pattern, text):
                return metric

        return DEFAULT_METRIC

    # -----------------------------------------------------------------
    # Utilitaires
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Minuscules, sans accents, apostrophes unifiées."""

        text = text.replace("’", "'").replace("‘", "'")
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _find(text: str, patterns: list[str]) -> list[str]:
        """Expressions trouvées dans la question."""

        found: list[str] = []

        for pattern in patterns:
            match = re.search(pattern, text)

            if match:
                found.append(match.group(0))

        return found


# ---------------------------------------------------------------------
# Fonction pratique
# ---------------------------------------------------------------------


def route_question(question: str) -> RoutingDecision:
    """Raccourci : route_question("Prévois le CA") -> RoutingDecision."""

    return MLRouter().route(question)


# ---------------------------------------------------------------------
# Test rapide : python -m app.ml.ml_router
# ---------------------------------------------------------------------

if __name__ == "__main__":

    test_questions = [
        # --- questions simples (doivent rester en SQL) ---
        "Quel est le chiffre d'affaires par mois en 2024 ?",
        "Quelle région a le plus gros chiffre d'affaires ?",
        "Chiffre d'affaires par groupe de produits",
        "Chiffre d'affaires des 6 derniers mois",
        "Montre l'évolution du chiffre d'affaires depuis 2024",
        # --- prévisions ---
        "Prévois le chiffre d'affaires pour les 6 prochains mois.",
        "Prevois le CA sur douze mois",
        "Quelle sera la marge l'année prochaine ?",
        "Prévois la quantité vendue au prochain trimestre",
        "Chiffre d'affaires prévu pour les 2 prochaines années",
        "Quel sera le CA le mois prochain ?",
        # --- autres tâches ML ---
        "Y a-t-il des anomalies dans mes ventes ?",
        "Peux-tu segmenter mes clients ?",
        "Quels clients risquent de ne plus acheter ?",
    ]

    router = MLRouter()

    for question in test_questions:
        decision = router.route(question)

        print("=" * 78)
        print(f"Question : {question}")
        print(
            f"  -> mode={decision.mode.value} | tâche={decision.ml_task.value}"
            f" | horizon={decision.horizon_months} | indicateur={decision.metric}"
        )
        print(f"  mots-clés : {decision.keywords}")