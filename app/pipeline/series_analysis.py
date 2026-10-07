# code : pipeline/series_analysis.py

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MONTHS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)


def period_label(ts: Any) -> str:
    ts = pd.Timestamp(ts)
    return f"{MONTHS_FR[ts.month - 1]} {ts.year}"


def series_from_dataframe(
    df: pd.DataFrame, measure: str | None = None
) -> tuple[pd.Series, str]:
    """
    Série mensuelle (index = 1er du mois) à partir d'un résultat SQL.
    Le résultat doit contenir une colonne Date, ou les colonnes Annee et Mois,
    plus au moins une mesure numérique (une ligne par mois).
    """
    cols = {str(c).lower(): c for c in df.columns}

    if "date" in cols:
        dates = pd.to_datetime(df[cols["date"]], errors="coerce")
    elif "annee" in cols and "mois" in cols:
        dates = pd.to_datetime(
            pd.DataFrame(
                {
                    "year": pd.to_numeric(df[cols["annee"]]).astype(int),
                    "month": pd.to_numeric(df[cols["mois"]]).astype(int),
                    "day": 1,
                }
            )
        )
    else:
        raise ValueError(
            "La requête doit renvoyer une colonne Date, ou les colonnes Annee et Mois, "
            "plus une mesure numérique (une ligne par mois)."
        )

    skip = {cols.get(k) for k in ("date", "annee", "mois")}
    candidates: dict[str, pd.Series] = {}
    for col in df.columns:
        if col in skip:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            candidates[str(col)] = numeric

    if not candidates:
        raise ValueError("Aucune colonne numérique à analyser dans le résultat.")

    if measure and measure.lower() in {k.lower() for k in candidates}:
        name = next(k for k in candidates if k.lower() == measure.lower())
    else:
        name = next(iter(candidates))

    series = pd.Series(candidates[name].to_numpy(dtype=float), index=dates)
    series = series[series.index.notna()]
    series.index = series.index.to_period("M").to_timestamp()
    return series.groupby(level=0).sum().sort_index(), name


def analyze_series(s: pd.Series, name: str = "valeur") -> dict[str, Any]:
    """Faits chiffrés : totaux, évolution, tendance, saisonnalité, anomalies."""
    s = s.dropna().astype(float).sort_index()
    if s.empty:
        return {"erreur": "série vide"}

    full = pd.date_range(s.index[0], s.index[-1], freq="MS")
    missing = full.difference(s.index)

    facts: dict[str, Any] = {
        "indicateur": name,
        "debut": period_label(s.index[0]),
        "fin": period_label(s.index[-1]),
        "nb_mois": int(len(s)),
        "total": round(float(s.sum()), 2),
        "moyenne_mensuelle": round(float(s.mean()), 2),
        "meilleur_mois": {"periode": period_label(s.idxmax()), "valeur": round(float(s.max()), 2)},
        "pire_mois": {"periode": period_label(s.idxmin()), "valeur": round(float(s.min()), 2)},
    }
    if len(missing):
        facts["mois_manquants"] = [period_label(d) for d in missing[:12]]

    # --- Par année et évolution sur années complètes ---
    by_year = s.groupby(s.index.year).agg(["sum", "count"])
    facts["par_annee"] = [
        {"annee": int(y), "total": round(float(r["sum"]), 2), "mois_couverts": int(r["count"])}
        for y, r in by_year.iterrows()
    ]
    complete = by_year.loc[by_year["count"] == 12, "sum"]
    facts["evolution_annuelle_pct"] = [
        {"annee": int(y), "vs": int(y) - 1, "pct": round(float(v / complete[y - 1] - 1) * 100, 1)}
        for y, v in complete.items()
        if (y - 1) in complete.index and complete[y - 1] != 0
    ]
    if len(s) >= 24:
        previous = s.iloc[-24:-12].sum()
        if previous != 0:
            facts["12_derniers_vs_12_precedents_pct"] = round(
                float(s.iloc[-12:].sum() / previous - 1) * 100, 1
            )

    if len(s) < 6:
        return facts

    # --- Saisonnalité (au moins 2 ans, valeurs positives) ---
    pos = np.array(
        [(d.year - full[0].year) * 12 + d.month - full[0].month for d in s.index]
    )
    y = s.to_numpy()
    factor = None
    adjusted = y

    if len(s) >= 24:
        monthly = s.groupby(s.index.month).mean()
        if (monthly > 0).all():
            factor = monthly / monthly.mean()
            ranked = factor.sort_values()
            facts["saisonnalite"] = {
                "mois_forts": [
                    {"mois": MONTHS_FR[int(m) - 1], "indice": round(float(v), 2)}
                    for m, v in ranked.iloc[::-1].head(3).items()
                ],
                "mois_faibles": [
                    {"mois": MONTHS_FR[int(m) - 1], "indice": round(float(v), 2)}
                    for m, v in ranked.head(3).items()
                ],
            }
            adjusted = y / factor.reindex(s.index.month).to_numpy()

    # --- Tendance (série désaisonnalisée) ---
    slope, intercept = np.polyfit(pos, adjusted, 1)
    mean = float(np.mean(np.abs(y)))
    facts["tendance_pct_par_mois"] = round(float(slope) / mean * 100, 2) if mean else None
    facts["volatilite_pct"] = (
        round(float(np.std(y) / abs(np.mean(y))) * 100, 1) if np.mean(y) else None
    )
    trend = slope * pos + intercept

    # --- Anomalies : z-score robuste (médiane / MAD) sur les résidus ---
    resid = adjusted - trend
    median = np.median(resid)
    mad = np.median(np.abs(resid - median))
    anomalies: list[dict[str, Any]] = []

    if mad > 0:
        z = 0.6745 * (resid - median) / mad
        for i in np.argsort(-np.abs(z))[:5]:
            if abs(z[i]) < 3:
                break
            f = float(factor[s.index[i].month]) if factor is not None else 1.0
            expected = float(trend[i]) * f
            anomalies.append(
                {
                    "periode": period_label(s.index[i]),
                    "valeur": round(float(y[i]), 2),
                    "attendu": round(expected, 2),
                    "ecart_pct": round(float(y[i] / expected - 1) * 100, 1) if expected else None,
                    "sens": "hausse" if z[i] > 0 else "baisse",
                }
            )
    facts["anomalies"] = anomalies
    return facts