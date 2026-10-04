import pandas as pd
import pytest

from app.presentation.result_formatter import ResultFormatter


def test_format_dataframe():
    df = pd.DataFrame(
        {
            "Region": ["Nord", "Sud"],
            "Chiffre_Affaires_Total": [
                125000,
                98000,
            ],
        }
    )

    formatter = ResultFormatter()

    result = formatter.format(df)

    assert "2 ligne(s) retournée(s)." in result
    assert "Region" in result
    assert "Chiffre_Affaires_Total" in result
    assert "Nord" in result
    assert "Sud" in result


def test_format_empty_dataframe():
    df = pd.DataFrame()

    formatter = ResultFormatter()

    result = formatter.format(df)

    assert result == "Aucun résultat."


def test_format_limits_displayed_rows():
    df = pd.DataFrame(
        {
            "Produit": [
                "Produit A",
                "Produit B",
                "Produit C",
            ],
            "Marge_Totale": [
                100,
                200,
                300,
            ],
        }
    )

    formatter = ResultFormatter(
        max_display_rows=2,
    )

    result = formatter.format(df)

    assert "Produit A" in result
    assert "Produit B" in result
    assert "Produit C" not in result
    assert "Affichage limité aux 2 premières lignes sur 3." in result


def test_invalid_max_display_rows():
    with pytest.raises(ValueError):
        ResultFormatter(
            max_display_rows=0,
        )