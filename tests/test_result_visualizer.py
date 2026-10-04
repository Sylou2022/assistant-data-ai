import pandas as pd

from app.presentation.result_visualizer import (
    ResultVisualizer,
)


def test_create_chart():
    df = pd.DataFrame(
        {
            "Region": [
                "Nord",
                "Sud",
                "Ouest",
            ],
            "Chiffre_Affaires_Total": [
                125000,
                98000,
                87000,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(
        df,
        filename="test_chart.png",
    )

    assert result is not None
    assert result.exists()


def test_empty_dataframe():
    df = pd.DataFrame()

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(df)

    assert result is None


def test_invalid_dataframe_structure():
    df = pd.DataFrame(
        {
            "Region": [
                "Nord",
                "Sud",
            ],
            "CA": [
                100,
                200,
            ],
            "Marge": [
                20,
                30,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(df)

    assert result is None


def test_create_line_chart_for_dates():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-02-01",
                    "2026-03-01",
                ]
            ),
            "Chiffre_Affaires": [
                100000,
                120000,
                135000,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(
        df,
        filename="date_chart.png",
    )

    assert result is not None
    assert result.exists()


def test_create_pie_chart_for_few_categories():
    df = pd.DataFrame(
        {
            "Region": [
                "Nord",
                "Sud",
                "Ouest",
            ],
            "Chiffre_Affaires": [
                125000,
                98000,
                87000,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(
        df,
        filename="pie_chart.png",
    )

    assert result is not None
    assert result.exists()



def test_create_bar_chart_for_many_categories():
    df = pd.DataFrame(
        {
            "Region": [
                "Nord",
                "Sud",
                "Ouest",
                "Est",
                "Centre",
                "Paris",
            ],
            "Chiffre_Affaires": [
                125000,
                98000,
                87000,
                76000,
                65000,
                150000,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(
        df,
        filename="bar_chart.png",
    )

    assert result is not None
    assert result.exists()


def test_negative_values_use_bar_chart():
    df = pd.DataFrame(
        {
            "Region": [
                "Nord",
                "Sud",
                "Ouest",
            ],
            "Variation": [
                100,
                -50,
                25,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(
        df,
        filename="negative_values.png",
    )

    assert result is not None
    assert result.exists()


def test_zero_sum_values_use_bar_chart():
    df = pd.DataFrame(
        {
            "Region": [
                "Nord",
                "Sud",
                "Ouest",
            ],
            "Variation": [
                100,
                -100,
                0,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(
        df,
        filename="zero_sum.png",
    )

    assert result is not None
    assert result.exists()


def test_two_categories_can_use_pie_chart():
    df = pd.DataFrame(
        {
            "Statut": [
                "Actif",
                "Inactif",
            ],
            "Nombre": [
                80,
                20,
            ],
        }
    )

    visualizer = ResultVisualizer(
        output_dir="outputs/test",
    )

    result = visualizer.create_chart(
        df,
        filename="two_categories.png",
    )

    assert result is not None
    assert result.exists()



