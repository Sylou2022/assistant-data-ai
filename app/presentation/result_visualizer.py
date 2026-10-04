# code : presentation/result_visualizer.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Backend sans fenêtre : indispensable avec Gradio (le graphique est
# créé dans un thread du serveur, sans écran).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

if TYPE_CHECKING:
    from app.ml.forecasting import ForecastResult


class ResultVisualizer:
    """
    Génère les graphiques à partir des résultats SQL ou ML.

    Cas supportés :
        - DataFrame SQL classique
        - ForecastResult contenant :
            * historical
            * forecast (avec éventuellement Lower / Upper)
            * metrics
    """

    def __init__(
        self,
        output_dir: str | Path = "outputs",
    ) -> None:
        self.output_dir = Path(output_dir)

    # ==========================================================
    # API PUBLIQUE
    # ==========================================================

    def create_chart(
        self,
        df: pd.DataFrame,
        filename: str = "result.png",
    ) -> Path | None:
        """
        Crée automatiquement un graphique à partir d'un DataFrame SQL.
        """

        if df is None or df.empty:
            return None

        if len(df.columns) != 2:
            return None

        category_column = df.columns[0]
        value_column = df.columns[1]

        if not pd.api.types.is_numeric_dtype(
            df[value_column]
        ):
            return None

        chart_type = self._get_chart_type(
            df,
            category_column,
            value_column,
        )

        if chart_type is None:
            return None

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self.output_dir / filename

        if chart_type == "line":
            self._create_line_chart(
                df,
                category_column,
                value_column,
                output_path,
            )

        elif chart_type == "pie":
            self._create_pie_chart(
                df,
                category_column,
                value_column,
                output_path,
            )

        else:
            self._create_bar_chart(
                df,
                category_column,
                value_column,
                output_path,
            )

        return output_path

    def create_forecast_chart(
        self,
        forecast_result: ForecastResult,
        filename: str = "forecast.png",
    ) -> Path | None:
        """
        Crée un graphique combinant :

            historique
                  +
            prévisions futures
                  +
            intervalle de confiance (si disponible)

        ForecastResult doit exposer :

            historical : DataFrame
            forecast   : DataFrame
        """

        if forecast_result is None:
            return None

        historical = getattr(
            forecast_result,
            "historical",
            None,
        )

        forecast = getattr(
            forecast_result,
            "forecast",
            None,
        )

        if not isinstance(
            historical,
            pd.DataFrame,
        ):
            return None

        if not isinstance(
            forecast,
            pd.DataFrame,
        ):
            return None

        if historical.empty and forecast.empty:
            return None

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self.output_dir / filename

        self._create_forecast_line_chart(
            historical=historical,
            forecast=forecast,
            output_path=output_path,
        )

        return output_path

    # ==========================================================
    # DÉTECTION DU TYPE DE GRAPHIQUE
    # ==========================================================

    @staticmethod
    def _get_chart_type(
        df: pd.DataFrame,
        category_column: str,
        value_column: str,
    ) -> str | None:
        series = df[category_column]
        values = df[value_column]

        if ResultVisualizer._is_date_column(series):
            return "line"

        if (
            2 <= len(df) <= 5
            and (values >= 0).all()
            and values.sum() > 0
        ):
            return "pie"

        return "bar"

    @staticmethod
    def _is_date_column(
        series: pd.Series,
    ) -> bool:
        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            return True

        if not pd.api.types.is_object_dtype(
            series
        ):
            return False

        converted = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        )

        return converted.notna().all()

    # ==========================================================
    # GRAPHIQUE BARRES
    # ==========================================================

    @staticmethod
    def _create_bar_chart(
        df: pd.DataFrame,
        category_column: str,
        value_column: str,
        output_path: Path,
    ) -> None:
        plt.figure(figsize=(8, 5))

        plt.bar(
            df[category_column].astype(str),
            df[value_column],
        )

        plt.xlabel(
            str(category_column)
        )

        plt.ylabel(
            str(value_column)
        )

        plt.title(
            f"{value_column} par {category_column}"
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

    # ==========================================================
    # GRAPHIQUE COURBE
    # ==========================================================

    @staticmethod
    def _create_line_chart(
        df: pd.DataFrame,
        category_column: str,
        value_column: str,
        output_path: Path,
    ) -> None:
        dates = pd.to_datetime(
            df[category_column],
            errors="coerce",
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            dates,
            df[value_column],
            marker="o",
        )

        plt.xlabel(
            str(category_column)
        )

        plt.ylabel(
            str(value_column)
        )

        plt.title(
            f"{value_column} dans le temps"
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

    # ==========================================================
    # GRAPHIQUE CAMEMBERT
    # ==========================================================

    @staticmethod
    def _create_pie_chart(
        df: pd.DataFrame,
        category_column: str,
        value_column: str,
        output_path: Path,
    ) -> None:
        plt.figure(figsize=(7, 7))

        plt.pie(
            df[value_column],
            labels=df[category_column].astype(str),
            autopct="%1.1f%%",
        )

        plt.title(
            f"{value_column} par {category_column}"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

    # ==========================================================
    # GRAPHIQUE FORECASTING
    # ==========================================================

    @staticmethod
    def _create_forecast_line_chart(
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """
        Historique (ligne pleine) + prévision (pointillés)
        + intervalle de confiance (zone colorée) si disponible.
        """

        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter

        historical_df = historical.copy()
        forecast_df = forecast.copy()

        historical_date = ResultVisualizer._find_date_column(historical_df)
        forecast_date = ResultVisualizer._find_date_column(forecast_df)

        historical_value = ResultVisualizer._find_value_column(historical_df)
        forecast_value = ResultVisualizer._find_forecast_value_column(
            forecast_df
        )

        if historical_date is None or historical_value is None:
            return

        if not forecast_df.empty and (
            forecast_date is None or forecast_value is None
        ):
            return

        # -- Nettoyage ------------------------------------------------

        historical_df[historical_date] = pd.to_datetime(
            historical_df[historical_date], errors="coerce"
        )
        historical_df = historical_df.dropna(
            subset=[historical_date, historical_value]
        ).sort_values(historical_date)

        if not forecast_df.empty:
            forecast_df[forecast_date] = pd.to_datetime(
                forecast_df[forecast_date], errors="coerce"
            )
            forecast_df = forecast_df.dropna(
                subset=[forecast_date, forecast_value]
            ).sort_values(forecast_date)

        if historical_df.empty and forecast_df.empty:
            return

        history_color = "#4f46e5"
        forecast_color = "#ea580c"

        fig, ax = plt.subplots(figsize=(10, 5.5))

        try:
            # -- Zone de prévision -----------------------------------

            if not historical_df.empty and not forecast_df.empty:
                last_hist = historical_df[historical_date].max()

                ax.axvspan(
                    last_hist,
                    forecast_df[forecast_date].max(),
                    color="#fff7ed",
                    alpha=0.8,
                    zorder=0,
                )
                ax.axvline(
                    last_hist,
                    linestyle=":",
                    color="#9ca3af",
                    linewidth=1.2,
                )

            # -- Historique ------------------------------------------

            if not historical_df.empty:
                ax.plot(
                    historical_df[historical_date],
                    historical_df[historical_value],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=history_color,
                    label="Historique",
                )

            # -- Prévision -------------------------------------------

            if not forecast_df.empty:

                # Raccord entre le dernier point connu et la prévision
                if not historical_df.empty:
                    ax.plot(
                        [
                            historical_df[historical_date].iloc[-1],
                            forecast_df[forecast_date].iloc[0],
                        ],
                        [
                            historical_df[historical_value].iloc[-1],
                            forecast_df[forecast_value].iloc[0],
                        ],
                        linestyle="--",
                        linewidth=1.5,
                        color=forecast_color,
                    )

                # Intervalle de confiance
                if {"Lower", "Upper"} <= set(forecast_df.columns):
                    band = forecast_df.dropna(subset=["Lower", "Upper"])

                    if not band.empty:
                        ax.fill_between(
                            band[forecast_date],
                            band["Lower"],
                            band["Upper"],
                            color=forecast_color,
                            alpha=0.18,
                            linewidth=0,
                            label="Intervalle à 80 %",
                        )

                ax.plot(
                    forecast_df[forecast_date],
                    forecast_df[forecast_value],
                    marker="o",
                    markersize=4,
                    linestyle="--",
                    linewidth=2,
                    color=forecast_color,
                    label="Prévision",
                )

            # -- Axes ------------------------------------------------

            name = (
                str(historical_value)
                .replace("_", " ")
                .strip()
                .lower()
                .replace("chiffre affaires", "chiffre d'affaires")
            )
            name = name[:1].upper() + name[1:]

            months = len(historical_df) + len(forecast_df)

            ax.xaxis.set_major_locator(
                mdates.MonthLocator(interval=1)
                if months <= 18
                else mdates.MonthLocator(bymonth=[1, 4, 7, 10])
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
            )

            ax.set_title(
                f"{name} — historique et prévisions",
                loc="left",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel(name)
            ax.grid(axis="y", alpha=0.2)

            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            ax.legend(loc="upper left", frameon=False)

            fig.tight_layout()
            fig.savefig(output_path, dpi=150)

        finally:
            plt.close(fig)

    # ==========================================================
    # UTILITAIRES FORECAST
    # ==========================================================

    @staticmethod
    def _find_date_column(
        df: pd.DataFrame,
    ) -> str | None:
        preferred = (
            "Date",
            "date",
            "DATE",
        )

        for column in preferred:
            if column in df.columns:
                return column

        for column in df.columns:
            normalized = str(
                column
            ).lower()

            if "date" in normalized:
                return column

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                continue

            try:
                converted = pd.to_datetime(
                    df[column],
                    errors="coerce",
                )

                if (
                    not converted.empty
                    and converted.notna().mean() >= 0.8
                ):
                    return column

            except Exception:
                continue

        return None

    @staticmethod
    def _find_value_column(
        df: pd.DataFrame,
    ) -> str | None:
        preferred = (
            "Value",
            "Chiffre_Affaires",
            "Chiffre_Affaires_Total",
            "CA",
            "Revenue",
            "Sales",
        )

        for column in preferred:
            if column in df.columns:
                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):
                    return column

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                return column

        return None

    @staticmethod
    def _find_forecast_value_column(
        df: pd.DataFrame,
    ) -> str | None:
        preferred = (
            "Prediction",
            "prediction",
            "Forecast",
            "forecast",
            "Value",
            "value",
            "Chiffre_Affaires",
        )

        for column in preferred:
            if column in df.columns:
                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):
                    return column

        skip = {"Lower", "Upper"}

        for column in df.columns:
            if column in skip:
                continue

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                return column

        return None