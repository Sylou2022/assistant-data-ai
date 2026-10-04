import streamlit as st

from app.pipeline.data_pipeline import DataPipeline

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Assistant Data AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Initialisation de la session
# ---------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "question" not in st.session_state:
    st.session_state.question = ""

# ---------------------------------------------------------
# Fonctions
# ---------------------------------------------------------


def run_pipeline(question: str):
    """Exécute le pipeline de données."""
    pipeline = DataPipeline()
    return pipeline.run(question)


def display_result(result) -> None:
    """Affiche le résultat du pipeline."""

    # Clarification
    if result.status == "clarification_needed":
        st.warning("🤔 J'ai besoin d'une précision.")

        if result.clarification_question:
            st.info(result.clarification_question)

        return

    # Erreur de validation SQL
    if result.status == "validation_error":
        st.error("🚫 La requête SQL a été rejetée.")

        if result.errors:
            st.subheader("Erreurs détectées")

            for error in result.errors:
                st.error(error)

        if result.sql:
            with st.expander("🔎 Voir le SQL généré"):
                st.code(result.sql, language="sql")

        return

    # État inattendu
    if result.status != "ok":
        st.error(
            "Un état inattendu a été retourné "
            f"par le pipeline : {result.status}"
        )
        return

    # Résultat OK
    st.success("✅ Analyse terminée avec succès.")

    dataframe = result.dataframe

    # KPI
    if dataframe is not None:
        row_count = len(dataframe)
        column_count = len(dataframe.columns)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Lignes",
                f"{row_count:,}".replace(",", " "),
            )

        with col2:
            st.metric("Colonnes", column_count)

    st.divider()

    # Résultats
    st.subheader("📋 Résultats")

    if dataframe is not None and not dataframe.empty:
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = dataframe.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Télécharger les résultats en CSV",
            data=csv_data,
            file_name="resultats.csv",
            mime="text/csv",
        )
    else:
        st.info("La requête n'a retourné aucun résultat.")

    # Graphique
    if result.chart_path is not None:
        st.divider()
        st.subheader("📈 Visualisation")
        st.image(
            str(result.chart_path),
            use_container_width=True,
        )

    # SQL
    if result.sql:
        st.divider()

        with st.expander("🔎 Voir la requête SQL générée"):
            st.code(result.sql, language="sql")

    # Avertissements
    if result.warnings:
        with st.expander("⚠️ Avertissements"):
            for warning in result.warnings:
                st.warning(warning)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.title("📊 Assistant Data AI")

    st.caption(
        "Posez une question en langage naturel "
        "sur vos données."
    )

    st.divider()

    st.subheader("💡 Exemples")

    examples = [
        "Chiffre d'affaires par région",
        "Chiffre d'affaires par mois",
        "Marge par produit",
        "Quantité vendue par produit",
    ]

    for example in examples:
        if st.button(
            example,
            key=f"example_{example}",
            use_container_width=True,
        ):
            st.session_state.question = example
            st.rerun()

    st.divider()

    st.subheader("🕘 Historique")

    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            if st.button(
                item,
                key=f"history_{item}",
                use_container_width=True,
            ):
                st.session_state.question = item
                st.rerun()
    else:
        st.caption("Vos dernières questions apparaîtront ici.")

# ---------------------------------------------------------
# Page principale
# ---------------------------------------------------------

st.title("Assistant Data AI")

st.markdown(
    """
Posez votre question métier en langage naturel.
L'assistant va générer une requête SQL, la valider,
interroger les données et vous présenter le résultat.
"""
)

st.divider()

# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

question = st.text_area(
    "💬 Votre question",
    value=st.session_state.question,
    placeholder=(
        "Exemple : "
        "Quel est le chiffre d'affaires par région ?"
    ),
    height=100,
)

# ---------------------------------------------------------
# Boutons
# ---------------------------------------------------------

col1, col2 = st.columns([1, 5])

with col1:
    analyze = st.button(
        "🚀 Analyser",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear = st.button(
        "🗑️ Effacer",
        use_container_width=True,
    )

if clear:
    st.session_state.question = ""
    st.rerun()

# ---------------------------------------------------------
# Exécution
# ---------------------------------------------------------

if analyze:
    question = question.strip()

    if not question:
        st.warning("Veuillez saisir une question.")

    else:
        # Sauvegarde dans l'historique
        if question not in st.session_state.history:
            st.session_state.history.append(question)

        with st.status(
            "Analyse en cours...",
            expanded=True,
        ) as status:
            st.write("🧠 Analyse de la question...")

            try:
                result = run_pipeline(question)

                st.write("🔎 Génération et validation du SQL...")
                st.write("🗄️ Interrogation de la base...")

                status.update(
                    label="Analyse terminée",
                    state="complete",
                    expanded=False,
                )

            except Exception as error:
                status.update(
                    label="Erreur pendant l'analyse",
                    state="error",
                    expanded=True,
                )

                st.error(
                    "Une erreur est survenue "
                    "pendant l'exécution."
                )

                with st.expander("Détails techniques"):
                    st.exception(error)

                st.stop()

        display_result(result)