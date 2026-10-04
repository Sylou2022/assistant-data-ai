from __future__ import annotations

from typing import Any


class SemanticContextBuilder:
    def __init__(
        self,
        catalog: dict[str, Any],
        allowed_tables: list[dict[str, Any]] | None = None,
        business_dictionary: list[dict[str, Any]] | None = None,
        kpis: list[dict[str, Any]] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        rules: list[dict[str, Any]] | None = None,
        examples: list[dict[str, Any]] | None = None,
        synonyms: list[dict[str, Any]] | None = None,
    ) -> None:
        self.catalog = catalog
        self.allowed_tables = allowed_tables or []
        self.business_dictionary = business_dictionary or []
        self.kpis = kpis or []
        self.relationships = relationships or []
        self.rules = rules or []
        self.examples = examples or []
        self.synonyms = synonyms or []

    def build(self) -> str:
        sections = [
            self._build_header(),
            self._build_scope(),
            self._build_business_dictionary(),
            self._build_kpis(),
            self._build_schema(),
            self._build_relationships(),
            self._build_synonyms(),
            self._build_rules(),
            self._build_examples(),
        ]

        return "\n\n".join(
            section for section in sections if section
        )

    def _build_header(self) -> str:
        return """============================================================
CONTEXTE SÉMANTIQUE POUR L'ASSISTANT DATA
============================================================
Objectif :
Transformer une question métier en requête SQL Server en
respectant le modèle de données, les définitions métier et
les règles de sécurité."""

    def _build_scope(self) -> str:
        lines = ["PÉRIMÈTRE DES DONNÉES :"]

        if not self.allowed_tables:
            lines.append(
                "- Aucune table autorisée n'a été configurée."
            )
            return "\n".join(lines)

        for table in self.allowed_tables:
            table_name = self._get_value(
                table,
                "Table_Name",
                "table_name",
                "name",
            )

            description = self._get_value(
                table,
                "Usage_Description",
                "usage_description",
                "description",
            )

            lines.append(f"- {table_name}")

            if description:
                lines.append(f"  Usage : {description}")

        return "\n".join(lines)

    def _build_business_dictionary(self) -> str:
        lines = ["DICTIONNAIRE MÉTIER :"]

        if not self.business_dictionary:
            lines.append(
                "- Aucun dictionnaire métier disponible."
            )
            return "\n".join(lines)

        for item in self.business_dictionary:
            table_name = self._get_value(
                item,
                "Table_Name",
                "table_name",
            )

            column_name = self._get_value(
                item,
                "Column_Name",
                "column_name",
            )

            business_name = self._get_value(
                item,
                "Nom_Metier",
                "nom_metier",
                "business_name",
            )

            description = self._get_value(
                item,
                "Description_Metier",
                "description_metier",
                "description",
            )

            business_type = self._get_value(
                item,
                "Type_Metier",
                "type_metier",
            )

            unit = self._get_value(
                item,
                "Unite",
                "unite",
            )

            rule = self._get_value(
                item,
                "Regle_Metier",
                "regle_metier",
            )

            example = self._get_value(
                item,
                "Exemple",
                "exemple",
            )

            if not table_name or not column_name:
                continue

            lines.append(
                f"- {table_name}.{column_name}"
            )

            if business_name:
                lines.append(
                    f"  Nom métier : {business_name}"
                )

            if description:
                lines.append(
                    f"  Définition : {description}"
                )

            if business_type:
                lines.append(
                    f"  Type métier : {business_type}"
                )

            if unit:
                lines.append(f"  Unité : {unit}")

            if rule:
                lines.append(
                    f"  Règle métier : {rule}"
                )

            if example:
                lines.append(f"  Exemple : {example}")

        return "\n".join(lines)

    def _build_kpis(self) -> str:
        lines = ["KPI ET MESURES MÉTIER :"]

        if not self.kpis:
            lines.append("- Aucun KPI disponible.")
            return "\n".join(lines)

        for kpi in self.kpis:
            name = self._get_value(
                kpi,
                "Nom_KPI",
                "nom_kpi",
                "name",
            )

            description = self._get_value(
                kpi,
                "Description",
                "description",
            )

            formula = self._get_value(
                kpi,
                "Formule_SQL",
                "formule_sql",
                "formula",
            )

            unit = self._get_value(
                kpi,
                "Unite",
                "unite",
            )

            frequency = self._get_value(
                kpi,
                "Frequence",
                "frequence",
            )

            if not name:
                continue

            lines.append(f"- {name}")

            if description:
                lines.append(
                    f"  Définition : {description}"
                )

            if formula:
                lines.append(
                    f"  Formule SQL : {formula}"
                )

            if unit:
                lines.append(f"  Unité : {unit}")

            if frequency:
                lines.append(
                    f"  Fréquence : {frequency}"
                )

        return "\n".join(lines)

    def _build_schema(self) -> str:
        lines = ["SCHÉMA MÉTIER ET TECHNIQUE :"]

        allowed_names = {
            self._normalize_table_name(
                self._get_value(
                    item,
                    "Table_Name",
                    "table_name",
                    "name",
                )
            )
            for item in self.allowed_tables
        }

        for table in self.catalog.get("tables", []):
            schema_name = table.get("schema", "dbo")
            table_name = table.get("name", "unknown")
            full_table_name = f"{schema_name}.{table_name}"

            if (
                allowed_names
                and self._normalize_table_name(
                    full_table_name
                ) not in allowed_names
                and self._normalize_table_name(
                    table_name
                ) not in allowed_names
            ):
                continue

            table_description = self._find_table_description(
                full_table_name
            )

            lines.append(f"- Table : {full_table_name}")

            if table_description:
                lines.append(
                    f"  Description : {table_description}"
                )

            for column in table.get("columns", []):
                column_name = column.get(
                    "name",
                    "unknown",
                )

                data_type = column.get(
                    "data_type",
                    "unknown",
                )

                nullable = column.get(
                    "nullable",
                    True,
                )

                is_primary_key = column.get(
                    "is_primary_key",
                    False,
                )

                nullable_label = (
                    "nullable"
                    if nullable
                    else "obligatoire"
                )

                key_label = (
                    ", clé primaire"
                    if is_primary_key
                    else ""
                )

                business_definition = (
                    self._find_column_description(
                        full_table_name,
                        column_name,
                    )
                )

                line = (
                    f"  - {column_name}: "
                    f"{data_type}, "
                    f"{nullable_label}"
                    f"{key_label}"
                )

                if business_definition:
                    line += (
                        f" — {business_definition}"
                    )

                lines.append(line)

        return "\n".join(lines)

    def _build_relationships(self) -> str:
        lines = ["RELATIONS ET JOINTURES AUTORISÉES :"]

        if not self.relationships:
            lines.append(
                "- Aucune relation métier disponible."
            )
            return "\n".join(lines)

        for relationship in self.relationships:
            source_table = self._get_value(
                relationship,
                "Source_Table",
                "source_table",
                "source_schema",
            )

            source_column = self._get_value(
                relationship,
                "Source_Column",
                "source_column",
            )

            target_table = self._get_value(
                relationship,
                "Target_Table",
                "target_table",
                "target_schema",
            )

            target_column = self._get_value(
                relationship,
                "Target_Column",
                "target_column",
            )

            relationship_type = self._get_value(
                relationship,
                "Relationship_Type",
                "relationship_type",
            )

            description = self._get_value(
                relationship,
                "Description",
                "description",
                "meaning",
            )

            if not source_table or not target_table:
                continue

            lines.append(
                f"- {source_table}.{source_column} = "
                f"{target_table}.{target_column}"
            )

            if relationship_type:
                lines.append(
                    f"  Type : {relationship_type}"
                )

            if description:
                lines.append(
                    f"  Signification : {description}"
                )

        return "\n".join(lines)

    def _build_synonyms(self) -> str:
        lines = ["SYNONYMES MÉTIER :"]

        if not self.synonyms:
            lines.append("- Aucun synonyme disponible.")
            return "\n".join(lines)

        for synonym in self.synonyms:
            entity_id = self._get_value(
                synonym,
                "Entity_ID",
                "entity_id",
            )

            synonym_value = self._get_value(
                synonym,
                "Synonym",
                "synonym",
            )

            if synonym_value:
                lines.append(
                    f"- {synonym_value} "
                    f"(entité : {entity_id})"
                )

        return "\n".join(lines)

    def _build_rules(self) -> str:
        lines = ["RÈGLES DE GÉNÉRATION SQL :"]

        if not self.rules:
            lines.append("- Aucune règle SQL disponible.")
            return "\n".join(lines)

        for rule in self.rules:
            rule_name = self._get_value(
                rule,
                "Rule_Name",
                "rule_name",
                "name",
            )

            rule_type = self._get_value(
                rule,
                "Rule_Type",
                "rule_type",
                "type",
            )

            rule_value = self._get_value(
                rule,
                "Rule_Value",
                "rule_value",
                "value",
            )

            description = self._get_value(
                rule,
                "Description",
                "description",
            )

            if rule_name:
                lines.append(f"- {rule_name}")

            if rule_type:
                lines.append(f"  Type : {rule_type}")

            if rule_value:
                lines.append(f"  Valeur : {rule_value}")

            if description:
                lines.append(
                    f"  Description : {description}"
                )

        return "\n".join(lines)

    def _build_examples(self) -> str:
        lines = ["EXEMPLES VALIDÉS :"]

        if not self.examples:
            lines.append("- Aucun exemple disponible.")
            return "\n".join(lines)

        for example in self.examples:
            question = self._get_value(
                example,
                "Question",
                "question",
            )

            intent = self._get_value(
                example,
                "Intent",
                "intent",
            )

            sql_query = self._get_value(
                example,
                "SQL_Query",
                "sql_query",
                "query",
            )

            explanation = self._get_value(
                example,
                "Explanation",
                "explanation",
            )

            if question:
                lines.append(
                    f"- Question : {question}"
                )

            if intent:
                lines.append(
                    f"  Intention : {intent}"
                )

            if sql_query:
                lines.append(
                    f"  SQL validé : {sql_query}"
                )

            if explanation:
                lines.append(
                    f"  Explication : {explanation}"
                )

        return "\n".join(lines)

    def _find_table_description(
        self,
        full_table_name: str,
    ) -> str:
        for item in self.business_dictionary:
            table_name = self._get_value(
                item,
                "Table_Name",
                "table_name",
            )

            if (
                self._normalize_table_name(table_name)
                == self._normalize_table_name(
                    full_table_name
                )
            ):
                description = self._get_value(
                    item,
                    "Description_Table",
                    "description_table",
                )

                if description:
                    return description

        return ""

    def _find_column_description(
        self,
        full_table_name: str,
        column_name: str,
    ) -> str:
        for item in self.business_dictionary:
            table_name = self._get_value(
                item,
                "Table_Name",
                "table_name",
            )

            current_column = self._get_value(
                item,
                "Column_Name",
                "column_name",
            )

            same_table = (
                self._normalize_table_name(table_name)
                == self._normalize_table_name(
                    full_table_name
                )
            )

            same_column = (
                str(current_column).lower()
                == str(column_name).lower()
            )

            if same_table and same_column:
                return self._get_value(
                    item,
                    "Description_Metier",
                    "description_metier",
                    "description",
                )

        return ""

    @staticmethod
    def _normalize_table_name(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _get_value(
        item: dict[str, Any],
        *keys: str,
    ) -> Any:
        for key in keys:
            if key in item and item[key] is not None:
                return item[key]

        return None