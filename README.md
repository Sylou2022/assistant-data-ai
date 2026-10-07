---
title: Assistant Data
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---



# Assistant Data AI

Assistant conversationnel permettant d'interroger une base SQL Server
avec des questions en langage naturel.

## Fonctionnalités

- Connexion SQL Server avec SQLAlchemy et pyodbc.
- Chargement automatique du schéma de la base.
- Construction d'une Semantic Layer.
- Dictionnaire métier et synonymes.
- Gestion des KPI.
- Génération de requêtes T-SQL.
- Validation des tables et colonnes autorisées.
- Blocage des instructions d'écriture.
- Exécution en lecture seule.
- Résultats sous forme de DataFrame.
- Export CSV.
- Interface Gradio.
- Compatible avec un fournisseur LLM et un modèle local.

## Architecture

Question utilisateur
→ Semantic Layer
→ LLM
→ Générateur SQL
→ Validateur
→ Exécuteur
→ Résultats

## Technologies

- Python
- SQL Server
- SQLAlchemy
- pyodbc
- pandas
- Gradio
- OpenAI API
- SQLGlot
>>>>>>> 4baafb6db95ef4b99a5e3e027a1913027a293ef9
