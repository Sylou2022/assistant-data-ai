# code : ml/model_registry.py

"""
app/ml/model_registry.py

Registre centralisé des modèles ML de l'Assistant Data IA.

Responsabilités :
- enregistrer les modèles disponibles
- récupérer un modèle par son nom
- associer un modèle à une tâche ML
- lister les modèles disponibles
- éviter que le DataPipeline connaisse directement
  les différentes implémentations ML

Architecture :

    MLRouter
        ↓
    tâche ML
        ↓
    ModelRegistry
        ↓
    modèle ML
        ↓
    résultat
"""


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .ml_router import MLTask


@dataclass
class ModelEntry:
    """
    Représente un modèle enregistré dans le registry.
    """

    name: str
    task: MLTask
    factory: Callable[..., Any]
    description: str = ""
    version: str = "1.0"
    enabled: bool = True


class ModelRegistry:
    """
    Registre central des modèles ML.

    Exemple :

        registry = ModelRegistry()

        model = registry.get_model(
            task=MLTask.FORECASTING
        )

        result = model.run(df)
    """

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}

        self._register_default_models()

    # ------------------------------------------------------------------
    # Enregistrement
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        task: MLTask,
        factory: Callable[..., Any],
        description: str = "",
        version: str = "1.0",
        enabled: bool = True,
    ) -> None:
        """
        Enregistre un modèle dans le registry.

        Parameters
        ----------
        name:
            Nom unique du modèle.

        task:
            Tâche ML associée.

        factory:
            Classe ou fonction permettant de créer le modèle.

        description:
            Description du modèle.

        version:
            Version du modèle.

        enabled:
            Permet d'activer ou désactiver le modèle.
        """

        if not name:
            raise ValueError(
                "Le nom du modèle ne peut pas être vide."
            )

        if not isinstance(task, MLTask):
            raise TypeError(
                "task doit être une instance de MLTask."
            )

        if not callable(factory):
            raise TypeError(
                "factory doit être une classe ou une fonction callable."
            )

        if name in self._models:
            raise ValueError(
                f"Le modèle '{name}' est déjà enregistré."
            )

        self._models[name] = ModelEntry(
            name=name,
            task=task,
            factory=factory,
            description=description,
            version=version,
            enabled=enabled,
        )

    # ------------------------------------------------------------------
    # Suppression
    # ------------------------------------------------------------------

    def unregister(
        self,
        name: str,
    ) -> bool:
        """
        Supprime un modèle du registry.

        Retourne True si le modèle existait.
        """

        if name not in self._models:
            return False

        del self._models[name]

        return True

    # ------------------------------------------------------------------
    # Récupération par nom
    # ------------------------------------------------------------------

    def get_model(
        self,
        name: str,
        **kwargs,
    ) -> Any:
        """
        Instancie et retourne un modèle enregistré par son nom.
        """

        if name not in self._models:
            raise KeyError(
                f"Modèle inconnu : '{name}'. "
                f"Modèles disponibles : {self.list_models()}"
            )

        entry = self._models[name]

        if not entry.enabled:
            raise RuntimeError(
                f"Le modèle '{name}' est actuellement désactivé."
            )

        return entry.factory(**kwargs)

    # ------------------------------------------------------------------
    # Récupération par tâche
    # ------------------------------------------------------------------

    def get_model_for_task(
        self,
        task: MLTask,
        **kwargs,
    ) -> Any:
        """
        Retourne le premier modèle actif associé à une tâche ML.
        """

        models = self.get_models_for_task(task)

        if not models:
            raise KeyError(
                f"Aucun modèle actif trouvé pour la tâche "
                f"'{task.value}'."
            )

        entry = models[0]

        return entry.factory(**kwargs)

    # ------------------------------------------------------------------
    # Recherche par tâche
    # ------------------------------------------------------------------

    def get_models_for_task(
        self,
        task: MLTask,
    ) -> List[ModelEntry]:
        """
        Retourne tous les modèles actifs correspondant
        à une tâche ML.
        """

        if not isinstance(task, MLTask):
            raise TypeError(
                "task doit être une instance de MLTask."
            )

        return [
            entry
            for entry in self._models.values()
            if entry.task == task
            and entry.enabled
        ]

    # ------------------------------------------------------------------
    # Informations sur un modèle
    # ------------------------------------------------------------------

    def get_model_info(
        self,
        name: str,
    ) -> ModelEntry:
        """
        Retourne les informations d'un modèle.
        """

        if name not in self._models:
            raise KeyError(
                f"Modèle inconnu : '{name}'."
            )

        return self._models[name]

    # ------------------------------------------------------------------
    # Activation / désactivation
    # ------------------------------------------------------------------

    def enable(
        self,
        name: str,
    ) -> None:
        """
        Active un modèle.
        """

        if name not in self._models:
            raise KeyError(
                f"Modèle inconnu : '{name}'."
            )

        self._models[name].enabled = True

    def disable(
        self,
        name: str,
    ) -> None:
        """
        Désactive un modèle.
        """

        if name not in self._models:
            raise KeyError(
                f"Modèle inconnu : '{name}'."
            )

        self._models[name].enabled = False

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_models(
        self,
        enabled_only: bool = True,
    ) -> List[str]:
        """
        Retourne la liste des noms de modèles disponibles.
        """

        models = self._models.values()

        if enabled_only:
            models = [
                model
                for model in models
                if model.enabled
            ]

        return [
            model.name
            for model in models
        ]

    def list_model_entries(
        self,
        enabled_only: bool = True,
    ) -> List[ModelEntry]:
        """
        Retourne les entrées complètes du registry.
        """

        models = list(self._models.values())

        if enabled_only:
            models = [
                model
                for model in models
                if model.enabled
            ]

        return models

    # ------------------------------------------------------------------
    # Modèles par défaut
    # ------------------------------------------------------------------

    def _register_default_models(self) -> None:
        """
        Enregistre les modèles ML natifs du projet.

        Les imports sont effectués ici afin d'éviter
        les imports circulaires au chargement du module.
        """

        try:
            from .forecasting import ForecastingModel

            self.register(
                name="forecasting_holt_winters",
                task=MLTask.FORECASTING,
                factory=ForecastingModel,
                description=(
                    "Prévision temporelle du chiffre d'affaires "
                    "avec lissage exponentiel (Holt-Winters)."
                ),
                version="1.0",
            )

        except ImportError:
            # Le registry reste utilisable même si un modèle
            # optionnel n'est pas encore installé.
            pass

        # --------------------------------------------------------------
        # Anomaly detection
        # --------------------------------------------------------------
        #
        # Le modèle sera ajouté lorsque anomaly_detection.py
        # sera implémenté.
        #
        # Exemple futur :
        #
        # from .anomaly_detection import AnomalyDetectionModel
        #
        # self.register(
        #     name="anomaly_detection",
        #     task=MLTask.ANOMALY_DETECTION,
        #     factory=AnomalyDetectionModel,
        #     description="Détection des valeurs inhabituelles.",
        # )
        #

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def has_model(
        self,
        name: str,
    ) -> bool:
        """
        Vérifie si un modèle existe dans le registry.
        """

        return name in self._models

    def has_model_for_task(
        self,
        task: MLTask,
    ) -> bool:
        """
        Vérifie si un modèle actif existe pour une tâche.
        """

        return len(
            self.get_models_for_task(task)
        ) > 0

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Nombre total de modèles enregistrés.
        """

        return len(self._models)

    def __repr__(self) -> str:
        """
        Représentation lisible du registry.
        """

        return (
            f"ModelRegistry("
            f"models={self.list_models(enabled_only=False)}"
            f")"
        )


# ----------------------------------------------------------------------
# Registry global
# ----------------------------------------------------------------------

_default_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Retourne l'instance globale du ModelRegistry.

    Cela permet d'utiliser le même registry dans toute
    l'application sans recréer les modèles à chaque appel.
    """

    global _default_registry

    if _default_registry is None:
        _default_registry = ModelRegistry()

    return _default_registry


# ----------------------------------------------------------------------
# Fonction pratique
# ----------------------------------------------------------------------

def get_model_for_task(
    task: MLTask,
    **kwargs,
) -> Any:
    """
    Raccourci pour récupérer un modèle associé à une tâche ML.

    Exemple :

        model = get_model_for_task(
            MLTask.FORECASTING
        )
    """

    registry = get_model_registry()

    return registry.get_model_for_task(
        task,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Test local
# ----------------------------------------------------------------------

if __name__ == "__main__":

    print("=== MODEL REGISTRY ===")

    registry = ModelRegistry()

    print("\nModèles disponibles :")
    for model_name in registry.list_models():
        print(f"  - {model_name}")

    print(
        f"\nNombre de modèles : {len(registry)}"
    )

    # --------------------------------------------------------------
    # Vérification du forecasting
    # --------------------------------------------------------------

    if registry.has_model_for_task(
        MLTask.FORECASTING
    ):
        print(
            "\nModèle de forecasting trouvé."
        )

        model = registry.get_model_for_task(
            MLTask.FORECASTING
        )

        print(
            f"Classe du modèle : "
            f"{model.__class__.__name__}"
        )

    # --------------------------------------------------------------
    # Informations détaillées
    # --------------------------------------------------------------

    print("\n=== INFORMATIONS ===")

    for entry in registry.list_model_entries():

        print(f"\nNom        : {entry.name}")
        print(f"Tâche      : {entry.task.value}")
        print(f"Version    : {entry.version}")
        print(f"Activé     : {entry.enabled}")
        print(f"Description: {entry.description}")

