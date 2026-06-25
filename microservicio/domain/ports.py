from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from domain.entities import (
    ConsultaClinica,
    Inferencia,
    ResultadoAnomalia,
    ResultadoExtraccion,
)


class ExtractorPort(ABC):
    @abstractmethod
    def extraer(self, texto: str) -> ResultadoExtraccion:
        ...


class AnomalyDetectorPort(ABC):
    @abstractmethod
    def detectar(self, consulta: ConsultaClinica) -> ResultadoAnomalia:
        ...


class RepositorioInferenciasPort(ABC):
    @abstractmethod
    def guardar(self, inferencia: Inferencia) -> Inferencia:
        ...

    @abstractmethod
    def listar(
        self,
        tipo:   Optional[str],
        desde:  Optional[datetime],
        hasta:  Optional[datetime],
        limit:  int,
        offset: int,
    ) -> tuple:
        """Devuelve (lista[Inferencia], total_count)."""
        ...
