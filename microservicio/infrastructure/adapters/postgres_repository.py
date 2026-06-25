"""
Adapter que implementa RepositorioInferenciasPort usando SQLAlchemy + PostgreSQL.
"""
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from domain.entities import Inferencia
from domain.ports import RepositorioInferenciasPort
from infrastructure.db.models import InferenciaORM


class PostgresRepository(RepositorioInferenciasPort):

    def __init__(self, db: Session):
        self._db = db

    def guardar(self, inferencia: Inferencia) -> Inferencia:
        orm = InferenciaORM(
            tipo=inferencia.tipo,
            input_json=inferencia.input_json,
            output_json=inferencia.output_json,
            score=inferencia.score,
            es_anomalia=inferencia.es_anomalia,
        )
        self._db.add(orm)
        self._db.commit()
        self._db.refresh(orm)

        inferencia.id         = str(orm.id)
        inferencia.created_at = orm.created_at
        return inferencia

    def listar(
        self,
        tipo:   Optional[str],
        desde:  Optional[datetime],
        hasta:  Optional[datetime],
        limit:  int,
        offset: int,
    ) -> tuple:
        q = self._db.query(InferenciaORM)

        if tipo:
            q = q.filter(InferenciaORM.tipo == tipo)
        if desde:
            q = q.filter(InferenciaORM.created_at >= desde)
        if hasta:
            q = q.filter(InferenciaORM.created_at <= hasta)

        total = q.count()
        rows  = q.order_by(InferenciaORM.created_at.desc()).offset(offset).limit(limit).all()

        inferencias = [
            Inferencia(
                id=str(r.id),
                tipo=r.tipo,
                input_json=r.input_json,
                output_json=r.output_json,
                score=r.score,
                es_anomalia=r.es_anomalia,
                created_at=r.created_at,
            )
            for r in rows
        ]
        return inferencias, total
