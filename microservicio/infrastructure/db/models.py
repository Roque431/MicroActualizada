import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class InferenciaORM(Base):
    __tablename__ = "inferencias"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tipo        = Column(String(20), nullable=False)
    input_json  = Column(JSONB, nullable=False)
    output_json = Column(JSONB, nullable=False)
    score       = Column(Float, nullable=True)
    es_anomalia = Column(Boolean, nullable=True)
    created_at  = Column(DateTime, nullable=False, default=datetime.utcnow)
