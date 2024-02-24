from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base, engine, SessionLocal


class Post(Base):
    __tablename__ = "post"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)


