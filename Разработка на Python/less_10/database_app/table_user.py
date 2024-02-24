from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from .database import Base, engine, SessionLocal


class User(Base):
    __tablename__ = "user"
    __table_args__ = {"schema": "public"}

    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    id = Column(Integer, primary_key=True)
    os = Column(Integer)
    source = Column(String)


    