from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:0000@localhost:5432/postgres"
SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
