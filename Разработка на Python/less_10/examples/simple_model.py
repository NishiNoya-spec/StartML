from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:0000@localhost:5432/postgres"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()



class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    surname = Column(String)
    age = Column(Integer)


if __name__ == '__main__':
    Base.metadata.create_all(engine)
    
    