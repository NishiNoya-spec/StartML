from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from .database import Base, engine, SessionLocal


class Post(Base):
    __tablename__ = "post"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)


class User(Base):
    __tablename__ = "user"
    __table_args__ = {"schema": "public"}

    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    id = Column(Integer, primary_key=True)
    os = Column(String)
    source = Column(String)


class Feed(Base):
    __tablename__ = "feed_action"
    __table_args__ = {"schema": "public"}

    action = Column(String)
    post_id = Column(Integer, ForeignKey("public.post.id"), primary_key=True)
    post = relationship(Post, foreign_keys=[post_id])
    time = Column(TIMESTAMP)
    user_id = Column(Integer, ForeignKey("public.user.id"), primary_key=True)
    user = relationship(User, foreign_keys=[user_id])



if __name__ == "__main__":
    session = SessionLocal()
    results = (
        session.query(Feed).limit(10).all()
    )
    for feed in results:
        print(feed.post.text)



