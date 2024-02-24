from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import SessionLocal
from .table_user import User
from .table_post import Post
from .table_feed import Feed
from .schema import UserGet, PostGet, FeedGet


app = FastAPI()


def get_db():
    with SessionLocal() as db:
        try:
            yield db
        finally:
            db.close()


@app.get("/user/{id}", response_model=UserGet)
def get_user_by_id(id: int, db: Session = Depends(get_db)):

    result = db.query(User).filter(User.id == id).one_or_none()

    if not result:
        raise HTTPException(status_code=404, detail="user not found")
    else:
        return result


@app.get("/post/{id}", response_model=PostGet)
def get_post_by_id(id: int, db: Session = Depends(get_db)):

    result = db.query(Post).filter(Post.id == id).one_or_none()

    if not result:
        raise HTTPException(status_code=404, detail="post not found")
    else:
        return result


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    results = (
        db.query(Feed)
        .filter(Feed.user_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )

    return results


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: Optional[int] = 10, db: Session = Depends(get_db)):
    results = (
        db.query(Feed)
        .filter(Feed.post_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )

    return results


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_post_recommendations(
    id: Optional[int] = None, limit: Optional[int] = 10, db: Session = Depends(get_db)
):
    """
    SELECT f.post_id, COUNT(f.post_id)
    FROM feed_action f
    WHERE f.action = 'like'
    GROUP BY f.post_id
    ORDER BY COUNT(f.post_id) DESC
    LIMIT 10;
    """

    # Сначала создадим подзапрос для подсчета лайков
    subquery = (
        db.query(Feed.post_id, func.count(Feed.post_id).label("like_count"))
        .filter(Feed.action == "like")
        .group_by(Feed.post_id)
        .subquery()
    )

    # Затем выполним объединение с таблицей Post и отсортируем результаты
    posts = (
        db.query(Post, subquery.c.like_count)
        .join(subquery, Post.id == subquery.c.post_id)
        .order_by(subquery.c.like_count.desc())
        .limit(limit)
        .all()
    )

    # Преобразуем результаты в список объектов PostGet
    post_results = []
    for post, like_count in posts:
        post_data = {
            "id": post.id,
            "text": post.text,
            "topic": post.topic,
            "like_count": like_count or 0,
        }
        post_results.append(post_data)

    return post_results
