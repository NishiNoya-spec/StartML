import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserGet(BaseModel):
    age: int
    city: str = ""
    country: str = ""
    exp_group: int
    gender: int
    id: int
    os: str = ""
    source: str = ""

    class Config:
        orm_mode = True


class PostGet(BaseModel):
    id: int
    text: str = ""
    topic: str = ""
    like_count: int = 0

    class Config:
        orm_mode = True


class FeedGet(BaseModel):
    action: str = ""
    post_id: int
    post: PostGet
    time: datetime.datetime
    user_id: int
    user: UserGet

    class Config:
        orm_mode = True
