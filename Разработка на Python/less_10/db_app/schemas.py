import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    name: str
    surname: str



class UserGet(BaseModel):
    first_name: str = ""
    surname: str = ""
    address: str = ""
    recommended_by: Optional["UserGet"] = None

    class Config:
        orm_mode = True



class BookingGet(BaseModel):
    momber_id: int
    member: UserGet
    facility_id: int
    start_time: datetime.datetime
    slots: int

    class Config:
        orm_mode = True
