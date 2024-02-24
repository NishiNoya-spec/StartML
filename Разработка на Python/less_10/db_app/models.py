from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .database import Base, engine, SessionLocal


class Member(Base):
    __tablename__ = "members"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True, name="memid")
    surname = Column(String)
    first_name = Column(String, name="firstname")
    address = Column(String)
    zipcode = Column(String)
    telephone = Column(String)
    recommended_by_id = Column(
        Integer, ForeignKey("public.members.memid"), name="recommendedby"
    )
    recommended_by = relationship("Member", remote_side=[id])
    join_date = Column(TIMESTAMP, name="joindate")


class Facility(Base):
    __tablename__ = "facilities"
    __table_args__ = {"schema": "public"}
    id = Column(Integer, primary_key=True, name="facid")
    name = Column(String)
    member_cost = Column(Float, name="membercost")
    guest_cost = Column(Float, name="guestcost")
    initial_outlay = Column(Float, name="initialoutlay")
    monthly_maintenance = Column(Float, name="monthlymaintenance")


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = {"schema": "public"}
    id = Column(Integer, primary_key=True, name="bookid")
    facility_id = Column(
        Integer, ForeignKey("public.facilities.facid"), primary_key=True, name="facid"
    )
    facility = relationship("Facility", foreign_keys=[facility_id])
    member_id = Column(
        Integer, ForeignKey("public.members.memid"), primary_key=True, name="memid"
    )
    member = relationship("Member", foreign_keys=[member_id])
    start_time = Column(TIMESTAMP, name="starttime")
    slots = Column(Integer)


if __name__ == "__main__":
    session = SessionLocal()
    results = (
        session.query(Booking)
        .join(Member)
        .filter(Member.first_name == "Tim")
        .limit(5)
        .all()
    )
    for x in results:
        print(f"name = {x.member.zipcode}, strat_time = {x.start_time}")
