from .simple_model import User, SessionLocal

if __name__ == "__main__":
    user = User(name="Danil", surname="random", age=23)
    session = SessionLocal()
    for user in (
        session.query(User)
        .filter(User.name == "Danil")
        .filter(User.age == 23)
        .limit(2)
        .all()
    ):
        print(user.id)
