import requests
from loguru import logger

r = requests.get("http://localhost:8000/sub?a=10&b=3")

logger.info(r.status_code)


r = requests.post(
    "http://localhost:8000/user/validate", 
    json={
        "name": "Danil", 
        "surname": "Temnkhudov"
    }
)

logger.info(
    f"main info: {r.status_code}"
)

logger.info(
    f"text: {r.text}"
)

logger.info(
    f"text: {r.json()}"
)

