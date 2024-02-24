from fastapi import FastAPI
from fastapi import HTTPException, Depends
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import datetime

app = FastAPI()


#-----------------------

""" 4
> Простой endpoint
Во всех заданиях на написание кода для успешной сдачи необходимо писать код в файле
с названием app.py и создавать экземпляр приложения через app = FastAPI() (т.е. с именем app).

Напишите endpoint GET /, т.е. который будет слушать корень "/" сайта и метод GET.
Endpoint должен возвращать строку "hello, world".

Используйте для этого FastAPI. 
"""

#@app.get("/")
#def hello_world():
#    return "hello, world"

#-----------------------

""" 5
> GET: принимаем параметры
Напишите endpoint GET /, который будет принимать два числа на вход и возвращать их сумму.

Пример использования: GET /?a=5&b=3 вернет строку 8 и status_code=200

Чтобы проверить у себя работоспособность, запустите сервис через uvicorn app:app --reload
--port 8899, затем в Postman зайдите на http://localhost:8899/?a=5&b=3, должно вернуться 8.
Вы можете также зайти по этой же ссылке в браузере и увидеть одну надпись 8 на белом фоне.

Для сдачи задания отправьте файл app.py
"""

#@app.get("/")
#def sum(a: int, b: int) -> int:
#    return a + b

#-----------------------

""" 6
> GET: принимаем сложные типы данных
Задание 5
Потренируемся в валидации данных через указание типа переменной.

Напишите endpoint GET /sum_date, который будет принимать два параметра: current_date и offset.
current_date будет иметь вид YYYY-MM-DD - это дата в формате год-месяц-день (например, 2022-01-01).
offset будет целым числом (может быть отрицательным).

Ваш endpoint должен провалидировать, что current_date имеет тип datetime.date
(используйте подсказку типов, когда будете указывать список аргументов функции!) и валидировать,
что offset имеет тип int. Затем endpoint должен прибавить к дате current_date дни в количестве offset
и вернуть ответом строку в формате год-месяц-день.

Пример использования:

GET /sum_date?current_date=2008-09-15&offset=2 вернет "2008-09-17"

Как обычно, вы можете использовать Postman для проверки кода на этапе разработки. Пример GET запроса:
localhost:8899/sum_date?current_date=2008-01-15&offset=2, должно вернуть "2008-01-17" и status code 200.

Отправьте файл app.py с реализацией этого endpoint.
"""

#@app.get("/sum_date")
#def sum_date(current_date: datetime.date, offset: int):
#    return current_date + datetime.timedelta(days=offset)

#-----------------------

""" 7
> POST: эмулируем регистрацию (1/2)
В этом задании потренируемся в валидации данных через модели (BaseModel из pydantic).

В следующем задании нужно будет написать endpoint POST /user/validate.
Мы обязательно будем валидировать входные данные: ни один из ключей в JSON не должен 
быть пропущен и все они должны иметь тип, как указано выше. Для валидации воспользуемся 
моделями pydantic.

В этом задании нужно написать класс User, который будет наследоваться от BaseModel из pydantic. 
Опишите в нем поля name, surname, age, registration_date, укажите их типы (через : - как вот name: str).

Как обычно, пишите код в файле app.py. Для сдачи задания отправьте этот файл с описанием класса User.
"""

""" 8
> POST: эмулируем регистрацию (2/2)
Теперь, когда у нас есть описание модели, можно приступить к написанию endpoint.

Напишите endpoint POST /user/validate, который будет принимать JSON в формате из 
прошлого задания и валидировать его. Для валидации укажите в функции, что она принимает 
аргумент типа User (возьмите этот класс из прошлого пункта). Наконец, верните в endpoint 
строку "Will add user: <имя> <фамилия> with age <возраст>"
"""

#class User(BaseModel):
#        name: str 
#        surname: str
#        age: int
#        registration_date: datetime.date

#@app.post("/user/validate")
#def user(user: User):
#      return f"Will add user: {user.name} {user.surname} with age {user.age}"


#-----------------------

""" 9
> GET: подключаем БД
Потренируемся в подключении PostgreSQL СУБД к нашим приложениям. 
ля этого будем использовать таблицу user из базы данных startml (6 урок).

Напишите endpoint GET /user/<id>, который будет принимать ID пользователя, 
искать его в БД и возвращать gender, age и city в формате JSON. Обратите 
внимание на то, как передается ID в запросе - прямо в самой строке.
"""


"""
@app.get("/user/{user_id}")
def find_user(user_id):
    connection = psycopg2.connect(
            database="startml",
            user="robot-startml-ro",
            password="pheiph0hahj1Vaif",
            host="postgres.lab.karpov.courses",
            port=6432, 
            cursor_factory=RealDictCursor
    )
    cursor = connection.cursor()

    cursor.execute(
        f"
        SELECT gender, age, city
        FROM "user"
        WHERE id = {user_id}
        "
    )

    result = cursor.fetchone()

    cursor.close()

    connection.close()

    return result
"""


#-----------------------

""" 10
> Обрабатываем ошибки
В прошлом задании можно заметить, что GET /user/1 даст нам null со status code 200.
Кажется, что это не очень информативно: если пользователя не нашлось, стоит вернуть 404 и отдать сообщение.

Перепишите endpoint из прошлого задания так, чтобы он возвращал status code 404 и JSON

```python
{
  "detail": "user not found"
}
```

"""
"""

@app.get("/user/{user_id}")
def find_user(user_id):
    connection = psycopg2.connect(
            database="startml",
            user="robot-startml-ro",
            password="pheiph0hahj1Vaif",
            host="postgres.lab.karpov.courses",
            port=6432, 
            cursor_factory=RealDictCursor
    )
    cursor = connection.cursor()

    cursor.execute(
        f"
        SELECT gender, age, city
        FROM "user"
        WHERE id = {user_id}
        "
    )

    result = cursor.fetchone()

    cursor.close()

    connection.close()

    if not result:
        raise HTTPException(status_code=404, detail="user not found")
    else:
        return result

"""

#-----------------------

""" 11
> Dependency injection
Для работы с базой данных мы создавали подключение явно: либо на каждый запрос, либо
один раз при старте приложения. В FastAPI есть более гибкий механизм предоставления
различных сервисов - dependency injection.

Концептуально это выглядит так: вы пишете функцию, которая возвращает объект, нужный
для обработки endpoint, а endpoint заявляет, что ему нужен объект из некой функции.
FastAPI изнутри себя производит состыковку, вызывает нужную функцию и передает ее
результат в обработчик endpoint-а.

Подробнее про это можно почитать в документации.

Давайте попробуем переписать наше взаимодействие с базой данных на использование механизма
dependency injection. Напишите функцию get_db(), которая будет возвращать объект
psycopg2.connection. Затем перепишите свой endpoint GET /user/{id} на использование результата
get_db() как Dependency. Это будет выглядеть в духе:

@app.get(...)
def my_func(db = Depends(get_db)):
  ...


Затем вы можете создать курсор через with db.cursor() as cursor и работать аналогичным образом.

Отправьте файл app.py с реализацией функции get_db() и endpoint-а, использующего get_db как зависимость.
```

"""

def get_db():
    connection = psycopg2.connect(
            database="startml",
            user="robot-startml-ro",
            password="pheiph0hahj1Vaif",
            host="postgres.lab.karpov.courses",
            port=6432, 
            cursor_factory=RealDictCursor
    )

    return connection



@app.get("/user/{user_id}")
def find_user(user_id: int, db = Depends(get_db)):

    with db.cursor() as cursor:

        cursor.execute(
            f"""
            SELECT gender, age, city
            FROM "user"
            WHERE id = {user_id}
            """
        )
        result = cursor.fetchone()
        cursor.close()

    if not result:
        raise HTTPException(status_code=404, detail="user not found")
    else:
        return result


#-----------------------

""" 11
> Валидация ответа
Задание 11
В прошлых заданиях для того, чтобы возвращать данные в нужном формате, мы использовали
RealDictCursor и ограничивали список колонок в SELECT-запросе. Но не всегда такой трюк
получается и рано или поздно встанет вопрос - как проверять, что мы возвращаем значения
в правильном формате.

Для валидации ответа можно использовать те же pydantic модели.

Напишите эндпойнт GET /post/{id}, который будет возвращать информацию о постах по их id и
валидировать ответ. Будем обращаться к таблице post из базы данных startml, 6 урок. Для
этого напишите класс PostResponse, наследующийся от pydantic.BaseModel, и в классе объявите
поля id, text и topic с нужными типами (по аналогии с заданием на класс User).
Добавьте orm_mode = True таким же способом, как делали на лекции (или в документации).

Затем укажите, что функция-обработчик для endpoint (это функция, которую вы декорировали)
возвращает тип PostResponse. Для указания типа возвращаемого значения используется синтаксис ->

def my_func_returning_bool() -> bool:  # после -> может быть встроенный тип, а может быть класс
  return False


Наконец, скажите своему декоратору, чтобы валидировал возвращаемое значение против модели PostResponse:

# было: @app.get(...)
# станет:
@app.get(..., response_model=PostResponse)
def my_func(db):
  ...
  
  result = cursor.fetchone()
  # было, не контролируем формат
  # return result
  # станет
  return PostResponse(**result)


Не забудьте, что обработку 404 надо делать заранее.

По итогу у вас получится следующая схема

cursor.fetchone() возвращает dict-объект, который может иметь любые ключи.
PostResponse попытается из этого dict-объект создать объект класса PostResponse,
валидируя при этом входные данные (т.е. данные из словаря).
Если ошибок валидации нет, то объект класса создается, затем тут же передается в
return у функции-обработчика endpoint.
FastAPI видит, что в return ушел объект из модели pydantic и понимает, что его
надо сериализировать в JSON (превратить в JSON).
FastAPI делает сериализацию, работая с провалидированным объектом.
FastAPI возвращает ответ на запрос в формате JSON.
Пользователь получает данные строго в том формате, в каком они описаны в классе PostReponse.
Реализуйте валидацию описанным способом и отправьте файл app.py.
```

"""

class PostResponse(BaseModel):
    id: int
    text: str
    topic: str

    class Config():
        orm_mode = True


@app.get("/post/{id}", response_model = PostResponse)
def get_post(id: int, db = Depends(get_db)) -> PostResponse:

    with db.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT id, text, topic
            FROM post
            WHERE id = {id}
            """
        )
        result = cursor.fetchone()
        cursor.close()

    if not result:
        raise HTTPException(status_code=404, detail="post not found")
    else:
        return PostResponse(**result)
