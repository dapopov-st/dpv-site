from fastapi import FastAPI
from enum import Enum
import typing as t
from pydantic import BaseModel

app = FastAPI()

@app.get("/") #route/endpoint
def home_page():
    return {"message":"Hello World!"}

@app.get("/items/{item_id}") #item_id is the path parameter
async def read_item(item_id: int):
    return {"item_id":item_id}

@app.get("/users/me") # will not work if placed after, must be before to be valid
async def read_user_current():
    return {"user_id":"Current user"}

@app.get("/users/{user_id}") 
async def read_user(user_id: int):
    return {"user_id":user_id}

class ModelName(str,Enum):
    ALEXNET = 'ALEXNET'
    RESNET = 'RESNET'
    LENET = 'LENET'

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name == ModelName.ALEXNET:
        return {'model_name':model_name}
    elif model_name.value == "LENET":
        return {'model_name': model_name}
    else:
        return {'model_name':f"You have selected {model_name.value}"}
    
@app.get("files/{file_path:path}")
async def read_file(file_path:str):
    return {"file_path":file_path}

animal_db = [{"animal_name":'cat'},{"animal_name":'llama'},{"animal_name":'alpaca'}]

@app.get("/animals/")
async def read_animal(skip: int=0, limit: int=10, optional_param: t.Optional[int]=None):
    return {"animals": animal_db[skip:skip+limit], "optional_parameter":optional_param}

@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: int, q: t.Optional[str]=None, short:bool=False
):
    item = {"item_id":item_id, "owner_id":user_id}
    if q:
        item.update({"q":q})
    if not short:
        item.update({'description':'great item with long description'})
    return item

books_db = []
class Book(BaseModel):
    name:str
    author:str
    description:t.Optional[str]
    price:float

@app.post("/books/")
async def create_item(book:Book):
    books_db.append(book)
    return book 
@app.get("/books/")
async def get_books():
    return books_db