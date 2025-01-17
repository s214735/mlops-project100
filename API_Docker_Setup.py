from __future__ import annotations

import re
from enum import Enum
from http import HTTPStatus
import anyio
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


app = FastAPI()

from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Hello")
    yield
    print("Goodbye")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}