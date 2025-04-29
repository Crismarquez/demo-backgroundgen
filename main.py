import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from routers.photosession import router as marketing_router

app = FastAPI()
app.title = "Background Generation API"
app.version = "0.0.1"

app.include_router(marketing_router)

#app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/', tags=['home'])
def message():
    return HTMLResponse('<h1>Background Generation API</h1>')

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=5000)
