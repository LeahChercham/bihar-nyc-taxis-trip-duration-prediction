import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message" : "hello"}


if(__name__) == '__main__':
    uvicorn.run("main:app", host = "O.O.O.O", port=5000, reload=True)

