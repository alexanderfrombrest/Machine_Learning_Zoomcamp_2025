import uvicorn
from fastapi import FastAPI

app = FastAPI(title='ping')

@app.get("/ping")
def ping():
    return "PONG"

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9696)