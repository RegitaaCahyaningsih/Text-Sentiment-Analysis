from fastapi import FastAPI, Depends, Header
from pydantic import BaseModel
import uvicorn
from custom_logging import CustomizeLogger
import logging
from pathlib import Path
from decouple import config
import security
from NLP_Models import deepSentiment as dst

logger = logging.getLogger(__name__)

config_path=Path(__file__).with_name(config('LOGGING'))

def create_app() -> FastAPI:
    app = FastAPI(title='API SENTIMENT ANALYSIS')
    logger = CustomizeLogger.make_logger(config_path)
    app.logger = logger

    return app

app = create_app()

class Request(BaseModel):
    requestid : str
    text : str
@app.route('/')
def hello_world():
    return ("hello world")

@app.post('/data-mining/text-classification', tags=['SENTIMENT ANALYSIS'])
async def main(data: Request, userapi: str=Header(None),  enduser: str=Header(None), APIKey = Depends(security.get_api_key), client_ip: str = Depends(security.get_client_ip)):
    try:
        app.logger.info({"code": 0, "userapi": userapi, "enduser" : enduser, "client": client_ip, "message" : "success", "requestid" : data.requestid, "content" :data.text, "message": "success"})
        result = dst.predict(data.text)
        return {"code": 0, "message" : "success", "requestid" : data.requestid, "content" : result}
    except Exception as e:
        app.logger.info({"code": -1, "userapi": userapi, "enduser" : enduser, "client": client_ip, "message" : "success", "requestid" : data.requestid, "content" :data.text, "message": str(e)})
        return {"code": -1, "message" : "error", "requestid" : data.requestid, "content" : {}}

# @app.post('/Sentiment Analysis')
#     print("hello word")
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",port=1328)