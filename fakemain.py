from fastapi import FastAPI,Request, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json

app = FastAPI()
# Allow frontend from localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "http://localhost:3000","http://localhost:4200","http://127.0.0.1:4200",
                    "http://localhost:5000","http://localhost:8000","http://127.0.0.1:5000","http://127.0.0.1:8000"],  # <-- adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fake api1
# return type: str,int,str,int
class TextPayload(BaseModel):
    text: str
    account: str
    user: str
@app.post("/api/submit")
async def submit_text(payload: TextPayload):
    return {
        "message": "Text received from fake API!",
        "length": 32,
        "id": "ae32d016-a21a-4b76-b1ee-677bfc78f354",
        "query":0
    }

#fake api2
#return type:str,str,str,str,str,str
class UUIDRequest(BaseModel):
    uuid: str

@app.post("/api/api2")
async def diag_prc(request: UUIDRequest):
    return {
                "note": "fake medical notes",
                "diag_list": '["fake diag1","fake diag2"]',
                "prc_list": '["fake prc1"]',
                "summary":"fake summary",
                "diag_hover":'["fake diag1 hover","fake diag2 hover"]',
                "prc_hover": '["fake prc1 hover"]',
            }


#fake api3
#return type:str
class TextPayload(BaseModel):
    text: str

@app.post("/api/api3")
async def submit_text(payload: TextPayload):
    return {
        "label":"generated fake hovering label by using the input text."
            }


class TextPayload(BaseModel):
    diag_list: str
    prc_list: str
    diag_hover: str
    prc_hover: str
    uuid: str
@app.post("/api/api4")
async def generate_ICD10(payload: TextPayload):
    return {
        #ICD_CM
        "ICD10_diag":'["fake ICD diag1x","fake diag2","fake ICD diag3","fake ICD diag4","fake ICD diag5","fake ICD diag6","fake ICD diag7,"fake ICD diag8"]',
        #ICD_PCS
        "ICD10_prc":'["fake ICD prc1x","fake prc2","fake ICD prc3","fake ICD prc4","fake ICD prc5","fake ICD prc6","fake ICD prc7","fake ICD prc8"]',
                "ICD10_diag_hovering":'["fake ICD diag1 hovering","fake diag2 hovering","fake ICD diag3 hovering","fake ICD diag hovering4","fake ICD diag5 hovering","fake ICD diag6 hovering","fake ICD diag7 hovering","fake ICD diag8  hovering"]',
                "ICD10_prc_hovering":'["fake ICD prc1  hovering","fake prc2  hovering","fake ICD prc3  hovering","fake ICD prc4  hovering","fake ICD prc5  hovering","fake ICD prc6  hovering","fake ICD prc7 hovering","fake ICD prc8 hovering"]',
            }



#for generate_DRG button
class TextPayload(BaseModel):
    uuid: str
@app.post("/api/api5")
async def generate_DRG(payload: TextPayload):
    return {"DRG":"980",
            "PRC_related":'["fake relationship1","fake relationship2","fake relationship3","fake relationship4","fake relationship5","fake relationship6","fake relationship7","fake relationship8"]',
            "DRG_hovering":"fake DRG rationality hovering"
            }


#for the small green edit DRG button
class TextPayload(BaseModel):
    uuid:str
    DRG: str
@app.post("/api/api6")
async def generate_DRG(payload: TextPayload):
    return {"DRG_hovering":"fake DRG hovering"}


#for Generate_result button
class TextPayload(BaseModel):
    uuid:str
    DRG: str
    DRG_hovering: str
@app.post("/api/api7")
async def generate_DRG(payload: TextPayload):
    return {"DRG_money":"11,000"}


class TextPayload(BaseModel):
    uuid:str
@app.post("/api/apiback1")
async def generate_DRG(payload: TextPayload):
    return 1





@app.get("/v1/aistudio/data/getFiles")
async def getFiles():
    return {
        "validation-files": [
            "0e488/1212/files/Normal_Routine/rawdata/Normal_Routine.csv",
            "0e488/1212/files/Diagonal_Routine/rawdata/Diagonal_Routine.csv",
            "0e488/1212/files/Speed_Routine/rawdata/Speed_Routine.csv"
        ],
        "train-test-file": [
            "0e488/1212/files/Half_Cycle_Routine/rawdata/Half_Cycle_Routine.csv"
        ],
        "metadata": {
        "files": [
            {
                "file_id": "Half_Cycle_Routine",
                "original_name": "Half_Cycle_Routine.csv",
                "server_path": "0e488/1212/files/Half_Cycle_Routine/rawdata/Half_Cycle_Routine.csv",
                "role": "train-test"
            },
            {
                "file_id": "Normal_Routine",
                "original_name": "Normal_Routine.csv",
                "server_path": "0e488/1212/files/Normal_Routine/rawdata/Normal_Routine.csv",
                "role": "validation"
            },
            {
                "file_id": "Diagonal_Routine",
                "original_name": "Diagonal_Routine.csv",
                "server_path": "0e488/1212/files/Diagonal_Routine/rawdata/Diagonal_Routine.csv",
                "role": "validation"
            },
            {
                "file_id": "Speed_Routine",
                "original_name": "Speed_Routine.csv",
                "server_path": "0e488/1212/files/Speed_Routine/rawdata/Speed_Routine.csv",
                "role": "validation"
            }
        ]
    }
    }

class TextPayload(BaseModel):
    file_id: str
    apikey: str

@app.post("/v1/aistudio/changeDatasetRole")
async def changeDatasetRole(
    format: str = Query(None),
    payload: TextPayload = Body(...)
):
    return {"details": "success"}
