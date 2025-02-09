import uvicorn
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
from io import BytesIO
from fastapi import Request


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend")

model = tf.keras.models.load_model('models/yoga_pose/yoga-modelv2.h5')

class_name = ['Bridge Pose','Child-Pose','CobraPose','Downward Dog pose','Pigeon pose','Standing Mountain Pose','Tree Pose','Triangle Pose','Warrior Pose']



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("yoga_pose.html", {"request": request})

@app.get('/ping')
async def check():
    return "Hello world"


def read_file_as_image(data):
    img = Image.open(BytesIO(data)).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array

@app.post('/predict')
async def prediction(file: UploadFile = File(...)):

     print("Prediction endpoint called")
     file.file.seek(0)
     img = read_file_as_image(await file.read())
     img = np.expand_dims(img, axis=0)

     predicted =  model.predict(img)
     result = class_name[np.argmax(predicted[0])]
     confidence = np.max(predicted[0])

     return{
         'class': result,
         'confidence':round(confidence * 100, 1)
     }


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8501)