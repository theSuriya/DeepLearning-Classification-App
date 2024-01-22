# server.py
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
# from path.to.your.module import CustomLayer

#Initializing The App
app = FastAPI()

#Secure Our APP Server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Mounting
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/models/yoga_pose/static", StaticFiles(directory='models/yoga_pose/static'), name='static_yoga_pose')
app.mount("/models/weather/static", StaticFiles(directory='models/weather/static'), name='static_weather')
app.mount("/models/sports_ball/static", StaticFiles(directory='models/sports_ball/static'), name='static_sports_ball')
app.mount("/models/mammals/static", StaticFiles(directory='models/mammals/static'), name='static_mammals')
app.mount("/models/flower/static", StaticFiles(directory='models/flower/static'), name='static_flower')
app.mount("/models/dog_breed/static", StaticFiles(directory='models/dog_breed/static'), name='static_dog')
#templates
templates = Jinja2Templates(directory="frontend")
templates1 = Jinja2Templates(directory="models")

#DL or Ml Models (Loading)..
sports_ball_model = tf.keras.models.load_model('models/sports_ball/Sports_ball_prediction_v2.h5')
# weather_model = tf.keras.models.load_model('models/weather/cards_model_v2.h5')
# flower_model = tf.keras.models.load_model('models/flower/flower_prediction.h5')
# yoga_pose_model = tf.keras.models.load_model('models/yoga_pose/yoga-modelv2.h5')
# mammals_model = tf.keras.models.load_model('models/mammals/Mammals_predictionv1.h5')
# dog_model = tf.keras.models.load_model("models/dog_breed/dog_breedv3.h5")

#classes For All Models
# yoga_class = ['Bridge Pose','Child-Pose','CobraPose',
#                    'Downward Dog pose','Pigeon pose','Standing Mountain Pose',
#                    'Tree Pose','Triangle Pose','Warrior Pose']

sports_ball_class = ['american_football', 'baseball', 'basketball', 'billiard_ball', 'bowling_ball', 'cricket_ball',
               'football', 'golf_ball', 'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock', 'table_tennis_ball',
               'tennis_ball','volleyball']

flower_class = ['astilbe','bellflower','black_eyed_susan', 'calendula','california_poppy',
                'carnation', 'common_daisy', 'coreopsis',
                'dandelion','iris','rose','sunflower','tulip','water_lily']

mammals_class = ['african_elephant', 'alpaca', 'american_bison', 'anteater', 'arctic_fox', 'armadillo', 'baboon',
              'badger', 'blue_whale', 'brown_bear', 'camel', 'dolphin', 'giraffe', 'groundhog', 'highland_cattle',
              'horse', 'jackal', 'kangaroo', 'koala', 'manatee', 'mongoose', 'mountain_goat', 'opossum', 'orangutan',
              'otter', 'polar_bear', 'porcupine', 'red_panda', 'rhinoceros', 'sea_lion', 'seal', 'snow_leopard',
              'squirrel', 'sugar_glider', 'tapir', 'vampire_bat', 'vicuna', 'walrus', 'warthog', 'water_buffalo',
              'weasel', 'wildebeest', 'wombat', 'yak', 'zebra']

dog_class = ['Afghan','African Wild Dog', 'Airedale', 'American Hairless',
             'American Spaniel', 'Basenji', 'Basset', 'Beagle', 'Bearded Collie',
             'Bermaise', 'Bichon Frise', 'Blenheim', 'Bloodhound', 'Bluetick', 'Border Collie','Borzoi',
              'Boston Terrier', 'Boxer', 'Bull Mastiff', 'Bull Terrier', 'Bulldog', 'Cairn', 'Chihuahua', 'Chinese Crested',
             'Chow', 'Clumber','Cockapoo', 'Cocker', 'Collie', 'Corgi', 'Coyote', 'Dalmation', 'Dhole', 'Dingo', 'Doberman', 'Elk Hound', 'French Bulldog',
 'German Sheperd',
 'Golden Retriever',
 'Great Dane',
 'Great Perenees',
 'Greyhound',
 'Groenendael',
 'Irish Spaniel',
 'Irish Wolfhound',
 'Japanese Spaniel',
 'Komondor',
 'Labradoodle',
 'Labrador',
 'Lhasa',
 'Malinois',
 'Maltese',
 'Mex Hairless',
 'Newfoundland',
 'Pekinese',
 'Pit Bull',
 'Pomeranian',
 'Poodle',
 'Pug',
 'Rhodesian',
 'Rottweiler',
 'Saint Bernard',
 'Schnauzer',
 'Scotch Terrier',
 'Shar_Pei',
 'Shiba Inu',
 'Shih-Tzu',
 'Siberian Husky',
 'Vizsla',
 'Yorkie']


cards_class = ['ace of clubs','ace of diamonds', 'ace of hearts', 'ace of spades','eight of clubs', 'eight of diamonds','eight of hearts','eight of spades','five of clubs',
 'five of diamonds',
 'five of hearts',
 'five of spades',
 'four of clubs',
 'four of diamonds',
 'four of hearts',
 'four of spades',
 'jack of clubs',
 'jack of diamonds','jack of hearts',
 'jack of spades',
 'joker',
 'king of clubs',
 'king of diamonds',
 'king of hearts',
 'king of spades',
 'nine of clubs',
 'nine of diamonds',
 'nine of hearts',
 'nine of spades',
 'queen of clubs',
 'queen of diamonds',
 'queen of hearts',
 'queen of spades',
 'seven of clubs',
 'seven of diamonds',
 'seven of hearts',
 'seven of spades',
 'six of clubs',
 'six of diamonds',
 'six of hearts',
 'six of spades',
 'ten of clubs',
 'ten of diamonds',
 'ten of hearts',
 'ten of spades',
 'three of clubs',
 'three of diamonds',
 'three of hearts',
 'three of spades',
 'two of clubs',
 'two of diamonds',
 'two of hearts',
 'two of spades']
#HTML Responses
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/models/yoga_pose/yoga_pose.html", response_class=HTMLResponse)
async def read_yoga_pose(request: Request):
    return templates1.TemplateResponse("yoga_pose/yoga_pose.html", {"request": request})

@app.get("/models/weather/weather.html", response_class=HTMLResponse)
async def read_yoga_pose(request: Request):
    return templates1.TemplateResponse("weather/weather.html", {"request": request})

@app.get("/models/sports_ball/sports_ball.html", response_class=HTMLResponse)
async def read_yoga_pose(request: Request):
    return templates1.TemplateResponse("sports_ball/sports_ball.html", {"request": request})

@app.get("/models/mammals/mammals.html", response_class=HTMLResponse)
async def read_yoga_pose(request: Request):
    return templates1.TemplateResponse("mammals/mammals.html", {"request": request})

@app.get("/models/flower/flower.html", response_class=HTMLResponse)
async def read_yoga_pose(request: Request):
    return templates1.TemplateResponse("flower/flower.html", {"request": request})


@app.get("/models/dog_breed/dog_breed.html", response_class=HTMLResponse)
async def read_yoga_pose(request: Request):
    return templates1.TemplateResponse("dog_breed/dog_breed.html", {"request": request})


#Function Converting Img --> Array
def read_file_as_image(data):
    img = Image.open(BytesIO(data)).resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return img_array


# Endpoint for Sports Ball Model
@app.post("/predict_sports_ball")
async def predict_sports_ball(file: UploadFile = File(...)):
    print("Prediction endpoint called")
    file.file.seek(0)
    img = read_file_as_image(await file.read())
    img = np.expand_dims(img, axis=0)

    predicted = sports_ball_model.predict(img)
    result = sports_ball_class[np.argmax(predicted[0])]
    confidence = np.max(predicted[0])

    return {
        'class': result,
        'confidence': round(confidence * 100, 1)
    }
# EndPoint For Flower Model
@app.post("/predict_flower")
async def predict_flower(file: UploadFile = File(...)):
    print("Prediction endpoint called")
    file.file.seek(0)
    img = read_file_as_image(await file.read())
    img = np.expand_dims(img, axis=0)

    predicted = flower_model.predict(img)
    result = flower_class[np.argmax(predicted[0])]
    confidence = np.max(predicted[0])

    return {
        'class': result,
        'confidence': round(confidence * 100, 1)
    }


# EndPoint For dog Model
@app.post("/predict_dog")
async def predict_dog(file: UploadFile = File(...)):
    print("Prediction endpoint called")
    file.file.seek(0)
    img = read_file_as_image(await file.read())
    img = np.expand_dims(img, axis=0)

    predicted = dog_model.predict(img)
    result = dog_class[np.argmax(predicted[0])]
    confidence = np.max(predicted[0])

    return {
        'class': result,
        'confidence': round(confidence * 100, 1)
    }



# Endpoint for Weather Model
@app.post("/predict_weather")
async def weather(file: UploadFile = File(...)):
    print("Prediction endpoint called")
    file.file.seek(0)
    img = read_file_as_image(await file.read())
    img = np.expand_dims(img, axis=0)

    predicted = weather_model.predict(img)
    result = cards_class[np.argmax(predicted[0])]
    confidence = np.max(predicted[0])

    return {
        'class': result,
        'confidence': round(confidence * 100, 1)
    }


# Endpoint for Yoga Pose Model
@app.post("/predict_yoga_pose")
async def predict_yoga_pose(file: UploadFile = File(...)):
        print("Prediction endpoint called")
        file.file.seek(0)
        img = read_file_as_image(await file.read())
        img = np.expand_dims(img, axis=0)

        predicted = yoga_pose_model.predict(img)
        result = yoga_class[np.argmax(predicted[0])]
        confidence = np.max(predicted[0])

        return {
            'class': result,
            'confidence': round(confidence * 100, 1)
        }



#Endpoint for Mammals Model
@app.post("/predict_mammals")
async def predict_mammals(file: UploadFile = File(...)):
    print("Prediction endpoint called")
    file.file.seek(0)
    img = read_file_as_image(await file.read())
    img = np.expand_dims(img, axis=0)

    predicted = mammals_model.predict(img)
    result = mammals_class[np.argmax(predicted[0])]
    confidence = np.max(predicted[0])

    return {
        'class': result,
        'confidence': round(confidence * 100, 1)
    }

# Run The Server In Localhost via Uvicorn
if __name__ == '__main__':
     import os
     uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT',10000)))

