import time

from flask import Flask, request
from waitress import serve
from detection_model import modelPrediction
from flask_cors import CORS, cross_origin
import io

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


class_names = ['F_Banana', 'F_Lemon', 'F_Lulo', 'F_Mango', 'F_Orange', 'F_Strawberry', 'F_Tamarillo',
               'F_Tomato', 'S_Banana', 'S_Lemon', 'S_Lulo', 'S_Mango', 'S_Orange', 'S_Strawberry', 'S_Tamarillo', 'S_Tomato']
batch_size = 32
img_height = 180
img_width = 180
num_classes = 16


def generateResponse(imgPath, freshness, outRequest, statuscode, start):

    if freshness[0] == "S":
        food_detection = "STALE"
    elif freshness[0] == "F":
        food_detection = "FRESH"
    else:
        food_detection = "NA"

    if (outRequest == "NA"):
        if (statuscode == "E50063"):
            return {"imgpath": imgPath, "foodStatus": food_detection, "foodLabel": outRequest, "statusCode": statuscode, "statusMessage": 'UNABLE TO LOAD MODEL', "timeTaken": time.time()-start}, 500
        elif (statuscode == "E50064"):
            return {"imgpath": imgPath, "foodStatus": food_detection, "foodLabel": outRequest, "statusCode": statuscode, "statusMessage": 'INPUT IMAGE NOT FOUND', "timeTaken": time.time()-start}, 500
        else:
            print("UNHANDLED EXCEPTION")
            return {"imgpath": imgPath, "foodStatus": food_detection, "foodLabel": outRequest, "statusCode": statuscode, "statusMessage": 'UNABLE TO LOAD MODEL', "timeTaken": time.time()-start}, 500
    else:
        return {"imgpath": imgPath, "foodStatus": food_detection,  "foodLabel": outRequest, "statusCode": statuscode, "statusMessage": 'FRESHNESS DETECTED SUCCESSFULLY', "timeTaken": time.time()-start}, 200


@app.errorhandler(Exception)
def handle_exception(e):
    # print(e)
    start = time.time()
    resp, httpCode = generateResponse("NA", "NA", "NA", 'E50010', start)
    return resp, 500


@app.route('/', methods=["POST"])
@cross_origin()
def damage_detection_api():
    start = time.time()
    f = request.files['file']
    imageBytes = io.BytesIO(f.read())
    try:
        out_label, freshness, statuscode = modelPrediction(
            imageBytes, img_height, img_width, num_classes, class_names)
        resp, httpCode = generateResponse(
            'Success', freshness, out_label, statuscode, start)

    except:
        resp, httpCode = generateResponse(
            'Error', "NA", "NA", "Aaaaaaa", start)
    return resp, httpCode


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
