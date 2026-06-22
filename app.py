from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)


# ---------------- YOLO ---------------- #

net = cv2.dnn.readNet(

    "yolo-coco/yolov4-tiny.weights",

    "yolo-coco/yolov4-tiny.cfg"

)

layer_names = net.getLayerNames()

output_layers = [

    layer_names[i - 1]

    for i in net.getUnconnectedOutLayers()

]

labels = open(

    "yolo-coco/coco.names"

).read().strip().split("\n")

print("YOLO Loaded")


# ---------------- Position ---------------- #

def get_position(x, w, frame_width):

    center_x = x + w // 2

    if center_x <= frame_width // 3:

        return "left"

    elif center_x <= (frame_width // 3) * 2:

        return "center"

    else:

        return "right"


# ---------------- Distance ---------------- #

def calculate_distance(height):

    focal_length = 100

    real_height = 50

    distance = (real_height * focal_length) / height

    return distance


# ---------------- Home ---------------- #

@app.route('/')

def index():

    return render_template("index.html")


# ---------------- Detect ---------------- #

@app.route('/detect', methods=['POST'])

def detect():

    data = request.json['image']


    image_data = data.split(',')[1]

    image_bytes = base64.b64decode(image_data)


    np_arr = np.frombuffer(

        image_bytes,

        np.uint8

    )


    frame = cv2.imdecode(

        np_arr,

        cv2.IMREAD_COLOR

    )


    H, W = frame.shape[:2]


    blob = cv2.dnn.blobFromImage(

        frame,

        1 / 255.0,

        (416, 416),

        swapRB=True,

        crop=False

    )


    net.setInput(blob)


    outputs = net.forward(output_layers)


    boxes = []

    confidences = []

    class_ids = []

    detection_text = ""


    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]


            if confidence > 0.4:


                box = detection[0:4] * np.array(

                    [W, H, W, H]

                )


                centerX, centerY, width, height = box.astype("int")


                x = int(centerX - width / 2)

                y = int(centerY - height / 2)


                boxes.append(

                    [x, y, int(width), int(height)]

                )


                confidences.append(

                    float(confidence)

                )


                class_ids.append(class_id)


    idxs = cv2.dnn.NMSBoxes(

        boxes,

        confidences,

        0.5,

        0.4

    )


    if len(idxs) > 0:


        for i in idxs.flatten():


            x, y, w, h = boxes[i]


            label = labels[class_ids[i]]


            position = get_position(

                x,

                w,

                W

            )


            distance = calculate_distance(h)


            detection_text = (

                f"{label} at "

                f"{distance:.0f} cm "

                f"{position}"

            )


            cv2.rectangle(

                frame,

                (x, y),

                (x + w, y + h),

                (0,255,0),

                2

            )


            cv2.putText(

                frame,

                detection_text,

                (x, y - 10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0,255,0),

                2

            )


    _, buffer = cv2.imencode(

        '.jpg',

        frame

    )


    jpg_as_text = base64.b64encode(

        buffer

    ).decode('utf-8')


    return jsonify(

        {

            "image": jpg_as_text,

            "text": detection_text

        }

    )


# ---------------- Run ---------------- #

if __name__ == "__main__":

    app.run(debug=True)