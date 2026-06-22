from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# ---------------- YOLOv4 ---------------- #

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

print("YOLOv4 Loaded")

# ---------------- Global Variables ---------------- #

camera = None

last_detection = ""


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


# ---------------- Frames ---------------- #

def generate_frames():

    global camera
    global last_detection

    while True:

        if camera is None:

            continue

        success, frame = camera.read()

        if not success:

            break

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

                text = f"{label} at {distance:.0f}cm {position}"

                last_detection = text

                cv2.rectangle(

                    frame,

                    (x, y),

                    (x + w, y + h),

                    (0, 255, 0),

                    2

                )

                cv2.putText(

                    frame,

                    text,

                    (x, y - 10),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (0, 255, 0),

                    2

                )

        ret, buffer = cv2.imencode(

            '.jpg',

            frame

        )

        frame = buffer.tobytes()

        yield (

            b'--frame\r\n'

            b'Content-Type: image/jpeg\r\n\r\n'

            + frame +

            b'\r\n'

        )


# ---------------- Routes ---------------- #

@app.route('/')
def index():

    return render_template("index.html")


@app.route('/video')
def video():

    return Response(

        generate_frames(),

        mimetype='multipart/x-mixed-replace; boundary=frame'

    )


@app.route('/start_camera')
def start_camera():

    global camera

    if camera is None:

        camera = cv2.VideoCapture(0)

    return "Started"


@app.route('/stop_camera')
def stop_camera():

    global camera

    if camera is not None:

        camera.release()

        camera = None

    return "Stopped"


@app.route('/get_detection')
def get_detection():

    global last_detection

    return last_detection


# ---------------- Run ---------------- #

if __name__ == "__main__":

    app.run(debug=True)