from flask import Flask, render_template, Response, request, jsonify
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
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

labels = open("yolo-coco/coco.names").read().strip().split("\n")

last_detection = ""

camera_active = False


# ---------------- LOGIC ---------------- #

def get_position(x, w, frame_width):
    center_x = x + w // 2

    if center_x <= frame_width // 3:
        return "left"
    elif center_x <= (frame_width // 3) * 2:
        return "center"
    else:
        return "right"


def calculate_distance(height):
    focal_length = 100
    real_height = 50
    return (real_height * focal_length) / height


def process_frame(frame):
    global last_detection

    H, W = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416),
        swapRB=True, crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4:
                box = detection[0:4] * np.array([W, H, W, H])
                cx, cy, w, h = box.astype("int")

                x = int(cx - w / 2)
                y = int(cy - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            label = labels[class_ids[i]]

            position = get_position(x, w, W)
            distance = calculate_distance(h)

            text = f"{label} at {distance:.0f}cm {position}"
            last_detection = text

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

    return frame


# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/start')
def start():
    global camera_active
    camera_active = True
    return "started"


@app.route('/stop')
def stop():
    global camera_active
    camera_active = False
    return "stopped"


@app.route('/detect', methods=['POST'])
def detect():
    if not camera_active:
        return jsonify({'image': None, 'text': 'Camera stopped'})

    data = request.json['image']

    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    frame = process_frame(frame)

    _, buffer = cv2.imencode('.jpg', frame)
    encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': encoded,
        'text': last_detection
    })


@app.route('/get_detection')
def get_detection():
    return last_detection


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)