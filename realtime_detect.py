import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = load_model("GTSRBmodel.keras")

# Use same dimensions you trained on
IMG_HEIGHT, IMG_WIDTH = 48, 48

# Updated class_info: serial number → (class_id, label)
class_info = {
    0: ('0', "Speed Limit 20"),
    1: ('1', "Speed Limit 30"),
    2: ('10', "No passing > 3.5 tons"),
    3: ('11', "Right-of-way"),
    4: ('12', "Priority road"),
    5: ('13', "Yield"),
    6: ('14', "Stop"),
    7: ('15', "No vehicles"),
    8: ('16', "No vehicles > 3.5 tons"),
    9: ('17', "No entry"),
    10: ('18', "General caution"),
    11: ('19', "Left curve"),
    12: ('2', "Speed Limit 50"),
    13: ('20', "Right curve"),
    14: ('21', "Double curve"),
    15: ('22', "Bumpy road"),
    16: ('23', "Slippery road"),
    17: ('24', "Road narrows"),
    18: ('25', "Road work"),
    19: ('26', "Traffic signals"),
    20: ('27', "Pedestrians"),
    21: ('28', "Children crossing"),
    22: ('29', "Bicycles crossing"),
    23: ('3', "Speed Limit 60"),
    24: ('30', "Ice/snow"),
    25: ('31', "Wild animals"),
    26: ('32', "End restrictions"),
    27: ('33', "Turn right"),
    28: ('34', "Turn left"),
    29: ('35', "Ahead only"),
    30: ('36', "Go straight or right"),
    31: ('37', "Go straight or left"),
    32: ('38', "Keep right"),
    33: ('39', "Keep left"),
    34: ('4', "Speed Limit 70"),
    35: ('40', "Roundabout"),
    36: ('41', "End no passing"),
    37: ('42', "End no passing > 3.5 tons"),
    38: ('5', "Speed Limit 80"),
    39: ('6', "End Speed Limit 80"),
    40: ('7', "Speed Limit 100"),
    41: ('8', "Speed Limit 120"),
    42: ('9', "No passing")
}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Grab a center square ROI from the frame
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    roi = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

    # Resize and normalize
    img = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    # Show prediction if confidence is high
    if confidence > 0.70:
        class_label = class_info[class_id][1]  # Extract label from class_info
        class_code = class_info[class_id][0]   # Optional: use class_id (e.g. '5', '20') if needed
        label = f"{class_code} | {class_label} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
