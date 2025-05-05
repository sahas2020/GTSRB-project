import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load model
model = load_model("GTSRBmodel.keras")

# Image size the model expects
IMG_HEIGHT, IMG_WIDTH = 48, 48

# Updated class_info dictionary (serial: (class_id, label))
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

# Load your test image
image_path = "00008.png"  # 👈 Replace this with your image path
img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)
confidence = np.max(prediction)

# Get class_id and label from updated class_info
class_id, class_label = class_info[predicted_class]
label = f"Class ID: {class_id} | {class_label} ({confidence*100:.2f}%)"
print(label)

# Display image with label
img_bgr = cv2.imread(image_path)
img_bgr = cv2.resize(img_bgr, (300, 300))
cv2.putText(img_bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imshow("Prediction", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
