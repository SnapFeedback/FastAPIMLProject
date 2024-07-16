from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import cv2
import dlib
import base64

app = FastAPI()

# Load the pre-trained model
parallel_model = tf.keras.models.load_model('models/parallel_model.h5')
detector = dlib.get_frontal_face_detector()


class ImageData(BaseModel):
    image_data: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the SnapFeedBack Emotion Detection API"}


@app.post("/predict")
@app.post("/predict")
def predict_emotion(image: ImageData):
    results = []

    # Decode the base64 image
    try:
        img_data = base64.b64decode(image.image_data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Decoded image is None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

    # Detect faces
    faces = detector(img)
    if len(faces) == 0:
        return {"detail": "No face detected"}

    rois = []
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Ensure the coordinates are within the image boundaries
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            continue

        face_roi = cv2.resize(img[y1:y2, x1:x2], (48, 48))
        rois.append(face_roi)

    if len(rois) == 0:
        return {"detail": "No valid face found"}

    for roi in rois:
        roi = np.array(roi).reshape(-1, 48, 48, 1)

        # Make predictions
        predictions = parallel_model.predict(roi)

        result = {
            "boredom": f"{round(float(predictions[0][0][1]), 3)}",
            "engagement": f"{round(float(predictions[1][0][1]), 3)}",
            "confusion": f"{round(float(predictions[2][0][1]), 3)}",
            "frustration": f"{round(float(predictions[3][0][1]), 3)}"
        }
        results.append(result)

    return results