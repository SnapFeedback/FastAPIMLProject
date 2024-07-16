import requests
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def predict_image(image_path):
    url = "http://127.0.0.1:8000/predict"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    image_data = get_base64_image(image_path)
    payload = {"image_data": image_data}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def main():
    # Test with a bored face image
    bored_face_image_path = "TestImage/Bored-face-pictures-pictures.jpg"
    bored_face_response = predict_image(bored_face_image_path)
    print("Response for bored face image:", bored_face_response)

    # Test with an image with no face
    no_face_image_path = "TestImage/mountain-lake-nature-forest-landscape-scenery-4K-157.jpg"
    no_face_response = predict_image(no_face_image_path)
    print("Response for no face image:", no_face_response)

    # Test with two person face with one on the bacl
    two_face_image_path = "TestImage/TwoFace.jpeg"
    two_face_response = predict_image(two_face_image_path)
    print("Response for two person face image:", two_face_response)

    # Test with man face
    man_face_image_path = "TestImage/man.jpeg"
    man_face_response = predict_image(man_face_image_path)
    print("Response for man face image:", man_face_response)

    # Test with focus face
    focus_face_image_path = "TestImage/focus.jpg"
    focus_face_response = predict_image(focus_face_image_path)
    print("Response for focus face image:", focus_face_response)

if __name__ == "__main__":
    main()