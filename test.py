import requests
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_predict(image_path):
    url = "http://127.0.0.1:8000/predict"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    image_data = get_base64_image(image_path)
    payload = {"image_data": image_data}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def main():
    # Test with a bored face image
    bored_face_image_path = "TestImage/Bored-face-pictures-pictures.jpg"  # Change to the image path of a bored face
    bored_face_response = test_predict(bored_face_image_path)
    print("Response for bored face image:", bored_face_response)

    # Test with an image with no face
    no_face_image_path = "TestImage/mountain-lake-nature-forest-landscape-scenery-4K-157.jpg"  # Change this to your actual image path
    no_face_response = test_predict(no_face_image_path)
    print("Response for no face image:", no_face_response)

if __name__ == "__main__":
    main()
