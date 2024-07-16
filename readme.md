# SnapFeedBack Emotion Detection API

## Introduction

Welcome to the SnapFeedBack Emotion Detection API. This API leverages deep learning to detect emotions such as boredom, engagement, confusion, and frustration from grayscale images. It uses a pre-trained convolutional neural network model to predict these emotions from detected faces within the provided image.

## API Endpoints

### 1. Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a welcome message.
- **Response**:
  ```json
  {
      "message": "Welcome to the SnapFeedBack Emotion Detection API"
  }
### 2. Predict Emotion
- **URL**: /predict
- **Method**: `POST`
- **Description:** Accepts a grayscale image encoded in base64 format and returns the predicted emotions for any faces detected within the image with four main category,boredom,engagement,confusion and frustration
- **Request Body**:
    ```json
    {
        "image_data": "base64_encoded_image_string"
    }
- **Response**:
- Success:
    ```json
        {
            "boredom": 0.654,
            "engagement": 0.324,
            "confusion": 0.02,
            "frustration": 0.01
        }
- Failure:
    ```json
        {
            "detail": "Error message"
        }
- **Usage**
1. Testing with a Sample Script
You can use the following Python script to test the API. Make sure to replace the image paths with your actual image files.

    ### python
    
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
            bored_face_image_path = "path_to_bored_face_image.jpg"
            bored_face_response = test_predict(bored_face_image_path)
            print("Response for bored face image:", bored_face_response)
        
            # Test with an image with no face
            no_face_image_path = "path_to_no_face_image.jpg"
            no_face_response = test_predict(no_face_image_path)
            print("Response for no face image:", no_face_response)
    
            if __name__ == "__main__":
                main()
                
- Above test code can be found in the test.py, test can be done once the server is deploy in dev and using the commang
    ```bash
        python test.py
- JavaScript local test sample can be as below
       
- ```javascript
        const axios = require('axios');

        // Function to test the API endpoint
        async function testAPI() {
            try {
                // Replace with your API endpoint URL
                const apiUrl = 'http://127.0.0.1:8000/predict';
    
                // Example base64 encoded image data (replace with your actual image data)
                const imageData = 'base64_encoded_image_data_here';
    
                // Example payload with image data
                const payload = {
                    image_data: imageData
                };
    
                // Make a POST request to the API endpoint
                const response = await axios.post(apiUrl, payload, {
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                });
    
                // Log the response from the API
                console.log('API Response:', response.data);
            } catch (error) {
                // Log any errors that occur during the request
                console.error('Error:', error.message);
            }
            }
    
        // Call the function to test the API
        testAPI();

## 2. Running the API Locally
- **Download and Install Anaconda**
    - Installer: https://www.anaconda.com/download/success
    - Command line in linux os: https://docs.anaconda.com/anaconda/install/linux/
- **Clone the Repository**:

    ```bash
        git clone https://github.com/PatrickTongg/FastAPIMLProject.git
- **Create a Conda Environment**:
    ```bash
        conda env create -f environment.yml
        conda activate emotion-detection-env
- **Start the API in development mode**:

    ```bash
        fastapi dev main.py
- The API will be available at http://127.0.0.1:8000.


## 3. Deployment
- **Development Deployment**
    For development deployment, you can run the API locally  as shown above. This allows you to test and debug the application on your local machine.

- **Production Deployment**
For production deployment, it's recommended to use a production-ready server like Gunicorn with Uvicorn workers or using .
  - Resource for deployment

    - https://www.youtube.com/watch?v=SgSnz7kW-Ko
    - https://fastapi.tiangolo.com/deployment/

## 4. Remarks
- current version of application is developed and tested with anacoda3 env on MacOS Sonona 14.5 