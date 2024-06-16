# Helmet Detection Project

This project uses the YOLO (You Only Look Once) object detection algorithm to detect helmets in images and videos. The project includes a web application for uploading and processing images to identify whether a person is wearing a helmet.

## Features

- **Image Upload**: Allows users to upload images for helmet detection.
- **YOLOv3 Integration**: Utilizes YOLOv3 for accurate helmet detection.
- **Flask Web Application**: Provides an easy-to-use web interface for uploading and viewing results.
- **Real-time Processing**: Processes images in real-time and displays results immediately.
- **Customizable**: Easy to modify for detecting other objects by changing the YOLO configuration and weights.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/helmet-detection.git
    cd helmet-detection
    ```

2. **Create and activate a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download YOLO weights**:

    Download the pre-trained YOLO weights file from the [official YOLO website](https://pjreddie.com/media/files/yolov3.weights) and place it in the project directory.

## Usage

1. **Run the Flask application**:

    ```bash
    python app.py
    ```

2. **Interact with the application**:

    - Open your web browser and navigate to `http://127.0.0.1:5000/`.
    - Upload an image for helmet detection.
    - View the processed image with detected helmets highlighted.

## File Structure

The application is designed with a modular structure for scalability and ease of maintenance:
```bash
Helmet_detection/
│
├── __pycache__/         # Python bytecode files
├── outputs/             # Directory to save processed output files
├── templates/           # HTML templates for the web application
│   └── index.html       # Main page of the web application
├── uploads/             # Directory where uploaded images are stored
├── _config.yml          # Configuration file for the web application
├── app.py               # Main application code
├── helmet.names         # File containing names of the objects/classes for helmet detection
├── yolov3-helmet.cfg    # YOLO configuration file customized for helmet detection
├── yolov3.weights       # YOLO pre-trained weights file
├── requirements.txt     # List of required Python packages
└── README.md            # This readme file
```

## Libraries Used
- Flask: For building the web application.
  - Library: flask
- OpenCV: For image processing and YOLO integration.
  - Library: opencv-python-headless
- NumPy: For numerical operations.
  - Library: numpy
- os: For accessing environment variables and handling other OS-level operations.
  - Library: os (standard Python library)

## Flowchart
  
  ![image](https://github.com/Bhaskar-10/Helmet-Detection-YOLOV3/assets/116245937/3ea78a7f-3d1c-4c4d-8dea-ad64b0ebfe08)

## Output

   ![image_test_helmet](https://github.com/Bhaskar-10/Helmet-Detection-YOLOV3/assets/116245937/32938b30-31f7-4b16-ad8a-379a1a8b9f36)

   https://github.com/Bhaskar-10/Helmet-Detection-YOLOV3/assets/116245937/29e74c19-b0d7-4bc5-bfdb-82a54fd0abac



## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Flask for providing an excellent framework for building web applications.
- OpenCV for powerful image processing capabilities.
- YOLO for the robust object detection algorithm.
