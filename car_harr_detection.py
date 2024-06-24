import cv2
import numpy as np
import os

# Load the pre-trained Haar Cascade classifier for car detection
car_cascade = cv2.CascadeClassifier('cars.xml')

# Load the background image to initialize the background model
background = cv2.imread('car_images/bg.jpg')

# List of test image paths
test_images = ['car_images/bg.jpg','car_images/no_detection_2024-05-13 18:48:21.832544..jpg', 'car_images/no_detection_2024-05-13 19:09:18.262933..jpg']

# Function to put text on an image
def put_text(image, text, position=(30, 100)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_color = (255, 255, 255)
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

# Function to check if an image contains a car using Haar Cascade
def detect_car(car_cascade, test_image_path):
    # Load the test image
    test_image = cv2.imread(test_image_path)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Detect cars in the image
    cars = car_cascade.detectMultiScale(test_gray, scaleFactor=1.1, minNeighbors=2, minSize=(60, 60), maxSize=(300, 300))

    # Draw bounding boxes around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert 2D images to 3D for display
    test_image_3d = cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR)

    # Put text on the image
    put_text(test_image, "Original with Bounding Boxes")
    put_text(test_image_3d, "Grayscale Image")

    # Show combined image
    combined_image = np.hstack((test_image, test_image_3d))
    cv2.imshow('Combined Image', combined_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return len(cars) > 0

# Iterate over the test images and detect cars
for test_image_path in test_images:
    contains_car = detect_car(car_cascade, test_image_path)
    print(f"Image {test_image_path} contains a car: {contains_car}")
