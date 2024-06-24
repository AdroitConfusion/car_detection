import cv2
import numpy as np
import os
import argparse

# Load the background image
background = cv2.imread(r'car_images/bg.jpg')
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
# back_sub.apply(background, learningRate=1)

# List of test image paths
# test_images = ['car_images/bg.jpg','car_images/no_detection_2024-05-13 18:48:21.832544..jpg', 'car_images/no_detection_2024-05-13 18:59:03.900086..jpg','car_images/no_detection_2024-05-13 19:09:18.262933..jpg']
image_folder = 'car_images'
test_images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Get labels
labels = []
labels_file_path = 'labels.txt'
try:
    with open(labels_file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip().lower()
            if stripped_line == "true":
                labels.append(True)
            elif stripped_line == "false":
                labels.append(False)
            else:
                print(f"Warning: Invalid line encountered: {line.strip()}")
except FileNotFoundError:
    print(f"Error: The file {labels_file_path} does not exist.")

# Function to put text on an image
def put_text(image, text, position=(30, 100)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_color = (255, 255, 255)
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

def convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
    # Calculate center coordinates
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    
    # Calculate width and height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return x_center, y_center, width, height

# Function to check if an image contains a car
def detect_car(background_gray, test_image_path, show_images=False):
    # Load the test image
    test_image = cv2.imread(test_image_path)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the background and the test image
    diff = cv2.absdiff(background_gray, test_gray)
    # diff = back_sub.apply(test_image, learningRate=0)
    
    # Apply blur
    blur = cv2.GaussianBlur(diff, (7,7), 0)
    # blur = cv2.bilateralFilter(diff, 9, 75, 75)
    # _, thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    

    # Apply a binary threshold to the difference image
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to remove noise and fill gaps
    kernel_open = np.ones((4, 4), np.uint8)
    kernel_close = np.ones((10, 10), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)
       
    # Find contours in the edge-detected image
    stencil_edges = cv2.Canny(thresh, 50, 150)
    stencil_contours, _ = cv2.findContours(stencil_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Specify the fill color (white)
    fill_color = [255, 255, 255]

    # Create a stencil (mask) with zeros
    stencil = np.zeros(thresh.shape).astype(thresh.dtype)

    # Fill the contours in the stencil
    cv2.fillPoly(stencil, stencil_contours, fill_color)

    # Apply the stencil to the original image
    stencil_img = cv2.bitwise_or(thresh, stencil)

    # Apply Canny edge detection
    edges = cv2.Canny(stencil_img, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Approximate contours
    approx_contours = []
    for contour in contours:
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, closed=True)
        approx = cv2.approxPolyDP(hull, 0.03 * peri, closed=True)
        approx_contours.append(approx)
    # approx_contours = [cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) for contour in contours]

    # Draw contours
    image_height, image_width = test_image.shape[:2]
    image_contours  = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image_approx_contours = image_contours.copy()
    cv2.drawContours(image_contours, contours, -1, (255, 0, 0), 3)
    cv2.drawContours(image_approx_contours, approx_contours, -1, (255, 0, 0), 3)

    output_file_path = 'bounding_boxes.txt'
    with open(output_file_path, 'a') as output_file:
        # Function to check if a contour might be a car
        def is_car_contour(contour):
            
            # Calculate the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Check extent
            contour_area = cv2.contourArea(contour)
            rect_area = w * h
            extent = float(contour_area) / rect_area if rect_area > 0 else 0
            
            if 600000 < rect_area < (image_width * image_height - 10000) and 1 < aspect_ratio < 3 and 0.65 < extent:
                # print(f"Contour Area: {contour_area}\nRect Area: {rect_area}")
                return True
            return False

        # Loop over the contours and check if they are likely to be cars
        is_car = False
        for contour in approx_contours:
            if is_car_contour(contour):
                x_min, y_min, bbox_width, bbox_height = cv2.boundingRect(contour)
                x_max = x_min + bbox_width
                y_max = y_min + bbox_height

                x_center, y_center, width, height = convert_to_yolo_format(x_min, y_min, x_max, y_max, image_width, image_height)
                # Write the coordinates to the file
                output_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                cv2.rectangle(test_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
                is_car = True
                # break

   
    if show_images:
        # Convert 2D images to 3D
        diff_3d = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        # blur_3d = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        # stencil_3d = cv2.cvtColor(stencil, cv2.COLOR_GRAY2BGR)
        thresh_3d = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        stencil_img_3d = cv2.cvtColor(stencil_img, cv2.COLOR_GRAY2BGR)
        edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Put text on each image
        put_text(test_image, "Original with Bounding Boxes")
        put_text(diff_3d, "Background Subtraction")
        # put_text(blur_3d, "Blur")
        put_text(thresh_3d, "Threshold")
        # put_text(stencil_3d, "Stencil")
        put_text(stencil_img_3d, "Stenciled Image")
        # put_text(edges_3d, "Edges")
        put_text(image_contours, "Contours")
        put_text(image_approx_contours, "Approx Contours")

        # Show combined image
        row_one = np.hstack((test_image, diff_3d))
        row_two = np.hstack((thresh_3d, stencil_img_3d))
        row_three = np.hstack((image_contours, image_approx_contours))
        combined_image = np.vstack((row_one, row_two, row_three))
        cv2.imshow('Combined Image', combined_image)
        cv2.imwrite("./threshold.jpeg", thresh_3d)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return is_car


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Detect cars in images.')
    parser.add_argument('--show-images', action='store_true', help='Show images with detection results')
    args = parser.parse_args()

    # Iterate over the test images and detect cars
    predictions = []
    for i, test_image_path in enumerate(test_images):
        contains_car = detect_car(background_gray, test_image_path, show_images=args.show_images)
        # if contains_car != labels[i]:
        print(test_image_path)
        predictions.append(contains_car)
        # print(f"Image {test_image_path} contains a car: {contains_car}")

    # Calculate score
    print(f"Labels Length: {len(labels)}, Predictions Length: {len(predictions)}")
    if len(labels) != len(predictions):
        raise ValueError("Lists must be of the same length")
        
    differences = sum(1 for a, b in zip(labels, predictions) if a != b)
    false_negs = sum(1 for a,b in zip(labels, predictions) if a and not b)
    false_pos = differences - false_negs
    print = print(f"Incorrect: {differences}\nFalse Negatives: {false_negs}\nFalse Positives: {false_pos}\nScore: {100 * ((len(labels) - differences) / len(labels))}")