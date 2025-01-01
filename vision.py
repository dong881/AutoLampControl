import cv2
import numpy as np
from openvino.runtime import Core

# Initialize OpenVINO inference engine
ie = Core()

# Load the network model
model = ie.read_model(model="person-detection-retail-0013.xml")
compiled_model = ie.compile_model(model=model, device_name="MYRIAD")  # Use "CPU" if MYRIAD is not available

# Get input and output layer info
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Set up video capture (webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame (resize to model's input shape and change format)
    input_shape = input_layer.shape
    resized_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))  # Resize to model input size (W, H)
    input_image = np.transpose(resized_frame, (2, 0, 1))  # Change data layout from HWC to CHW
    input_image = input_image[np.newaxis, :]  # Add batch dimension [1, 3, H, W]
    input_image = input_image.astype(np.float32)  # Convert to float32 if necessary

    # Perform inference
    result = compiled_model([input_image])

    # Parse the inference result
    boxes = result[output_layer][0][0]  # Detected bounding boxes

    for box in boxes:
        # Box format: [image_id, label, confidence, x_min, y_min, x_max, y_max]
        confidence = box[2]
        if confidence > 0.5:  # Consider detections with confidence > 50%
            x_min, y_min, x_max, y_max = (box[3:] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]).astype(int)
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # If the person is sitting (based on box height)
            if (y_max - y_min) / frame.shape[0] > 0.5:
                print("True - 有人坐下")
            else:
                print("False - 無人坐下")

    # Display the frame with detections
    cv2.imshow("Person Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
