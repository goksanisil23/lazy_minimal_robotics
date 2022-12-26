import cv2

# Open the default camera
camera = cv2.VideoCapture(0)

# Set the width and height of the camera frames
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set the index for the saved images
image_index = 0

while True:
    # Read a frame from the camera
    success, frame = camera.read()

    # Check if the frame was successfully read
    if not success:
        break

    # Display the frame
    cv2.imshow("Camera", frame)

    # Check if the 's' key was pressed
    cv2.waitKey(100)

    # Save the frame to a file
    cv2.imwrite("imgs/image{}.jpg".format(image_index), frame)
    image_index += 1


# Release the camera and destroy the window
camera.release()
cv2.destroyAllWindows()
