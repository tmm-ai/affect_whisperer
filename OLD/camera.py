import sys

# sys.path.append('/Users/Tom/opt/anaconda3/lib/python3.9/site-packages')
import time
import cv2

# Open the camera
cap = cv2.VideoCapture(0)
time.sleep(0.3)
# Take a photo
ret, frame = cap.read()

# Save the photo to a file
cv2.imwrite("photo_mac_test.jpg", frame)

# Open the photo
image = cv2.imread("photo_mac_test.jpg")

# Display the photo
# cv2.imshow("Photo", image)
cv2.imshow("Photo", frame)
# Release the camera
cap.release()
# Wait for the user to close the window
cv2.waitKey(0)
# Destroy the window
cv2.destroyAllWindows()