
# add camera photo analysis API and text results to photo

import sys
import numpy
import cv2
import time
import matplotlib.pyplot as plt
import textwrap
from luxand import luxand

camera = cv2.VideoCapture(0)

# Set the width and height of the camera
camera.set(3, 640)
camera.set(4, 480)

# Set the font for the text
font = cv2.FONT_HERSHEY_SIMPLEX

# Set the counter for the photos
counter = 1

# Take 5 photos
while counter <= 4:
    # Read a frame from the camera
    print("get ready for photo:",counter)
    plt.pause(3)
    ret, frame = camera.read()
    cv2.imwrite("photo" + str(counter) + ".jpg", frame)
    img = cv2.imread("photo" + str(counter) + ".jpg")


    # Add text to the bottom of the photo
    # cv2.putText(frame, "Photo "+str(counter)+" and then Love is the answer now!",
    #             (10, 460), font, 1, (255, 255,255), 2,
    #             cv2.LINE_AA)
    # image = cv2.imread("photo.jpg")
    # Calculate the x and y position of the text
    text = "Love is the answer now now now now now now d "
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_thickness = 1
    height, width, channel = img.shape
    text_size = cv2.getTextSize(text, font, font_size, font_thickness)[0]
    x = int((width - text_size[0]) / 2)
    y = int(height - 50)
    wrapped_text = textwrap.wrap(text, width=35)
    for i, line in enumerate(wrapped_text):
        print("line",line,"counter is", counter, "line 47" )
        print("i",i)
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        gap = textsize[1] + 10

        y = int((img.shape[0] + textsize[1]) //1.22) + i * gap
        x = int((img.shape[1] - textsize[0]) // 2)

        cv2.putText(img, line, (x, y), font,
                    font_size,
                    (255, 255, 0),
                    font_thickness,
                    lineType=cv2.LINE_AA)

    # Display the image using matplotlib
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Photo adjusted" + str(counter))
    plt.pause(3)  # Pause for 1 second
    plt.clf()  # Clear the figure for the next image

    # Save the photo
    cv2.imwrite("photo adjusted" + str(counter) + ".jpg", img)

    # Increment the counter
    print("Bottom of list.. by counter")
    counter += 1
# Release the camera
camera.release()

# Destroy all windows
cv2.destroyAllWindows()
