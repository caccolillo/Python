import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        #save image in colour
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        #convert image in grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_name_gray = "opencv_frame_gray{}.png".format(img_counter)
        cv2.imwrite(img_name_gray, grayscale)
        print("{} written!".format(img_name))
        img_counter += 1

        # Detect faces
        faces = face_cascade.detectMultiScale(grayscale, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(grayscale, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the output
        cv2.imshow('img', grayscale)


cam.release()

cv2.destroyAllWindows()
