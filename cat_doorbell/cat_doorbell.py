
# import the libraries
import cv2
import numpy as np
  
#image pre-processing function  
def preprocessing(frame):
   # convert the image to grayscale format
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('grayscale', img_gray)

    #histogram equalization
    gray_img_eqhist=cv2.equalizeHist(img_gray)
    #cv2.imshow('grayscale histogram equalized', gray_img_eqhist)

    #denoise grayscale
    ddenoised = cv2.fastNlMeansDenoising(gray_img_eqhist,  None, 3, 4, 2)
    #cv2.imshow('grayscale denoised', ddenoised)

    #gaussian blurring
    blur = cv2.GaussianBlur(ddenoised,(5,5),0)
    #cv2.imshow('grayscale blurred', blur)

    #Otsu thresholding
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #cv2.imshow('thresholded', th3)
    
    #edge detection
    edgedet = cv2.Canny(image=blur, threshold1=120, threshold2=200) # Combined X and Y Sobel Edge Detection
    #cv2.imshow('edge detected', edgedet)

    #find the intersection of edge detected and binarized
    img_bwa = cv2.bitwise_and(th3,edgedet)
    #cv2.imshow('intersection ', img_bwa)

    #erosion
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.dilate(img_bwa,kernel,iterations = 1)   
    #cv2.imshow('eroded intersection ', erosion)

    #returns 
    return erosion, blur

#frame averaging
def img_averaging(vid):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    np_frame = np.array(frame).astype(np.float32)
    
    ret, frame1 = vid.read()
    np_frame1 = np.array(frame1).astype(np.float32)
    
    ret, frame2 = vid.read()
    np_frame2 = np.array(frame2).astype(np.float32)
    
    ret, frame3 = vid.read()
    np_frame3 = np.array(frame3).astype(np.float32)
    
    average_frame = (np_frame+np_frame1+np_frame2+np_frame3)/4
    average_frameint = average_frame .astype(np.uint8)
    return average_frameint

    #return(cv2.add(frame,frame1,frame2,frame3)/4)


# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  


    # Display the resulting frame
    cv2.imshow(' frame ', frame)
      
    avg = img_averaging(vid)
    cv2.imshow(' averaged ', avg)


    erosion, blur = preprocessing(avg)
    cv2.imshow(' eroded intersection ', erosion)
    cv2.imshow(' blurred  ', blur)
    # image detection with CNN tutorial
    # https://pyimagesearch.com/2020/06/22/turning-any-cnn-image-classifier-into-an-object-detector-with-keras-tensorflow-and-opencv/

    #motion detection
    #https://www.life2coding.com/opencv-simple-motion-detection/

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
