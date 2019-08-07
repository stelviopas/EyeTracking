import numpy as np
import cv2
import argparse
import dlib

FACIAL_LANDMARKS_IDs ={
	"mouth":(48, 68),
	"right_eyebrow": (17, 22),
	"left_eyebrow": (22, 27),
	"right_eye":(36, 42),
	"left_eye": (42, 48),
	"nose": (27, 35)}

RIGHT_EYE_IDs = FACIAL_LANDMARKS_IDs["right_eye"]
LEFT_EYE_IDs = FACIAL_LANDMARKS_IDs["left_eye"]

# initialize dlib face detector (HOG-based) and then create the facial landmark predictor
def predict_face_rectangle(image):
	detector = dlib.get_frontal_face_detector()
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_rectangle = detector(gray_image, 1)
	return face_rectangle


#Rectangle to bounding box
#convert a box, which was predicted by dlib to the format (x,y,width,height)


def rectangle_to_box(box):

	x = box[0][0]
	y = box[0][1]
	width = box[1][0]-x
	height = box[1][1]-y
	return (x,y,width,height)


def shape_to_np(shape):
	#list of x,y coordinates
	xy = np.zeros((68,2),dtype="int")
	#loop over the 68 facial landmarks and convert them to a tuple x,y coordinates
	for i in range (0,68):
		xy[i]=(shape.part(i).x,shape.part(i).y)
	return xy

#downscaling an image with preserved height to width ratio.
#resizing an image means changing the dimension of it. cv2 image It returns a tuple of number of rows, columns and channels (if image is color).If image is grayscale, tuple returned contains only number of rows and columns. So it is a good method to check if loaded image is grayscale or color image.
def downscale(img,scale_percent):
	width=int(img.shape[1]*scale_percent/100)
	height=int(img.shape[0]*scale_percent/100)
	dim=(width,height)
	resized=cv2.resize(img,dim,interpolation = cv2.INTER_AREA)
	return resized


def upscale(img,factor):
	width = int(img.shape[1]*factor)
	height = int(img.shape[0]*factor)
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
	return resized


def predict_face_shape(image, face_rectangle, shape_predictor):
	#load the shape predictor model
	predictor=dlib.shape_predictor(shape_predictor)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#for each face detected
	for (i, rect) in enumerate(face_rectangle):
		#determine face landmarks in gray image
		shape = predictor(gray_image,rect)
		#convert to numpy array
		shape=[(int(s.x), int(s.y)) for s in shape.parts()]
	return shape


def extract_face (image, face_rectangle):

	(x,y,w,h) = rectangle_to_box(face_rectangle)
	face = image[y:y+h, x:x+w]
	return face


def extract_eye(eyeID, shape, image):
	if shape is None:
		return None
	(i,j)=eyeID
	magic = 5

	(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

	return x-magic, y-magic, cv2.medianBlur(image[y-5: y+5+h, x-5: x+5+w], 5)  # image[y-magic: y+5+magic, x-magic: x+magic+w]  # cv2.bilateralFilter(image[y-magic: y+5+magic, x-magic: x+magic+w], 5, 20, 20) #

def circle_detection(img, dp, minDist, par1=100, par2=100, minr=0, maxr=0):

	# detect circles in the image
	circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=par1, param2=par2, minRadius=minr, maxRadius=maxr)

	# ensure at least some circles were found
	if circles is not None:
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		circle_xyrs = []
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(img, (x, y), r, (255, 255, 255), 1)
			cv2.circle(img, (x, y), 1, (255, 255, 255), 1)

			circle_xyrs.append([x, y])

		return img, circle_xyrs

	else:
		return img, None




def detect_circle(img,par1=100,par2=20,minr=10,maxr=30):

	gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 20, param1=par1,param2=par2,minRadius=minr,maxRadius=maxr)
	circles = np.uint16(np.around(circles))


	return circles


def draw_circle(img, circles):
	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

###################################################################
#Visualization	

def draw_face_rectangle (image, face_rectangle):
	for (i,rect) in enumerate(face_rectangle):
		(x,y,w,h)=rectangle_to_box(rect)
		# Arguments : (source image, vertex of the rectangle, opposite vertex, color, optional : type of the line(how fat is the line, default is also ok))
		cv2.rectangle(image, (x,y),(x+w,y+h),(0, 255, 0),2)

def point_face_features (image, shape):
	for (x,y) in shape:
		#(source image, center of the circle, radius, color)
		cv2.circle(image, (x,y), 1, (0,0,255))

def show_pic (descr, image):
	cv2.imshow(descr, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


