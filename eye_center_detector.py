import cv2
from face_features_recognition import circle_detection

def threshold_image(eye_frame, threshold, max_threshold):
	_, threshold_img = cv2.threshold(eye_frame, threshold, max_threshold, cv2.THRESH_BINARY_INV)
	#    Since opencv 3.2 source image is not modified by this function
	contours, _ = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

	return threshold_img, contours


# draws a rectangle around the found contours and calculates center of the box

def draw_box_and_get_center(contours, eye_frame):

	for cnt in contours:
		(x, y, w, h) = cv2.boundingRect(cnt)
		w_half, h_half = int(round(w / 2)), int(round(h / 2))
		# cv2.drawContours(eye_frame, [cnt], -1, (0, 0, 255), 3)
		cv2.rectangle(eye_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
		cv2.line(eye_frame, (x + w_half, y), (x + w_half, y + h), (0, 255, 0), 1)
		cv2.line(eye_frame, (x, y + h_half), (x+w, y + h_half), (0, 255, 0), 1)

		return x+w_half, y+h_half


def center_detection(eye_frame_list, threshold_lr):
	max_thresh_val = 255
	threshold_val_left = threshold_lr[0]
	threshold_val_right = threshold_lr[1]
	eyes_xys = []

	for (left_eye, right_eye) in eye_frame_list:
		left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
		right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
		_, contours_r = threshold_image(right_eye, threshold_val_right, max_thresh_val)
		right_eye_xy = draw_box_and_get_center(contours_r, right_eye)
	
		_, contours_l = threshold_image(left_eye, threshold_val_left, max_thresh_val)
		left_eye_xy = draw_box_and_get_center(contours_l, left_eye)

		eyes_xys.append([left_eye_xy, right_eye_xy])
	return eyes_xys

def circle_detection(eye_frame_list,dp, minDist, param1, param2, minRadius, maxRadius):
	eyes_xys = []

	for (left_eye, right_eye) in eye_frame_list:
		_, left_eye_xy = circle_detection(left_eye, dp, minDist, param1, param2, minRadius, maxRadius)
		_, right_eye_xy = circle_detection(right_eye, dp, minDist, param1, param2, minRadius, maxRadius)

		eyes_xys.append([(left_eye_xy[0][0], left_eye_xy[0][1]), (right_eye_xy[0][0], right_eye_xy[0][1])])
	return eyes_xys