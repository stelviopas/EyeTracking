import dlib
import numpy as np
from tkinter import *
import face_features_recognition as ff
from eye_center_detector import *
import PIL.ImageTk as Pimtk
from opencv_funcs import *
import PIL.Image as Pim
import polynom_funcs as pf

thresholds = [100, 100]
pxl, pyl, pxr, pyr = None, None, None, None,


class EyeTracker:
	global pxl, pyl, pxr, pyr, thresholds

	def __init__(self, window, window_title):
		self.window = window
		self.window.title(window_title)

		self.cam = cv2.VideoCapture(0)
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
		self.thresholding_button = Button(self.window, text="Start Thresholding", command=self.thresholding)
		self.calibration_button = Button(self.window, text="Start Calibration", command=self.calibration)
		self.tracking_button = Button(self.window, text="Start Live Tracking", command=self.tracking)
		self.thresholding_button.pack()
		self.calibration_button.pack()
		self.tracking_button.pack()
		self.delay = 5
		self.window.mainloop()

	def calibration(self):
		global thresholds, pxl, pyl, pxr, pyr
		c_win = Toplevel(self.window)
		c_win.wm_title("Calibration")

		ws = self.window.winfo_screenwidth()  # width of the screen
		hs = self.window.winfo_screenheight()  # height of the screen
		canvas = Canvas(c_win, width=ws, height=hs)
		canvas.pack()
		calibration_done_slider = Scale(c_win, variable=IntVar(), label='Calibration Done', to=1, orient=HORIZONTAL)
		calibration_done_slider.pack()

		x_div, y_div = 15, 12
		X_values = np.array([ws * 1 / x_div, ws * 1 / 2, ws * (x_div-1) / x_div])
		Y_values = np.array([hs * 1 / y_div, hs * 2 / 4, hs * (y_div-1) / y_div])

		# X_Values x Y_values
		XY_values = np.array([(int(round(x)), int(round(y))) for x in X_values for y in Y_values])

		# npArray with calibration points with respect to users screen. Top-left point of the box, in which circles are drawn
		# create Canvas - tool for drawing images like circles

		var = IntVar()
		print("xy_values:", XY_values)

		def click(event):
			if canvas.find_withtag(CURRENT):
				print("CURRENT", CURRENT)
				canvas.itemconfig(CURRENT, fill="blue")
				canvas.update_idletasks()
				canvas.after(200)
				canvas.itemconfig(CURRENT, fill="red")
				var.set(1)

		list_of_list_of_frames = []	 # wil hold a list of all image-lists constructed

		# draw all 9 circles
		CIRCLE_SIZE = 15
		for (x, y) in XY_values:

			circle = canvas.create_oval(x, y, x + CIRCLE_SIZE, y + CIRCLE_SIZE, fill="red")
			canvas.bind("<Button-1>", click)
			print("waiting...")
			c_win.wait_variable(var)
			print("done waiting.")
			##################################################################################
			## takes images
			image_list = take_and_return_images(5, self.cam)
			list_of_list_of_frames.append(image_list)
			##################################################################################
			canvas.delete(circle)

		# get function that estimates x and y coordinates for each eye

		eye_center_xys_left, eye_center_xys_right = pf.get_mapping_xys(
			list_of_list_of_frames, thresholds, self.detector, self.predictor)

		pxl, pyl = pf.get_polynoms(eye_center_xys_left, XY_values)
		pxr, pyr = pf.get_polynoms(eye_center_xys_right, XY_values)

		#print("pxl, pyl, pxr, pyr ", pxl, pyl, pxr, pyr )

	def thresholding(self):
		global thresholds
		t_win = Toplevel(self.window)
		t_win.wm_title("Thresholding")

		canvas = Canvas(t_win)
		canvas.pack()

		left_slider = Scale(t_win, variable=IntVar(), to=255, label="left eye", orient=HORIZONTAL, length=200)
		right_slider = Scale(t_win, variable=IntVar(), to=255, label="right eye", orient=HORIZONTAL, length=200)
		left_slider.pack()
		right_slider.pack()
		threshold_done_slider = Scale(t_win, variable=IntVar(), label='Thresholds Done', to=1, orient=HORIZONTAL)
		threshold_done_slider.pack()

		left_slider.set(thresholds[0])
		right_slider.set(thresholds[1])

		def update_thresholding():
			global thresholds

			# get frame from video source
			ret, self.left_eye, self.right_eye = self.get_eye_frames()

			# print("ret, left, right", ret, self.left_eye, self.right_eye)
			# get threshold values from sliders
			thresholds[0] = left_slider.get()
			thresholds[1] = right_slider.get()
			if ret:
				self.eye_frames = (Pimtk.PhotoImage(image=Pim.fromarray(self.left_eye)), Pimtk.PhotoImage(image=Pim.fromarray(self.right_eye)))
				canvas.delete('all')
				canvas.create_image(100, 100, image=self.eye_frames[0], anchor='nw')
				canvas.create_image(195, 100, image=self.eye_frames[1], anchor='nw')
				if threshold_done_slider.get() is 0:

					canvas.after(self.delay, update_thresholding)
				else:
					print("tresholds:", thresholds)
					canvas.delete('all')
					canvas.create_text(100, 100,text="can close window now", anchor =NW)

		update_thresholding()

	def get_eye_frames(self):
		global thresholds
		if self.cam.isOpened():
			ret, frame = self.cam.read()

			box = self.detector(frame, 1)
			if len(box) is not 0:
				box = box[0]
				shape = self.predictor(frame, box)
				shape = [(int(s.x), int(s.y)) for s in shape.parts()]
			else:
				return False, None, None
			max_thresh_val = 255

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# print("shape:", shape)
			origin_x_right, origin_y_right, right_eye = ff.extract_eye(ff.RIGHT_EYE_IDs, shape, frame)
			origin_x_left, origin_y_left, left_eye = ff.extract_eye(ff.LEFT_EYE_IDs, shape, frame)

			left_eye_binary_frame, contours_l = threshold_image(left_eye, thresholds[0], max_thresh_val)
			_ = draw_box_and_get_center(contours_l, left_eye)

			right_eye_binary_frame, contours_r = threshold_image(right_eye, thresholds[1], max_thresh_val)
			_ = draw_box_and_get_center(contours_r, right_eye)

			left = np.concatenate((left_eye_binary_frame, left_eye))
			right = np.concatenate((right_eye_binary_frame, right_eye))

			if ret:
				# return success flag and current frame
				return ret, left, right
			else:
				return ret, None
		else:
			return False, None

	def get_frame(self):
		if self.cam.isOpened():
			ret, frame = self.cam.read()
			if ret:
				# Return a boolean success flag and the current frame converted to BGR
				return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			else:
					return ret, None
		else:
			return False, None

	def tracking(self):
		global thresholds, pxl, pyl, pxr, pyr
		tr_win = Toplevel(self.window)
		tr_win.wm_title("Live Tracking")

		ws = self.window.winfo_screenwidth()  # width of the screen
		hs = self.window.winfo_screenheight()  # height of the screen
		canvas = Canvas(tr_win, width=ws, height=hs)
		canvas.pack()

		def update_tracking():
			global thresholds, pxl, pyl, pxr, pyr

			ret, self.frame = self.get_frame()

			if ret:

				dets = self.detector(self.frame, 1)
				if len(dets) is not 0:
					box = dets[0]
					shape = self.predictor(self.frame, box)
					shape = [(int(s.x), int(s.y)) for s in shape.parts()]
					orig_x_l, orig_y_l, left_frame = ff.extract_eye(ff.LEFT_EYE_IDs, shape, self.frame)
					orig_x_r, orig_y_r, right_frame = ff.extract_eye(ff.RIGHT_EYE_IDs, shape, self.frame)
					frame_centers_xys = center_detection([[left_frame, right_frame]], (thresholds[0], thresholds[1]))

					left_frame_xy = frame_centers_xys[0][0]
					right_frame_xy = frame_centers_xys[0][1]

					frame_xl = orig_x_l + left_frame_xy[0]
					frame_yl = orig_y_l + left_frame_xy[1]
					frame_xr = orig_x_r + right_frame_xy[0]
					frame_yr = orig_y_r + right_frame_xy[1]
					cv2.circle(self.frame, (frame_xl, frame_yl), 1, (255, 255, 255), 2)
					cv2.circle(self.frame, (frame_xr, frame_yr), 1, (0, 0, 0), 2)
					xl = int(round(pxl(frame_xl)))
					yl = int(round(pyl(frame_yl)))
					xr = int(round(pxr(frame_xr)))
					yr = int(round(pyr(frame_yr)))

					print("pxl ,pyl:", xl, yl)
					print("pxr ,pyr:", xr, yr)
				canvas.delete('all')
				self.photo = Pimtk.PhotoImage(image=Pim.fromarray(self.frame))
				canvas.create_image(ws / 2 - 320, hs / 2 - 240, image=self.photo, anchor=NW)
				HALF_RADIUS = 25

				canvas.create_oval(xl - HALF_RADIUS, yl - HALF_RADIUS, xl + HALF_RADIUS, yl + HALF_RADIUS, fill="white")
				canvas.create_oval(xr - HALF_RADIUS, yr - HALF_RADIUS, xr + HALF_RADIUS, yr + HALF_RADIUS, fill="black")
				canvas.create_oval((xl+xr)/2 - HALF_RADIUS, (yl+yr)/2 - HALF_RADIUS, (xl+xr)/2 + HALF_RADIUS, (yl+yr)/2 + HALF_RADIUS, fill="gray")
				canvas.after(self.delay, update_tracking)

		canvas.after(self.delay, update_tracking)




if __name__ == '__main__':

	EyeTracker(Tk(), "wup")