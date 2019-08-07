import eye_center_detector as ecd
import face_features_recognition as ff
import numpy as np


def get_mapping_xys_c(list_of_image_lists, detector, predictor, dp, minDist, param1, param2, minRadius, maxRadius):
	# print("list_of_image_lists: ", list_of_image_lists)

	list_of_eye_frame_lists = []  # list containing lists of eye-frames
	list_of_eye_frames_xy_in_orig_img = []  # list of where the eyeframes were cut out from the respective image

	# go through every image-list in this list of lists and extract left and right eye-frames in every image
	# create a list containing lists of eye-frames
	# and a list containing lists of [(x,y), (x,y)] where x,y corner of left/right eye-frame

	for image_list in list_of_image_lists:
		eye_frames_lr = []
		eye_frames_xy_in_orig_img = []
		for image in image_list:
			dets = detector(image, 1)
			if len(dets) is not 0:
				d = dets[0]
				predictions = predictor(image, d)
				shape = [(int(s.x), int(s.y)) for s in predictions.parts()]
				original_eye_l_x, original_eye_l_y, left_eye_frame = ff.extract_eye(ff.LEFT_EYE_IDs, shape, image)
				original_eye_r_x, original_eye_r_y, right_eye_frame = ff.extract_eye(ff.RIGHT_EYE_IDs, shape, image)
				eye_frames_lr.append([left_eye_frame, right_eye_frame])
				eye_frames_xy_in_orig_img.append(
					[(original_eye_l_x, original_eye_l_y), (original_eye_r_x, original_eye_r_y)])

		# print("eye_frames_lr:", eye_frames_lr)
		list_of_eye_frame_lists.append(eye_frames_lr)
		list_of_eye_frames_xy_in_orig_img.append(eye_frames_xy_in_orig_img)
	# print("xy orig:", list_of_eye_frames_xy_in_orig_img)
	# print("list_of_eye_frame_lists:", list_of_eye_frame_lists)

	# create a list with lists of the xy-positions of left and right eye-frames, respectively
	list_of_eyecenter_xy_lists = []

	for eye_list in list_of_eye_frame_lists:
		# ecd.center_detection returns list of [(xl ,yl) , (xr, yr)]'s
		list_of_eyecenter_xy_lists.append(ecd.circle_detection(eye_list, dp, minDist, param1, param2, minRadius, maxRadius))
	# print("list_of_eyecenter_xy_lists:", list_of_eyecenter_xy_lists)

	# find eye center coordinates in starting image
	list_of_eyecenter_xys_in_starting_img_lists = []

	for orig_list, centers_list in zip(list_of_eye_frames_xy_in_orig_img, list_of_eyecenter_xy_lists):

		eyecenter_xys_in_starting_img_list = []

		for ((orig_xl, orig_yl), (orig_xr, orig_yr)), ((eye_cxl, eye_cyl), (eye_cxr, eye_cyr)) in zip(orig_list,
																									  centers_list):
			eyecenter_xys_in_starting_img_list.append(
				[(orig_xl + eye_cxl, orig_yl + eye_cyl), (orig_xr + eye_cxr, orig_yr + eye_cyr)])

		list_of_eyecenter_xys_in_starting_img_lists.append(eyecenter_xys_in_starting_img_list)
	# print("list of eyecenter_xys:", list_of_eyecenter_xys_in_starting_img_lists)

	# calculate mean xys of left and right eye-centers

	list_of_xys_to_map_lists = []

	for xy_list in list_of_eyecenter_xys_in_starting_img_lists:
		sum_xr, sum_yr, sum_xl, sum_yl = 0, 0, 0, 0
		divisor = len(xy_list)

		for ((xc_l, yc_l), (xc_r, yc_r)) in xy_list:
			# print("xl,yl xr,yr:", xc_l, yc_l, xc_r, yc_r)
			sum_xl = sum_xl + xc_l
			sum_yl = sum_yl + yc_l
			sum_xr = sum_xr + xc_r
			sum_yr = sum_yr + yc_r
		list_of_xys_to_map_lists.append(
			[(round(sum_xl / divisor), round(sum_yl / divisor)),
			 (round(sum_xr / divisor), round(sum_yr / divisor))])

	mapping_xys_left = [(x, y) for ((x, y), (_, _)) in list_of_xys_to_map_lists]
	mapping_xys_right = [(x, y) for ((_, _), (x, y)) in list_of_xys_to_map_lists]

	return mapping_xys_left, mapping_xys_right


def get_mapping_xys(list_of_image_lists, thresholds, detector, predictor):
	# print("list_of_image_lists: ", list_of_image_lists)

	list_of_eye_frame_lists = []  # list containing lists of eye-frames
	list_of_eye_frames_xy_in_orig_img = []  # list of where the eyeframes were cut out from the respective image

	# go through every image-list in this list of lists and extract left and right eye-frames in every image
	# create a list containing lists of eye-frames
	# and a list containing lists of [(x,y), (x,y)] where x,y corner of left/right eye-frame

	for image_list in list_of_image_lists:
		eye_frames_lr = []
		eye_frames_xy_in_orig_img = []
		for image in image_list:
			dets = detector(image, 1)
			if len(dets) is not 0:
				d = dets[0]
				predictions = predictor(image, d)
				shape = [(int(s.x), int(s.y)) for s in predictions.parts()]
				original_eye_l_x, original_eye_l_y, left_eye_frame = ff.extract_eye(ff.LEFT_EYE_IDs, shape, image)
				original_eye_r_x, original_eye_r_y, right_eye_frame = ff.extract_eye(ff.RIGHT_EYE_IDs, shape, image)
				eye_frames_lr.append([left_eye_frame, right_eye_frame])
				eye_frames_xy_in_orig_img.append(
					[(original_eye_l_x, original_eye_l_y), (original_eye_r_x, original_eye_r_y)])

		# print("eye_frames_lr:", eye_frames_lr)
		list_of_eye_frame_lists.append(eye_frames_lr)
		list_of_eye_frames_xy_in_orig_img.append(eye_frames_xy_in_orig_img)
	# print("xy orig:", list_of_eye_frames_xy_in_orig_img)
	# print("list_of_eye_frame_lists:", list_of_eye_frame_lists)

	# create a list with lists of the xy-positions of left and right eye-frames, respectively
	list_of_eyecenter_xy_lists = []

	for eye_list in list_of_eye_frame_lists:
		# ecd.center_detection returns list of [(xl ,yl) , (xr, yr)]'s
		list_of_eyecenter_xy_lists.append(ecd.center_detection(eye_list, (thresholds[0], thresholds[1])))
	# print("list_of_eyecenter_xy_lists:", list_of_eyecenter_xy_lists)

	# find eye center coordinates in starting image
	list_of_eyecenter_xys_in_starting_img_lists = []

	for orig_list, centers_list in zip(list_of_eye_frames_xy_in_orig_img, list_of_eyecenter_xy_lists):

		eyecenter_xys_in_starting_img_list = []

		for ((orig_xl, orig_yl), (orig_xr, orig_yr)), ((eye_cxl, eye_cyl), (eye_cxr, eye_cyr)) in zip(orig_list,
																									  centers_list):
			eyecenter_xys_in_starting_img_list.append(
				[(orig_xl + eye_cxl, orig_yl + eye_cyl), (orig_xr + eye_cxr, orig_yr + eye_cyr)])

		list_of_eyecenter_xys_in_starting_img_lists.append(eyecenter_xys_in_starting_img_list)
	# print("list of eyecenter_xys:", list_of_eyecenter_xys_in_starting_img_lists)

	# calculate mean xys of left and right eye-centers

	list_of_xys_to_map_lists = []

	for xy_list in list_of_eyecenter_xys_in_starting_img_lists:
		sum_xr, sum_yr, sum_xl, sum_yl = 0, 0, 0, 0
		divisor = len(xy_list)

		for ((xc_l, yc_l), (xc_r, yc_r)) in xy_list:
			# print("xl,yl xr,yr:", xc_l, yc_l, xc_r, yc_r)
			sum_xl = sum_xl + xc_l
			sum_yl = sum_yl + yc_l
			sum_xr = sum_xr + xc_r
			sum_yr = sum_yr + yc_r
		list_of_xys_to_map_lists.append(
			[(round(sum_xl / divisor), round(sum_yl / divisor)), (round(sum_xr / divisor), round(sum_yr / divisor))])
	
	mapping_xys_left = [(x, y) for ((x, y), (_, _)) in list_of_xys_to_map_lists]
	mapping_xys_right = [(x, y) for ((_, _), (x, y)) in list_of_xys_to_map_lists]

	return mapping_xys_left, mapping_xys_right

# fit polynom between two lists of x coordinates and two lists of y coordinates, respectively
def x_y_mapping_polynoms(eye_xs, calibration_xs, eye_ys, calibration_ys):
	"""

	:param eye_xs: list of eye x-coordinates
	:param calibration_xs: list of calibration point x-coordinates of same length as eye_xs
	:param eye_ys: list of eye y-coordinates
	:param calibration_ys: list of calibration point y-coordinates of same length as eye_ys
	:return:  a polynom fitting the x-coordinates and a polynom fitting th y-coordinates

	"""
	x_coeffs = np.polyfit(eye_xs, calibration_xs, deg=2)
	y_coeffs = np.polyfit(eye_ys, calibration_ys, deg=2)

	eye_x_polynom = np.poly1d(x_coeffs)
	eye_y_polynom = np.poly1d(y_coeffs)

	#print("x polynom:", eye_x_polynom)
	#print("y polynom:", eye_y_polynom)

	return eye_x_polynom, eye_y_polynom


def get_polynoms(eye_center_xys, calibration_xys):

	xs_for_polynom = [x for (x, y) in eye_center_xys]
	ys_for_polynom = [y for (x, y) in eye_center_xys]

	#xs_for_polynom = [xs_for_polynom[2], xs_for_polynom[5], xs_for_polynom[8],
	#				  xs_for_polynom[1], xs_for_polynom[4], xs_for_polynom[7],
	#				  xs_for_polynom[0], xs_for_polynom[3], xs_for_polynom[6]]

	print("to map list:", eye_center_xys)
	print("xs, ys ", xs_for_polynom, ys_for_polynom)
	calib_xs, calib_ys = zip(*calibration_xys)
	#x_coeff_vec, y_coeff_vec = least_squares_9x5(eye_center_xys, calibration_xys)

	#polynom_creator = lambda a, b, c,  e, f: lambda x, y: a[0] + b[0]*x + c[0]*y + e[0]*x*x + f[0]*y*y

	#x_polynom = polynom_creator(*x_coeff_vec)
	#y_polynom = polynom_creator(*y_coeff_vec)

	x_polynom, y_polynom  = x_y_mapping_polynoms(xs_for_polynom, calib_xs, ys_for_polynom, calib_ys)


	return x_polynom, y_polynom


def least_squares_9x5(eyecenter_xys, screen_xys):

	# create 9x5 matrix

	A = []
	screen_xs, screen_ys = zip(*screen_xys)

	sx_vec = np.transpose(np.array([screen_xs]))  # from 1x9 to 9x1
	sy_vec = np.transpose(np.array([screen_xs]))  # from 1x9 to 9x1
	
	matrix_row = lambda eye_x, eye_y: [1, eye_x, eye_y, eye_x**2, eye_y**2]
	
	for (eye_x, eye_y)in eyecenter_xys:
		A.append(matrix_row(eye_x, eye_y))

	A = np.array(A)
	AT = np.transpose(A)
	ATA = AT.dot(A)
	ATsx_vec = AT.dot(sx_vec)
	ATsy_vec = AT.dot(sy_vec)

	x_coeff_vec = np.linalg.solve(ATA, ATsx_vec) # ATAx = ATb => x = (ATA)^-1ATb
	y_coeff_vec = np.linalg.solve(ATA, ATsy_vec)

	print("A", A)
	print("AT", AT)
	print("ATA", ATA)
	print("sx_vec", sx_vec)
	print("sy_vec", sy_vec)
	print("ATsx_vec", ATsx_vec)
	print("ATsy_vec", ATsy_vec)
	print("x_coeff_vec", x_coeff_vec)
	print("y_coeff_vec", y_coeff_vec)

	return x_coeff_vec, y_coeff_vec
