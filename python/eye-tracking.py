import cv2
import time
import dlib
import numpy as np
import matplotlib.pyplot as plt
from math import hypot

class eye:
	def __init__ (self, inner, top_inner, top_outer, outer, bottom_outer, bottom_inner, pos):
		self.inner_point = inner
		self.outer_point = outer

		self.top_inner_point = top_inner
		self.top_outer_point = top_outer

		self.bottom_inner_point = bottom_inner
		self.bottom_outer_point = bottom_outer

		self.points = np.array([(inner.x, inner.y),
						(top_inner.x, top_inner.y), 
						(top_outer.x, top_outer.y), 
						(outer.x, outer.y), 
						(bottom_outer.x, bottom_outer.y),
						(bottom_inner.x, bottom_inner.y)], np.int32)

		self.pos = pos

		self.top_point = midpoint(self.top_inner_point, self.top_outer_point)
		self.bottom_point = midpoint(self.bottom_inner_point, self.bottom_outer_point)
		self.left_point = inner if inner.x < outer.x else outer
		self.right_point = inner if inner.x > outer.x else outer

	def draw_dots (self, img, color = (255, 0, 255), radius = 3):
		for point in self.points:
			cv2.circle(img, (point.x, point.y), radius, color, cv2.FILLED)

	def draw_polylines (self, img, color = (0, 0, 255), thikness = 2):
		cv2.polylines(img, [self.points], True, color, thikness)

	def draw_polygon(self, img, color = (0, 0, 255), thikness = 2):
		cv2.polylines(img, [self.points], True, color, thikness)
		cv2.fillPoly(img, [self.points], color)

	def horizontal_line(self, img, color = (0, 255, 0), thikness = 1):
		self._horizontal_line = cv2.line(img, (self.inner_point.x, self.inner_point.y), 
											(self.outer_point.x, self.outer_point.y), color, thikness)

	def vertical_line(self, img, color = (0, 255, 0), thikness = 1):
		self._vertical_line = cv2.line(img, self.top_point, self.bottom_point, color, thikness)

	def img (self, img):
		return img[np.min(self.points[:, 1]):(np.max(self.points[:, 1]) + 1),
					np.min(self.points[:, 0]):(np.max(self.points[:, 0]) + 1)]

	def is_blink(self):
		horizontal_length = hypot(self.inner_point.x - self.outer_point.x, self.inner_point.y - self.outer_point.y)
		vertical_length = hypot(self.top_point[0] - self.bottom_point[0], self.top_point[1] - self.bottom_point[1])
		ratio = horizontal_length / vertical_length
		if ratio > 5.5:    # MUST BE CALIBRATED FOR EACH CASE and BE PROPORTIONAL TO HEAD ORIENTATION
			return True
		else:
			return False

	def threshold (self, img, threshold):
		_, threshold = cv2.threshold(self.img(img), threshold, 255, cv2.THRESH_BINARY) # MUST BE CALIBRATED
		return threshold

	def get_x (self, eye_thr):
		m = (self.right_point.y - self.left_point.y) / (self.right_point.x - self.left_point.x + 1)
		y0 = self.left_point.y - np.min(self.points[:, 1])
		x_mean = 0
		high_pics_num = 0
		for x in range(0, self.right_point.x - self.left_point.x + 1):
			y = int(m * x) + y0
			x_mean += x if eye_thr[y, x] == 0 else 0
			high_pics_num += int(eye_thr[y, x] == 0)
			eye_thr[y, x] = 127
		if high_pics_num == 0:
			return 0
		x_mean = int(x_mean / high_pics_num)
		y_mean = int(m * x_mean) + y0
		eye_thr[y_mean, x_mean] = 255
		x = hypot(x_mean, y_mean - y0) / hypot(self.inner_point.x - self.outer_point.x, self.inner_point.y - self.outer_point.y)

		return x

	def get_y (self):
		horizontal_length = hypot(self.inner_point.x - self.outer_point.x, self.inner_point.y - self.outer_point.y)
		vertical_length = hypot(self.top_point[0] - self.bottom_point[0], self.top_point[1] - self.bottom_point[1])
		ratio = vertical_length / horizontal_length
		return ratio


class GazeTracker:
	def __init__(self):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		self.reduction_faktor = 0.5

		self.board_width = 1200
		self.board_height = 600
		self.board = np.zeros((self.board_height, self.board_width, 3), np.uint8)

		self.board_eyes_width = 200

	def load_img (self, img):
		self.img = img
		self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		self.board = np.zeros((self.board_height, self.board_width, 3), np.uint8)

	def find_face (self):
		img_reduced = cv2.resize(self.img, (int(self.reduction_faktor * self.img.shape[1]), 
											int(self.reduction_faktor * self.img.shape[0])))
		gray_img_reduced = cv2.cvtColor(img_reduced, cv2.COLOR_BGR2GRAY)
		faces = self.detector(gray_img_reduced)

		if faces:
			face = faces[0]
			for tmp_face in faces:
				if tmp_face.area() > face.area():
					face = tmp_face
			self.face = dlib.rectangle(int(face.left() / self.reduction_faktor),
										int(face.top() / self.reduction_faktor),
										int(face.right() / self.reduction_faktor),
										int(face.bottom() / self.reduction_faktor))
			return True
		return False

	def find_eyes (self):
		landmarks = self.predictor(self.gray_img, self.face)
		self.landmarks = landmarks

		self.right_eye = eye(landmarks.part(39), landmarks.part(38), landmarks.part(37), landmarks.part(36), landmarks.part(41), 
						landmarks.part(40), 'right');
		self.left_eye = eye(landmarks.part(42), landmarks.part(43), landmarks.part(44), landmarks.part(45), landmarks.part(46), 
						landmarks.part(47), 'left');

	def get_gaze_threshold (self):
		mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
		self.left_eye.draw_polygon(mask, color=255)
		self.right_eye.draw_polygon(mask, color=255)
		self.eyes_masked_img = cv2.bitwise_and(self.gray_img, self.gray_img, mask=mask)
		self.eyes_masked_img += cv2.bitwise_not(mask)

		left_eye_thr = self.left_eye.threshold(self.eyes_masked_img, cv2.getTrackbarPos("Threshold", "Board"))
		right_eye_thr = self.right_eye.threshold(self.eyes_masked_img, cv2.getTrackbarPos("Threshold", "Board"))

		x = (self.left_eye.get_x(left_eye_thr) )#+ self.right_eye.get_x(right_eye_thr)) / 2
		y = self.left_eye.get_y()

		left_eye_thr = cv2.resize(left_eye_thr, (int(self.board_eyes_width),
												 int(left_eye_thr.shape[0] * self.board_eyes_width / left_eye_thr.shape[1])), cv2.INTER_NEAREST)
		right_eye_thr = cv2.resize(right_eye_thr, (int(self.board_eyes_width),
												 int(right_eye_thr.shape[0] * self.board_eyes_width / right_eye_thr.shape[1])))

		self.board[0:left_eye_thr.shape[0], 
					(self.board_width - left_eye_thr.shape[1]):self.board_width] = cv2.cvtColor(left_eye_thr, cv2.COLOR_GRAY2BGR)
		self.board[left_eye_thr.shape[0]:left_eye_thr.shape[0] + right_eye_thr.shape[0], 
					(self.board_width - right_eye_thr.shape[1]):self.board_width] = cv2.cvtColor(right_eye_thr, cv2.COLOR_GRAY2BGR)

		# cv2.imshow("left_eye", left_eye_thr)
		# cv2.imshow("right_eye", right_eye_thr)

		cv2.putText(self.board, 'x = ' + str(x),
				 (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.putText(self.board, 'y = ' + str(y),
				 (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		# x = (x - 0.55) * (-7000) + 600
		# y = (y - 0.25) * (-5000) + 300

		return x, y

	def draw_lines (self):
		self.right_eye.horizontal_line(img)
		self.left_eye.horizontal_line(img)
		self.right_eye.vertical_line(img)
		self.left_eye.vertical_line(img)

	def is_blink (self):
		if self.left_eye.is_blink() and self.right_eye.is_blink():
			return True
		return False

	def draw_face_rect(self):
		cv2.rectangle(self.img, (self.face.left(), self.face.top()), (self.face.right(), self.face.bottom()), (0, 255, 0), 2)

	def draw_polylines (self):
		self.left_eye.draw_polylines(self.img)
		self.right_eye.draw_polylines(self.img)

	def draw_landmarks (self):
		for i in range(54):
			cv2.circle(self.img, (self.landmarks.part(i).x, self.landmarks.part(i).y), 3, (255, 0, 255), cv2.FILLED)


def midpoint(p1, p2):
	return ((p1.x + p2.x) // 2, (p1.y + p2.y) // 2)

def nothing (val):
	pass

cap = cv2.VideoCapture(0)
duration = time.time()

gaze_tracker = GazeTracker()

cv2.namedWindow("Board")
cv2.createTrackbar("Threshold", "Board", 0, 255, nothing)
cv2.setTrackbarPos("Threshold", "Board", 70)

gaze_arr_size = 10
gaze_arr = np.zeros((gaze_arr_size, 2))
gaze_arr_index = 0

gaze = []
gaze_t = []
blink = []
blink_t = []


start = time.time()

while True:
	loop_start = time.time()
	_, img = cap.read()

	gaze_tracker.load_img(img)

	if(gaze_tracker.find_face()):
		gaze_tracker.find_eyes()

		
		

		cv2.circle(gaze_tracker.board, (int(np.mean(gaze_arr[:, 0])), int(np.mean(gaze_arr[:, 1]))), 10, (0, 255, 255), 2)

		if (gaze_tracker.is_blink()):
			cv2.putText(gaze_tracker.board, "Blinking", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
			blink += [gaze[-1]]
			blink_t += [time.time() - start]
		else:
			gaze_t += [time.time() - start]
			gaze_arr[gaze_arr_index, :] = np.array([gaze_tracker.get_gaze_threshold()])
			gaze_arr_index = (gaze_arr_index + 1) % gaze_arr_size
			gaze += [(np.mean(gaze_arr[:, 0]), np.mean(gaze_arr[:, 1]))]

		gaze_tracker.draw_face_rect()
		# gaze_tracker.draw_polylines()
		# gaze_tracker.draw_landmarks()
		# gaze_tracker.draw_lines()		

	cv2.putText(gaze_tracker.board, str((int) (duration * 1000)) + ' ms, ' + str((int) (1 / duration)) + ' FPS',
				 (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	# cv2.imshow("Mask", gaze_tracker.eyes_masked_img)
	cv2.imshow("Cam", img)
	cv2.imshow('Board', gaze_tracker.board)
	
	if cv2.waitKey(1) == 27:
		break
	loop_end = time.time()
	duration = loop_end - loop_start


cap.release
cv2.destroyAllWindows()

# plt.plot(gaze_t, gaze)
# plt.plot(blink_t, blink, 'o')
# plt.show()