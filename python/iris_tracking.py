import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import scipy.io
from math import hypot, inf, tan, atan, pi, sin, cos
from face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis
from mpl_toolkits.mplot3d import Axes3D


class Filter:
    def __init__(self, Num, Den):
        self.Num = Num
        self.Den = Den
        if len(Num) > len(Den):
            self.Den += [0] * (len(Num) - len(Den))
        elif len(Num) < len(Den):
            self.Num += [0] * (len(Den) - len(Num))
        self.order = len(Den)
        self.input_buffer = [0] * self.order
        self.output_buffer = [0] * self.order
        self.index = 0
        self.first_input = True

    def add_input(self, x):
        if self.first_input:
            self.input_buffer = [x] * self.order
            self.output_buffer = [x] * self.order
            self.first_input = False
        else:
            self.index = (self.index + 1) % self.order
            self.input_buffer[self.index] = x
            y = 0
            for i in range(self.order):
                j = (self.index - i) % self.order
                y += self.Num[i] * self.input_buffer[j]
                if i != 0:
                    y -= self.Den[i] * self.output_buffer[j]
            self.output_buffer[self.index] = y

    def output(self):
        return self.output_buffer[self.index]


class GazeCalculator:
    def __init__(self):
        self.right_gaze_coeffs = [11614.728433206808, -5930.995837491617, -3334.891897265645, 690.8138330408594]
        self.left_gaze_coeffs = [-15213.10861867692, 9284.722867724182, -11821.4602721843, 853.0727963753402]

        self.L_BLINK_RATIO = 0.1
        self.R_BLINK_RATIO = 0.15

        self.position_correct = [0, 0]
        self.position_coeffs = [1, 1, 1]

    def position2gaze(self, lix, liy, rix, riy, show=False, _board=None):
        # left_gaze_x = int(-14000 * tan(asin(left_x_filter.output() - 0.56)) + BOARD_WIDTH / 2)
        # right_gaze_x = int(14000 * tan(asin(right_x_filter.output() - 0.56)) + BOARD_WIDTH / 2)
        # left_gaze_y = int(-14000 * tan(asin(left_y_filter.output() - 0.31)) + BOARD_HEIGHT / 2)
        # right_gaze_y = int(-14000 * tan(asin(right_y_filter.output() - 0.31)) + BOARD_HEIGHT / 2)

        x_l = self.left_gaze_coeffs[0] * lix + self.left_gaze_coeffs[1]
        x_r = self.right_gaze_coeffs[0] * rix + self.right_gaze_coeffs[1]
        y_l = self.left_gaze_coeffs[2] * liy + self.left_gaze_coeffs[3]
        y_r = self.right_gaze_coeffs[2] * riy + self.right_gaze_coeffs[3]

        if (show):
            cv2.circle(_board, (int(x_l), int(y_l)), 10, (0, 255, 0), 2)
            cv2.circle(_board, (int((x_l + x_r)/2), int((y_l + y_r)/2)), 10, (255, 255, 255), 2)
            cv2.circle(_board, (int(x_r), int(y_r)), 10, (0, 0, 255), 2)

        return int((x_l + x_r)/2), int((y_l + y_r)/2)

    def head_position_correct(self, p, o):
        return p[0] - self.position_correct[0] * tan(pi * o[2] / 180) / p[2] , \
               p[1] - self.position_correct[1] * tan(pi * o[1] / 180) / p[2], p[2]

    def head_position(self, p, o):
        p = self.head_position_correct(p, o)
        return p[0] * self.position_coeffs[0],\
               p[1] * self.position_coeffs[1],\
               p[2] * self.position_coeffs[2]

    def head_orientation(self, p, o):
        return o[0] / pi, o[1] + 180 * atan(p[1]/p[2]) / pi, o[2] - 180 * atan(p[0]/p[2]) / pi

class HeadCoordinates:
    def __init__(self, _frame_width, _frame_height):
        # pseudo camera internals
        self.frame_width = _frame_width
        self.frame_height = _frame_height
        focal_length = _frame_width
        center = (_frame_width / 2, _frame_height / 2)
        self.camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double")
        self.pcf = PCF(near=1, far=10000, frame_height=_frame_height, frame_width=_frame_width, fy=self.camera_matrix[1, 1])
        self.dist_coeff = np.zeros((4, 1))

    def get_head_orientation(self, rvec, tvec):
        rmat = cv2.Rodrigues(rvec)[0]
        P = np.hstack((rmat, tvec))  # projection matrix

        # find euler angles
        euler_angles = cv2.decomposeProjectionMatrix(P)[6]
        _pitch = -euler_angles.item(0)
        _yaw = -euler_angles.item(1)
        _roll = euler_angles.item(2)

        # Ajust coordinate ranges
        if _pitch < 0:
            _pitch = 180 + _pitch
        else:
            _pitch = _pitch - 180

        t = '{y},{p},{r}'.format(y=round(_yaw),
                                 p=round(_pitch),
                                 r=round(_roll))
        # return round(_roll), round(_pitch), round(_yaw)
        return _roll, _pitch, _yaw

    def head_coordinates(self, _mesh_points_3d):
        metric_landmarks, pose_transform_mat = get_metric_landmarks(_mesh_points_3d.copy(), self.pcf)
        model_points = metric_landmarks[0:3, points_idx].T
        image_points = _mesh_points_3d[0:2, points_idx].T * np.array([self.frame_width, self.frame_height])[None, :]

        success, rotation_vector, _translation_vector = cv2.solvePnP(model_points, image_points, self.camera_matrix,
                                                                     self.dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 25.0)]), rotation_vector,
                                                         _translation_vector, self.camera_matrix, self.dist_coeff)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        _roll, _pitch, _yaw = self.get_head_orientation(rotation_vector, _translation_vector)

        return _roll, _pitch, _yaw, _translation_vector


def length(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def slope(src, dest):
    dy = dest[1] - src[1]
    dx = dest[0] - src[0]
    if dx == 0:
        return inf if dy>0 else -inf
    return dy/dx

def project(line_start_point, point, line_end_point):
    vect1 = [point[0] - line_start_point[0], point[1] - line_start_point[1]]
    vect2 = [line_end_point[0] - line_start_point[0], line_end_point[1] - line_start_point[1]]
    dot_product = vect1[0] * vect2[0] + vect1[1] * vect2[1]
    vect2_length = length(line_start_point, line_end_point)
    projected_length = dot_product / vect2_length
    projected_vect = projected_length*vect2[0]/vect2_length, projected_length*vect2[1]/vect2_length

    return [int(line_start_point[0] + projected_vect[0]), int(line_start_point[1] + projected_vect[1])]

def signed_dist(line_start_point, point, line_end_point):
    vect1 = [point[0] - line_start_point[0], point[1] - line_start_point[1]]
    vect2 = [line_end_point[0] - line_start_point[0], line_end_point[1] - line_start_point[1]]
    cross_product_z = vect1[0] * vect2[1] - vect1[1] * vect2[0]
    vect2_length = length(line_start_point, line_end_point)
    return cross_product_z/vect2_length


def cv2cross(_img, center, radius, color, thikness, angle=0.0):
    cv2.line(_img, (int(center[0] + radius * cos(3*pi/4 + angle)), int(center[1] + radius * sin(3*pi/4 + angle))),
                    (int(center[0] + radius * cos(-pi/4 + angle)), int(center[1] + radius * sin(-pi/4 + angle))), color, thikness)
    cv2.line(_img, (int(center[0] + radius * cos(pi/4 + angle)), int(center[1] + radius * sin(pi/4 + angle))),
                    (int(center[0] + radius * cos(5*pi/4 + angle)), int(center[1] + radius * sin(5*pi/4 + angle))), color, thikness)


def eyes_ratio(_mesh_points):
    l_h = length(_mesh_points[LEFT_EYE[0]], _mesh_points[LEFT_EYE[8]])
    r_h = length(_mesh_points[RIGHT_EYE[0]], _mesh_points[RIGHT_EYE[8]])
    l_v = length(_mesh_points[LEFT_EYE[4]], _mesh_points[LEFT_EYE[12]])
    r_v = length(_mesh_points[RIGHT_EYE[4]], _mesh_points[RIGHT_EYE[12]])
    l_ratio = l_v/l_h
    r_ratio = r_v/r_h

    return l_ratio, r_ratio

def iris_positions(_mesh_points):
    # calculate x coordinates based on eye horizontal line
    # left_iris_center = _mesh_points[LEFT_CENTER]
    # right_iris_center = _mesh_points[RIGHT_CENTER]
    #
    # left_projected_center = project(_mesh_points[LEFT_EYE[0]], left_iris_center, _mesh_points[LEFT_EYE[8]])
    # right_projected_center = project(_mesh_points[RIGHT_EYE[0]], right_iris_center, _mesh_points[RIGHT_EYE[8]])
    #
    # l_h = length(_mesh_points[LEFT_EYE[0]], _mesh_points[LEFT_EYE[8]])
    # r_h = length(_mesh_points[RIGHT_EYE[0]], _mesh_points[RIGHT_EYE[8]])
    #
    # left_iris_x = length(left_projected_center, _mesh_points[LEFT_EYE[0]]) / l_h
    # right_iris_x = length(right_projected_center, _mesh_points[RIGHT_EYE[0]]) / r_h


    # calculate x coordinates based on head horizontal line
    left_iris_center = _mesh_points[LEFT_CENTER]
    right_iris_center = _mesh_points[RIGHT_CENTER]

    left_projected_center = project(_mesh_points[RIGHT_EYE[0]], left_iris_center, _mesh_points[LEFT_EYE[0]])
    right_projected_center = project(_mesh_points[LEFT_EYE[0]], right_iris_center, _mesh_points[RIGHT_EYE[0]])

    l_h = length(_mesh_points[LEFT_EYE[0]], _mesh_points[LEFT_EYE[8]])
    r_h = length(_mesh_points[RIGHT_EYE[0]], _mesh_points[RIGHT_EYE[8]])

    left_iris_x = length(left_projected_center, _mesh_points[LEFT_EYE[0]]) / l_h
    right_iris_x = length(right_projected_center, _mesh_points[RIGHT_EYE[0]]) / r_h


    # calculate y coordinates based on eye ratio
    # l_ratio, r_ratio = eyes_ratio(_mesh_points)
    # left_iris_y = l_ratio
    # right_iris_y = r_ratio

    # calculate y coordinates based on head horizontal line
    left_iris_y = signed_dist(_mesh_points[RIGHT_EYE[0]], _mesh_points[LEFT_EYE[0]], left_iris_center) / l_h
    right_iris_y = signed_dist(_mesh_points[RIGHT_EYE[0]], _mesh_points[LEFT_EYE[0]], right_iris_center) / r_h


    # show axis and coordinates based on eye horizontal line
    # cv2.line(img, _mesh_points[LEFT_EYE[0]], _mesh_points[LEFT_EYE[8]], (127, 127, 127), 1)
    # cv2.line(img, _mesh_points[RIGHT_EYE[0]], _mesh_points[RIGHT_EYE[8]], (127, 127, 127), 1)
    # cv2.line(img, _mesh_points[LEFT_EYE[0]], left_iris_center, (255, 0, 255), 1)
    # cv2.line(img, _mesh_points[RIGHT_EYE[0]], right_iris_center, (255, 0, 255), 1)
    # cv2.circle(img, _mesh_points[RIGHT_EYE[0]], 2, (255, 0, 0), cv2.FILLED)
    # cv2.circle(img, _mesh_points[LEFT_EYE[0]], 2, (255, 0, 0), cv2.FILLED)
    # cv2.circle(img, left_projected_center, 2, (0, 0, 255), cv2.FILLED)
    # cv2.circle(img, right_projected_center, 2, (0, 0, 255), cv2.FILLED)


    # show axis and coordinates based on head horizontal line
    # cv2.line(img, _mesh_points[LEFT_EYE[0]], _mesh_points[RIGHT_EYE[0]], (0, 255, 0), 1)
    # cv2.line(img, _mesh_points[LEFT_EYE[0]], left_projected_center, (0, 0, 255), 1)
    # cv2.line(img, left_iris_center, left_projected_center, (255, 0, 0), 1)
    # cv2.line(img, _mesh_points[RIGHT_EYE[0]], right_projected_center, (0, 0, 255), 1)
    # cv2.line(img, right_iris_center, right_projected_center, (255, 0, 0), 1)
    # cv2.circle(img, left_projected_center, 2, (255, 0, 255), cv2.FILLED)
    # cv2.circle(img, right_projected_center, 2, (255, 0, 255), cv2.FILLED)


    # show horizontal and vertical lines
    # cv2.line(img, _mesh_points[LEFT_EYE[0]], _mesh_points[LEFT_EYE[8]], (0, 255, 0), 1)
    # cv2.line(img, _mesh_points[LEFT_EYE[4]], _mesh_points[LEFT_EYE[12]], (0, 255, 0), 1)
    # cv2.line(img, _mesh_points[RIGHT_EYE[0]], _mesh_points[RIGHT_EYE[8]], (0, 255, 0), 1)
    # cv2.line(img, _mesh_points[RIGHT_EYE[4]], _mesh_points[RIGHT_EYE[12]], (0, 255, 0), 1)


    return left_iris_x, right_iris_x, left_iris_y, right_iris_y

def is_blinked(_mesh_points, _gc):
    l_ratio, r_ratio = eyes_ratio(_mesh_points)
    right_blink = r_ratio < _gc.R_BLINK_RATIO
    left_blink = l_ratio < _gc.L_BLINK_RATIO
    return left_blink or right_blink


def get_mean_position(points_num, _hc, show_board = False, board_background = None):
    data = []
    t = []
    _FILTER_NUM = [0.1] * 10
    _FILTER_DEN = [1]
    lix_filter = Filter(_FILTER_NUM, _FILTER_DEN)
    liy_filter = Filter(_FILTER_NUM, _FILTER_DEN)
    rix_filter = Filter(_FILTER_NUM, _FILTER_DEN)
    riy_filter = Filter(_FILTER_NUM, _FILTER_DEN)

    while True:
        ret, img = cap.read()
        img_h, img_w = img.shape[:2]
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        if show_board:
            _board = board_background.copy()

        if results.multi_face_landmarks:
            mesh_points = np.array([[int(p.x * img_w), int(p.y * img_h)] for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3d = np.array([(p.x, p.y, p.z) for p in results.multi_face_landmarks[0].landmark])[:468].T

            left_iris_x, right_iris_x, left_iris_y, right_iris_y = iris_positions(mesh_points)
            _roll, _pitch, _yaw, _tvect = _hc.head_coordinates(mesh_points_3d)
            lix_filter.add_input(left_iris_x)
            liy_filter.add_input(left_iris_y)
            rix_filter.add_input(right_iris_x)
            riy_filter.add_input(right_iris_y)
            l_ratio, r_ratio = eyes_ratio(mesh_points)
            _tvect = list(np.reshape(_tvect, -1))
            data += [[lix_filter.output(), rix_filter.output(), liy_filter.output(), riy_filter.output(),
                      l_ratio, r_ratio, _roll, _pitch, _yaw] + _tvect]
            t += [_tvect[0]]
            if show_board:
                cv2cross(_board, (10*_tvect[0] + BOARD_WIDTH // 2, 10*_tvect[1] + BOARD_HEIGHT // 2), 10, (255, 255, 255), 2)
                cv2cross(_board, (int(10*_tvect[0] + BOARD_WIDTH / 2 + 10*_tvect[2]*tan(pi*_yaw/180)),
                                  int(10*_tvect[1] + BOARD_HEIGHT / 2 - 10*_tvect[2]*tan(pi*_pitch/180))), 10, (255, 0, 255), 2)
                cv2.imshow("Board", _board)

        if cv2.waitKey(1) == 32:
            data = np.array(data)
            return t, np.mean(data[-points_num:, :], 0)


def callibration(_gc, _hc):
    # position correct
    t = []
    t_stops = []
    POINTS_NUM = 20
    GRID = [3, 4]
    result = []
    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2.imshow("Board", cal_board)
    cv2.waitKey(0)

    cv2cross(cal_board, (BOARD_WIDTH // 2, BOARD_HEIGHT // 2), 10, (0, 255, 255), 2)
    cv2.imshow("Board", cal_board)
    returned_t, returned_result = get_mean_position(POINTS_NUM, _hc, True, cal_board)
    t += returned_t
    t_stops += [len(t) - 1]
    result += [returned_result]

    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2cross(cal_board, (BOARD_WIDTH // 2, BOARD_HEIGHT // 2), 10, (0, 255, 255), 2)
    cv2cross(cal_board, (int(BOARD_WIDTH / 2 + 10 * result[0][11] * tan(pi * 30 / 180)),
                      int(BOARD_HEIGHT / 2)), 10, (0, 255, 255), 2)
    cv2.imshow("Board", cal_board)
    returned_t, returned_result = get_mean_position(POINTS_NUM, _hc, True, cal_board)
    t += returned_t
    t_stops += [len(t) - 1]
    result += [returned_result]

    _gc.position_correct[0] = ((result[0][11] + result[1][11]) * result[1][9] /2) / tan(pi*30/180)

    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2cross(cal_board, (BOARD_WIDTH // 2, BOARD_HEIGHT // 2), 10, (0, 255, 255), 2)
    cv2cross(cal_board, (int(BOARD_WIDTH / 2),
                         int(BOARD_HEIGHT / 2 + 10 * result[0][11] * tan(pi * 30 / 180))), 10, (0, 255, 255), 2)
    cv2.imshow("Board", cal_board)
    returned_t, returned_result = get_mean_position(POINTS_NUM, _hc, True, cal_board)
    t += returned_t
    t_stops += [len(t) - 1]
    result += [returned_result]

    _gc.position_correct[1] = -((result[0][11] + result[1][11]) * result[2][10] / 2) / tan(pi * 30 / 180)

    print('Calibration results:')
    print('position_correct = ', end='')
    print(_gc.position_correct)

    # gaze
    result = []
    t = []
    t_stops = []

    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2.circle(cal_board, (BOARD_WIDTH // 2, BOARD_HEIGHT // 2), 15, (255, 0, 255), cv2.FILLED)
    cv2.imshow("Board", cal_board)
    returned_t, returned_result = get_mean_position(POINTS_NUM, _hc)
    t += returned_t
    t_stops += [len(t) - 1]
    result += [returned_result]

    for j in range(GRID[0]):
        for i in range(GRID[1]):
            cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
            cv2.circle(cal_board, (int(i * BOARD_WIDTH / (GRID[1] - 1)), int(j * BOARD_HEIGHT / (GRID[0] - 1))), 15, (255, 0, 255), cv2.FILLED)
            cv2.imshow("Board", cal_board)
            returned_t, returned_result = get_mean_position(POINTS_NUM, _hc)
            t += returned_t
            t_stops += [len(t) - 1]
            result += [returned_result]

    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2.putText(cal_board, 'Close Your Eyes and press space', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.imshow("Board", cal_board)
    returned_t, returned_result = get_mean_position(POINTS_NUM, _hc)
    result += [returned_result]

    POINTS = GRID[0]*GRID[1] + 1
    _gc.L_BLINK_RATIO = (3*result[POINTS][4] + min(np.array(result)[:POINTS, 4]))/4
    _gc.R_BLINK_RATIO = (3*result[POINTS][5] + min(np.array(result)[:POINTS, 5]))/4

    Y_x = np.array([BOARD_WIDTH/2] + list(np.array(list(range(GRID[1])) * GRID[0]) * BOARD_WIDTH / (GRID[1] - 1)))
    Y_y = np.array([BOARD_HEIGHT/2] + list(np.reshape(np.reshape(np.array(list(range(GRID[0])) * GRID[1]), (GRID[1], GRID[0])).T, -1) * BOARD_HEIGHT / (GRID[0] - 1)))

    X = np.concatenate((np.array([[1] * POINTS]).T, np.array([np.array(result)[:POINTS, 0]]).T), axis=1)
    coeffs_lx = np.matmul(np.linalg.pinv(X), Y_x)
    X = np.concatenate((np.array([[1] * POINTS]).T, np.array([np.array(result)[:POINTS, 1]]).T), axis=1)
    coeffs_rx = np.matmul(np.linalg.pinv(X), Y_x)
    X = np.concatenate((np.array([[1] * POINTS]).T, np.array([np.array(result)[:POINTS, 2]]).T), axis=1)
    coeffs_ly = np.matmul(np.linalg.pinv(X), Y_y)
    X = np.concatenate((np.array([[1] * POINTS]).T, np.array([np.array(result)[:POINTS, 3]]).T), axis=1)
    coeffs_ry = np.matmul(np.linalg.pinv(X), Y_y)

    _gc.left_gaze_coeffs = [coeffs_lx[1], coeffs_lx[0], coeffs_ly[1], coeffs_ly[0]]
    _gc.right_gaze_coeffs = [coeffs_rx[1], coeffs_rx[0], coeffs_ry[1], coeffs_ry[0]]

    print('left eye coeffs = ', end='')
    print(_gc.left_gaze_coeffs)
    print('right eye coeffs = ', end='')
    print(_gc.right_gaze_coeffs)

    # plt.figure()
    # plt.title('X')
    # plt.plot(t, 'b')
    # plt.plot(t_stops, np.array(t)[t_stops], 'kx')
    # for k in range(POINTS):
    #     plt.plot(range((t_stops[k] - POINTS_NUM + 1), t_stops[k] + 1), [result[k][9]] * POINTS_NUM, 'r')
    # plt.show()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('left iris x callibration')
    for k in range(GRID[0]):
        plt.plot(np.array(range(GRID[1])) * BOARD_WIDTH / (GRID[1] - 1), np.array(result)[(1 + k * GRID[1]):(1 + (k+1)*GRID[1]), 0])

    plt.subplot(2, 2, 2)
    plt.title('right iris x callibration')
    for k in range(GRID[0]):
        plt.plot(np.array(range(GRID[1])) * BOARD_WIDTH / (GRID[1] - 1), np.array(result)[(1 + k * GRID[1]):(1 + (k + 1) * GRID[1]), 1])

    plt.subplot(2, 2, 3)
    plt.title('left iris y callibration')
    for k in range(GRID[1]):
        plt.plot(np.array(range(GRID[0])) * BOARD_HEIGHT/(GRID[0] - 1), np.array(result)[list(range((1+k), POINTS, GRID[1])), 2])

    plt.subplot(2, 2, 4)
    plt.title('right iris y callibration')
    for k in range(GRID[1]):
        plt.plot(np.array(range(GRID[0])) * BOARD_HEIGHT / (GRID[0] - 1), np.array(result)[list(range((1 + k), POINTS, GRID[1])), 3])

    plt.figure()
    for p in result[:POINTS]:
        plt.plot(p[0], p[2], 'or')
        plt.plot(p[1], p[3], 'ob')
    plt.plot(result[0][0], result[0][2], '^c')
    plt.plot(result[0][1], result[0][3], '^c')
    plt.show()

    return result

def get_mean_gaze(points_num, _hc, _gc):
    data = []
    t = []
    _FILTER_NUM = [0.1] * 10
    _FILTER_DEN = [1]
    lix_filter = Filter(_FILTER_NUM, _FILTER_DEN)
    liy_filter = Filter(_FILTER_NUM, _FILTER_DEN)
    rix_filter = Filter(_FILTER_NUM, _FILTER_DEN)
    riy_filter = Filter(_FILTER_NUM, _FILTER_DEN)

    while True:
        ret, img = cap.read()
        img_h, img_w = img.shape[:2]
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            mesh_points = np.array([[int(p.x * img_w), int(p.y * img_h)] for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3d = np.array([(p.x, p.y, p.z) for p in results.multi_face_landmarks[0].landmark])[:468].T

            left_iris_x, right_iris_x, left_iris_y, right_iris_y = iris_positions(mesh_points)
            _roll, _pitch, _yaw, _tvect = _hc.head_coordinates(mesh_points_3d)
            lix_filter.add_input(left_iris_x)
            liy_filter.add_input(left_iris_y)
            rix_filter.add_input(right_iris_x)
            riy_filter.add_input(right_iris_y)
            _tvect = list(np.reshape(_tvect, -1))
            data += [_gc.position2gaze(lix_filter.output(), liy_filter.output(), rix_filter.output(), riy_filter.output())
]
            t += [_tvect[0]]

        if cv2.waitKey(1) == 32:
            data = np.array(data)
            return t, np.mean(data[-points_num:, :], 0)

def evaluation(_gc, _hc):
    t = []
    t_stops = []
    POINTS_NUM = 20
    GRID = [3, 4]
    result = []
    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2.imshow("Board", cal_board)
    cv2.waitKey(0)

    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2.circle(cal_board, (BOARD_WIDTH // 2, BOARD_HEIGHT // 2), 15, (255, 0, 255), cv2.FILLED)
    cv2.imshow("Board", cal_board)
    returned_t, returned_result = get_mean_gaze(POINTS_NUM, _hc, _gc)
    t += returned_t
    t_stops += [len(t) - 1]
    result += [returned_result]

    for j in range(GRID[0]):
        for i in range(GRID[1]):
            cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
            cv2.circle(cal_board, (int(i * BOARD_WIDTH / (GRID[1] - 1)), int(j * BOARD_HEIGHT / (GRID[0] - 1))), 15, (255, 0, 255), cv2.FILLED)
            cv2.imshow("Board", cal_board)
            returned_t, returned_result = get_mean_gaze(POINTS_NUM, _hc, _gc)
            t += returned_t
            t_stops += [len(t) - 1]
            result += [returned_result]

    POINTS = GRID[0]*GRID[1] + 1

    points = [[BOARD_WIDTH // 2, BOARD_HEIGHT // 2]]
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            points += [[j * BOARD_WIDTH // (GRID[1] - 1), i * BOARD_HEIGHT // (GRID[0] - 1)]]
    for p in range(POINTS):
        plt.plot([points[p][0], result[p][0]], [points[p][1], result[p][1]], 'k')
        plt.plot(points[p][0], points[p][1], 'ob')
        plt.plot(result[p][0], result[p][1], 'or')
    plt.xlabel('x')
    plt.ylabel('y')

    _fig = plt.figure()
    _ax = _fig.gca(projection='3d')
    X = np.reshape(np.array(points)[1:, 0], (GRID[0], GRID[1]))
    _X = np.reshape(np.array(result)[1:, 0], (GRID[0], GRID[1]))
    Y = np.reshape(np.array(points)[1:, 1], (GRID[0], GRID[1]))
    _Y = np.reshape(np.array(result)[1:, 1], (GRID[0], GRID[1]))
    Z = np.sqrt((X - _X)**2 + (Y - _Y)**2)
    _ax.plot_surface(X, Y, Z)
    plt.xlabel('x')
    plt.ylabel('y')
    _ax.set_zlabel('error (px)')

    plt.show()

cap = cv2.VideoCapture(0) #'eye_movement_test.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('resolution: ', end='')
print((frame_width, frame_height))
hc = HeadCoordinates(frame_width, frame_height)

mp_face_mesh = mp.solutions.face_mesh

px_per_cm = 44.44

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173]

RIGHT_IRIS = range(474, 478)
LEFT_IRIS = range(469, 473)

RIGHT_CENTER = 473
LEFT_CENTER = 468

# FILTER_NUM = [1]
# FILTER_DEN = [1]

# FILTER_NUM = [0.0011422, 0.0051294, 0.008849, 0.011549, 0.012542, 0.011337, 0.0077619, 0.0020454, -0.0051582, -0.012812, -0.019604, -0.024115, -0.02502, -0.021293, -0.012392, 0.0016148, 0.019999, 0.041412, 0.064017, 0.085696, 0.1043, 0.11791, 0.1251, 0.1251, 0.11791, 0.1043, 0.085696, 0.064017, 0.041412, 0.019999, 0.0016148, -0.012392, -0.021293, -0.02502, -0.024115, -0.019604, -0.012812, -0.0051582, 0.0020454, 0.0077619, 0.011337, 0.012542, 0.011549, 0.008849, 0.0051294, 0.0011422]
# FILTER_DEN = [1] + [0] * (len(FILTER_NUM) - 1)

# FILTER_NUM = [0.1] * 10
# FILTER_DEN = [1] + [0] * 9

# FILTER_NUM = [0.0084, 0.0252, 0.0252, 0.0084]
# FILTER_DEN = [1.0000, -2.0727, 1.5292, -0.3892]

FILTER_NUM = [0.00084, 0.00336, 0.00588, 0.00672, 0.00672, 0.00672, 0.00672, 0.00672, 0.00672, 0.00672, 0.00588, 0.00336, 0.00084]
FILTER_DEN = [1.0000, -2.0727, 1.5292, -0.3892]

right_x_filter = Filter(FILTER_NUM, FILTER_DEN)
right_y_filter = Filter(FILTER_NUM, FILTER_DEN)
left_x_filter = Filter(FILTER_NUM, FILTER_DEN)
left_y_filter = Filter(FILTER_NUM, FILTER_DEN)

gc = GazeCalculator()

BOARD_WIDTH = 1200
BOARD_HEIGHT = 600

points_idx = [33, 263, 61, 291, 199]  # [k for k in range(0,468)]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

fps = []
right_gaze_arr = []  # (t, x, y)
left_gaze_arr = []
right_blink_arr = []  # (t, x0, y0)
left_blink_arr = []
paint_points = []
test_arr = []

IDLE = 0
CALLIBRATION = 1
EVALUATION = 2
PAINT = 3
MODE = IDLE

PLOT3D = False

if PLOT3D:
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


# for k in range(100):
#     ax.plot([0, sin(k/10)], [0, cos(k/10)], [0, 1])
#     plt.xlim([-1, 1])
#     plt.ylim([-1, 1])
#     plt.draw()
#     plt.pause(0.02)
#     ax.cla()

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh: 
    
    duration = time.time()
    process_duration = time.time()
    start = time.time()

    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # line1, = ax.plot([0], [0], 'bo')
    # plt.ylim([0, 1])

    # fig = plt.figure('test')
    # ax = fig.add_subplot(111)
    # plt.ion()
    # my_plot, = ax.plot([0, 1], [0, 1], 'ro')

    while True:
        loop_start = time.time()
        # img = cv2.imread('pic1.jpg')
        # img = cv2.resize(img, (0, 0), fx = 0.4, fy=0.4)
        ret, img = cap.read()
        if not ret:
            break
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        process_start = time.time()
        img_h, img_w = img.shape[:2]
        # img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        top = np.zeros((480, 640, 3), np.uint8)
        board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)

        if results.multi_face_landmarks:
            mesh_points = np.array([[int(p.x * img_w), int(p.y * img_h)] for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3d = np.array([(p.x, p.y, p.z) for p in results.multi_face_landmarks[0].landmark])[:468].T

            if MODE == IDLE:
                # blink detection

                l_ratio, r_ratio = eyes_ratio(mesh_points)
                right_blink = r_ratio < gc.R_BLINK_RATIO
                left_blink = l_ratio < gc.L_BLINK_RATIO
                
                # calculate coordinates 
                left_iris_x, right_iris_x, left_iris_y, right_iris_y = iris_positions(mesh_points)

                # calculate head orientations
                roll, pitch, yaw, translation_vector = hc.head_coordinates(mesh_points_3d)
                head_x_raw = int(translation_vector[0] * px_per_cm + BOARD_WIDTH / 2)
                head_gaze_x_raw = int(translation_vector[0] * px_per_cm + px_per_cm * translation_vector[2] * tan(yaw * pi / 180) + BOARD_WIDTH / 2)
                head_y_raw = int(translation_vector[1] * px_per_cm + BOARD_HEIGHT / 2)
                head_gaze_y_raw = int(translation_vector[1] * px_per_cm + px_per_cm * translation_vector[2] * tan(-pitch * pi / 180) + BOARD_HEIGHT / 2)

                corrected_tvect = gc.head_position(translation_vector, [roll, pitch, yaw])
                head_x = corrected_tvect[0] * px_per_cm + BOARD_WIDTH/2
                head_y = corrected_tvect[1] * px_per_cm + BOARD_HEIGHT/2
                head_gaze_x = int(head_x + px_per_cm * translation_vector[2] * tan(gc.head_orientation(corrected_tvect, [roll, pitch, yaw])[2] * pi / 180))
                head_gaze_y = int(head_y + px_per_cm*translation_vector[2]*tan(-gc.head_orientation(corrected_tvect, [roll, pitch, yaw])[1] * pi / 180))
                # print((roll, pitch, yaw))
                # print(translation_vector)

                # calculate gaze
                if left_blink:
                    left_blink_arr += [[time.time() - start, left_x_filter.output(), left_y_filter.output()]]
                if right_blink:
                    right_blink_arr += [[time.time() - start, right_x_filter.output(), right_y_filter.output()]]

                if not is_blinked(mesh_points, gc):
                    # filter
                    left_x_filter.add_input(left_iris_x)
                    left_y_filter.add_input(left_iris_y)
                    right_x_filter.add_input(right_iris_x)
                    right_y_filter.add_input(right_iris_y)

                    left_gaze_arr += [[time.time() - start, left_x_filter.output(), left_y_filter.output()]]
                    right_gaze_arr += [[time.time() - start, right_x_filter.output(), right_y_filter.output()]]

                gaze_x, gaze_y = gc.position2gaze(left_x_filter.output(), left_y_filter.output(), right_x_filter.output(), right_y_filter.output(), True, board)

                # data to show
                test_arr += [[time.time() - start, length(mesh_points[LEFT_EYE[0]], mesh_points[LEFT_EYE[8]]),
                                                    length(mesh_points[RIGHT_EYE[0]], mesh_points[RIGHT_EYE[8]]),
                                                    length(mesh_points[RIGHT_EYE[0]], mesh_points[LEFT_EYE[0]]),
                                                    translation_vector[2]]]

                # show orientation calculations
                # for ii in points_idx: # range(landmarks.shape[1]):
                #     pos = np.array((frame_width*mesh_points_3d[0, ii], frame_height*mesh_points_3d[1, ii])).astype(np.int32)
                #     cv2.circle(img, tuple(pos), 2, (0, 255, 0), -1)
                # cv2.arrowedLine(img, p1, p2, (0,0,200), 2)

                # storing data
                # fps += [1/duration]

                # show landmarks
                # for p in mesh_points:
                #     cv2.circle(img, p, 1, (255, 0, 255), cv2.FILLED)
                # for p in mesh_points_3d.T:
                #     cv2.circle(top, (int(p[0] * img_w), int(p[2] * img_h + img_h/2)), 1, (255, 0, 255), cv2.FILLED) 

                # show iris circle
                # (l_x, l_y), l_r = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                # (r_x, r_y), r_r = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                # (l_x, l_y), l_r = (int(l_x), int(l_y)), int(l_r)
                # (r_x, r_y), r_r = (int(r_x), int(r_y)), int(r_r)
                # cv2.circle(img, (l_x, l_y), l_r, (0, 255, 0), 1)
                # cv2.circle(img, (r_x, r_y), r_r, (0, 255, 0), 1)

                # show middle points
                # cv2.circle(img, mesh_points[LEFT_CENTER], 2, (255, 0, 255), cv2.FILLED)
                # cv2.circle(img, mesh_points[RIGHT_CENTER], 2, (255, 0, 255), cv2.FILLED)

                # show eyes polygons
                # cv2.polylines(img, [mesh_points[LEFT_EYE]], True, (0, 0, 255), 1)
                # cv2.polylines(img, [mesh_points[RIGHT_EYE]], True, (0, 0, 255), 1)

                # gaze circle
                # if left_blink:
                #     cv2.circle(board, (left_gaze_x, left_gaze_y), 12, (0, 255, 255), cv2.FILLED)
                # else:
                #     cv2.circle(board, (left_gaze_x, left_gaze_y), 10, (0, 255, 255), 2)
                # if right_blink:
                #     cv2.circle(board, (right_gaze_x, right_gaze_y), 12, (255, 255, 0), cv2.FILLED)
                # else:
                #     cv2.circle(board, (right_gaze_x, right_gaze_y), 10, (255, 255, 0), 2)
                # if blinked:
                #     cv2.circle(board, ((right_gaze_x + left_gaze_x)//2, (right_gaze_y + left_gaze_y)//2),
                #                 12, (255, 255, 255), cv2.FILLED)
                # else:
                #     cv2.circle(board, ((right_gaze_x + left_gaze_x)//2, (right_gaze_y + left_gaze_y)//2),
                #                 10, (255, 255, 255), 2)

                # head position cross
                cv2cross(board, (int(head_x_raw), int(head_y_raw)), 10, (255, 255, 255), 2)
                cv2cross(board, (int(head_x), int(head_y)), 14, (255, 255, 255), 2)
                cv2cross(board, (head_gaze_x_raw, head_gaze_y_raw), 10, (255, 0, 255), 2, pi*roll/180)
                cv2cross(board, (head_gaze_x, head_gaze_y), 14, (255, 0, 255), 2, pi*roll/180)

                # show FPS and blink
                fps_str = str(int(duration * 1000))
                fps_str += ' = '
                fps_str += str(int(process_duration * 1000))
                fps_str += ' + '
                fps_str += str(int((duration - process_duration) * 1000))
                fps_str += ' ms, '
                fps_str += f"{(1 / duration):.1f}"
                cv2.putText(img, fps_str, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if left_blink:
                    cv2.putText(img, 'left blink', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                if right_blink:
                    cv2.putText(img, 'right blink', (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                # windows
                cv2.imshow("Cam", img)
                # cv2.imshow("Top", top)
                cv2.imshow("Board", board)

                # 3d plots
                if PLOT3D:
                    # 3d face mesh
                    ax.plot(mesh_points_3d[0], mesh_points_3d[2], -mesh_points_3d[1], 'ob', markersize=0.5)
                    plt.xlim([0, 1])
                    plt.ylim([-1, 1])
                    ax.set_zlim([-1, 0])

                    # equivalent robot

                    # common
                    plt.draw()
                    plt.pause(0.02)
                    ax.cla()

            elif MODE == CALLIBRATION:
                callibration_return = callibration(gc, hc)
                MODE = IDLE
                cv2.destroyAllWindows()

            elif MODE == PAINT:

                # calculate coordinates
                left_iris_x, right_iris_x, left_iris_y, right_iris_y = iris_positions(mesh_points)

                if not is_blinked(mesh_points, gc):
                    # filter
                    left_x_filter.add_input(left_iris_x)
                    left_y_filter.add_input(left_iris_y)
                    right_x_filter.add_input(right_iris_x)
                    right_y_filter.add_input(right_iris_y)

                paint_points += [gc.position2gaze(left_x_filter.output(), left_y_filter.output(),
                                                  right_x_filter.output(), right_y_filter.output())]

                if len(paint_points) > 1:
                    cv2.polylines(board, [np.array(paint_points)], False, (255, 0, 255), 2)

                cv2.imshow('Board', board)

            elif MODE == EVALUATION:
                evaluation(gc, hc)
                MODE = IDLE

        
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('i'):
            MODE = IDLE
            cv2.destroyAllWindows()
        elif key == ord('c'):
            MODE = CALLIBRATION
            cv2.destroyAllWindows()
        elif key == ord('e'):
            MODE = EVALUATION
            cv2.destroyAllWindows()
        elif key == ord('p'):
            paint_points = []
            MODE = PAINT
            cv2.destroyAllWindows()



        loop_end = time.time()
        duration = loop_end - loop_start
        process_duration = loop_end - process_start


cap.release
cv2.destroyAllWindows()

left_gaze_arr = np.array(left_gaze_arr)
left_blink_arr = np.array(left_blink_arr)
right_gaze_arr = np.array(right_gaze_arr)
right_blink_arr = np.array(right_blink_arr)
# test_arr = np.array(test_arr)

# scipy.io.savemat('lg640x480.mat', {'data': left_gaze_arr})

# plt.figure()
# plt.plot(test_arr[:, 0], test_arr[:, 1:])

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.title("left eye x")
# plt.plot(left_gaze_arr[:, 0], left_gaze_arr[:, 1])
# plt.subplot(2, 1, 2)
# plt.title("right eye x")
# plt.xlabel('t (s)')
# plt.plot(right_gaze_arr[:, 0], right_gaze_arr[:, 1])
# if left_blink_arr.size != 0:
#     plt.plot(left_blink_arr[:, 0], left_blink_arr[:, 1], 'o')
# if right_blink_arr.size != 0:
#     plt.plot(right_blink_arr[:, 0], right_blink_arr[:, 1], 'o')
# plt.legend(['left', 'right'])
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.title("left eye y")
# plt.plot(left_gaze_arr[:, 0], left_gaze_arr[:, 2])
# plt.subplot(2, 1, 2)
# plt.title("right eye y")
# plt.plot(right_gaze_arr[:, 0], right_gaze_arr[:, 2])
# if left_blink_arr.size != 0:
#     plt.plot(left_blink_arr[:, 0], left_blink_arr[:, 2], 'o')
# if right_blink_arr.size != 0:
#     plt.plot(right_blink_arr[:, 0], right_blink_arr[:, 2], 'o')
# plt.legend(['left', 'right'])

# plt.figure()
# plt.title("FPS")
# plt.plot(fps)

# plt.show()
