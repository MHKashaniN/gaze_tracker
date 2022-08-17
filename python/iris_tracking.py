import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from math import hypot, inf, cos, atan, sin, tan, asin, pi, sqrt
from face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis

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


def get_head_orientation(rvec, tvec):
    rmat = cv2.Rodrigues(rvec)[0]  
    P = np.hstack((rmat,tvec)) # projection matrix
    
    # find euler angles 
    euler_angles =  cv2.decomposeProjectionMatrix(P)[6]
    pitch = -euler_angles.item(0) 
    yaw = -euler_angles.item(1) 
    roll = euler_angles.item(2) 

    # Ajust coordinate ranges
    if pitch < 0:
      pitch = 180 + pitch
    else:
      pitch = pitch - 180

    t = '{y},{p},{r}'.format(y=round(yaw),
                             p=round(pitch),
                             r=round(roll)) 
    return round(roll), round(pitch), round(yaw)

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

    # left_projected_center = project(_mesh_points[LEFT_EYE[0]], left_iris_center, _mesh_points[LEFT_EYE[8]])
    # right_projected_center = project(_mesh_points[RIGHT_EYE[0]], right_iris_center, _mesh_points[RIGHT_EYE[8]])

    # l_h = length(_mesh_points[LEFT_EYE[0]], _mesh_points[LEFT_EYE[8]])
    # r_h = length(_mesh_points[RIGHT_EYE[0]], _mesh_points[RIGHT_EYE[8]])

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
    l_ratio, r_ratio = eyes_ratio(_mesh_points)
    left_iris_y = l_ratio
    right_iris_y = r_ratio

    # calculate y coordinates based on head horizontal line
    # left_iris_y = signed_dist(_mesh_points[RIGHT_EYE[0]], left_iris_center, _mesh_points[LEFT_EYE[0]]) / l_h
    # right_iris_y = signed_dist(_mesh_points[RIGHT_EYE[0]], right_iris_center, _mesh_points[LEFT_EYE[0]]) / r_h


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

def is_blinked(_mesh_points):
    l_ratio, r_ratio = eyes_ratio(_mesh_points)
    right_blink = r_ratio < R_BLINK_RATIO
    left_blink = l_ratio < L_BLINK_RATIO
    return left_blink or right_blink

def get_mean_position(points_num):
    data = []
    t = []
    while True:
        ret, img = cap.read()
        img_h, img_w = img.shape[:2]
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            mesh_points = np.array([[int(p.x * img_w), int(p.y * img_h)] for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3d = np.array([(p.x, p.y, p.z) for p in results.multi_face_landmarks[0].landmark])[:468].T

            if not is_blinked(mesh_points):
                left_iris_x, right_iris_x, left_iris_y, right_iris_y = iris_positions(mesh_points)
                data += [[left_iris_x, right_iris_x, left_iris_y, right_iris_y]]
                t += [left_iris_y]

        cv2.imshow('cam', img)

        if cv2.waitKey(1) == 32:
            data = np.array(data)
            return t, np.mean(data[-points_num:, :], 0)

def callibration():
    t = []
    POINTS_NUM = 20
    result = []
    cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
    cv2.imshow("Board", cal_board)
    cv2.waitKey(0)

    cv2.circle(cal_board, (BOARD_WIDTH // 2, BOARD_HEIGHT // 2), 15, (255, 0, 255), cv2.FILLED)
    cv2.imshow("Board", cal_board)
    returned_t, returned_reasult = get_mean_position(POINTS_NUM)
    t += returned_t
    result += [returned_reasult]

    for j in range(3):
        for i in range(4):
            cal_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)
            cv2.circle(cal_board, (int(i * BOARD_WIDTH / 3), int(j * BOARD_HEIGHT / 2)), 15, (255, 0, 255), cv2.FILLED)
            cv2.imshow("Board", cal_board)
            returned_t, returned_reasult = get_mean_position(POINTS_NUM)
            t += returned_t
            result += [returned_reasult]

    plt.figure('X')
    plt.plot(t)
    plt.show()

    plt.figure('left iris x callibration')
    plt.plot(np.array(range(4)) * BOARD_WIDTH / 3, np.array(result)[1:5, 0])
    plt.plot(np.array(range(4)) * BOARD_WIDTH / 3, np.array(result)[5:9, 0])
    plt.plot(np.array(range(4)) * BOARD_WIDTH / 3, np.array(result)[9:13, 0])
    plt.legend(['y = 0', 'y = ' + str(BOARD_HEIGHT/2), 'y = ' + str(BOARD_HEIGHT)])

    plt.figure('right iris x callibration')
    plt.plot(np.array(range(4)) * BOARD_WIDTH / 3, np.array(result)[1:5, 1])
    plt.plot(np.array(range(4)) * BOARD_WIDTH / 3, np.array(result)[5:9, 1])
    plt.plot(np.array(range(4)) * BOARD_WIDTH / 3, np.array(result)[9:13, 1])
    plt.legend(['y = 0', 'y = ' + str(BOARD_HEIGHT/2), 'y = ' + str(BOARD_HEIGHT)])

    plt.figure('left iris y callibration')
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[1, 5, 9], 2])
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[2, 6, 10], 2])
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[3, 7, 11], 2])
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[4, 8, 12], 2])
    plt.legend(['x = 0', 'x = ' + str(BOARD_WIDTH/3), 'x = ' + str(BOARD_WIDTH*2/3), 'x = ' + str(BOARD_WIDTH)])

    plt.figure('right iris y callibration')
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[1, 5, 9], 3])
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[2, 6, 10], 3])
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[3, 7, 11], 3])
    plt.plot(np.array(range(3)) * BOARD_WIDTH/2, np.array(result)[[4, 8, 12], 3])
    plt.legend(['x = 0', 'x = ' + str(BOARD_WIDTH/3), 'x = ' + str(BOARD_WIDTH*2/3), 'x = ' + str(BOARD_WIDTH)])

    plt.show()

    return result

cap = cv2.VideoCapture(0) #'eye_movement_test.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print('resolution: ', end='')
print((frame_width, frame_height))
# pseudo camera internals
focal_length = frame_width
center = (frame_width/2, frame_height/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")
pcf = PCF(near=1,far=10000,frame_height=frame_height,frame_width=frame_width,fy=camera_matrix[1,1])
dist_coeff = np.zeros((4, 1))

mp_face_mesh = mp.solutions.face_mesh

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE = [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173]

RIGHT_IRIS = range(474, 478)
LEFT_IRIS = range(469, 473)

RIGHT_CENTER = 473
LEFT_CENTER = 468

L_BLINK_RATIO = 0.1
R_BLINK_RATIO = 0.15

BUFFER_LENGTH = 10

BOARD_WIDTH = 1200
BOARD_HEIGHT = 600

points_idx =  [33,263,61,291,199] # [k for k in range(0,468)] 
points_idx = points_idx + [key for (key,val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

fps = []
right_gaze_arr = [] # (t, x, y)
left_gaze_arr = []
right_blink_arr = [] # (t, x0, y0)
left_blink_arr = []

right_gaze_buffer = np.zeros((BUFFER_LENGTH, 2)) # (x, y)
left_gaze_buffer = np.zeros((BUFFER_LENGTH, 2))
buffer_index = 0

left_gaze = [0, 0]
right_gaze = [0, 0]

IDLE = 0
CALLIBRATION = 1
EVALUATION = 2
MODE = IDLE

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh: 
    
    duration = time.time()
    process_duration = time.time()
    start = time.time()
    
    while True:
        loop_start = time.time()
        ret, img = cap.read()
        if not ret:
            break
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
                right_blink = r_ratio < R_BLINK_RATIO
                left_blink = l_ratio < L_BLINK_RATIO
                blinked = left_blink or right_blink
                
                # calculate coordinates 
                left_iris_x, right_iris_x, left_iris_y, right_iris_y = iris_positions(mesh_points)

                # calculate head orientations
                metric_landmarks, pose_transform_mat = get_metric_landmarks(mesh_points_3d.copy(), pcf)
                model_points = metric_landmarks[0:3, points_idx].T
                image_points = mesh_points_3d[0:2, points_idx].T * np.array([frame_width, frame_height])[None,:]

                success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, 
                                                                            dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)


                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 25.0)]), rotation_vector, 
                                                                    translation_vector, camera_matrix, dist_coeff)
              
                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                roll, pitch, yaw = get_head_orientation(rotation_vector, translation_vector)
                # print((roll, pitch, yaw))
                # print(translation_vector)

                # calculate gaze
                if left_blink:
                    left_blink_arr += [[time.time() - start, left_gaze[0], left_gaze[1]]]
                if right_blink:
                    right_blink_arr += [[time.time() - start, right_gaze[0], right_gaze[1]]]

                if not blinked:
                    # filter
                    left_gaze_buffer[buffer_index] = [left_iris_x, left_iris_y]
                    right_gaze_buffer[buffer_index] = [right_iris_x, right_iris_y]
                    buffer_index = (buffer_index + 1) % BUFFER_LENGTH

                    left_gaze = np.mean(left_gaze_buffer, 0)
                    right_gaze = np.mean(right_gaze_buffer, 0)

                    left_gaze_arr += [[time.time() - start, left_gaze[0], left_gaze[1]]]
                    right_gaze_arr += [[time.time() - start, right_gaze[0], right_gaze[1]]]

                left_gaze_x = int(-14000 * tan(asin(left_gaze[0] - 0.56)) + BOARD_WIDTH/2)
                right_gaze_x = int(14000 * tan(asin(right_gaze[0] - 0.56)) + BOARD_WIDTH/2)
                left_gaze_y = int(-14000 * tan(asin(left_gaze[1] - 0.31)) + BOARD_HEIGHT/2)
                right_gaze_y = int(-14000 * tan(asin(right_gaze[1] - 0.31)) + BOARD_HEIGHT/2)

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
                if left_blink:
                    cv2.circle(board, (left_gaze_x, left_gaze_y), 12, (0, 255, 255), cv2.FILLED)
                else:
                    cv2.circle(board, (left_gaze_x, left_gaze_y), 10, (0, 255, 255), 2)
                if right_blink:
                    cv2.circle(board, (right_gaze_x, right_gaze_y), 12, (255, 255, 0), cv2.FILLED)
                else:
                    cv2.circle(board, (right_gaze_x, right_gaze_y), 10, (255, 255, 0), 2)
                if blinked:
                    cv2.circle(board, ((right_gaze_x + left_gaze_x)//2, (right_gaze_y + left_gaze_y)//2), 
                                12, (255, 255, 255), cv2.FILLED)
                else:
                    cv2.circle(board, ((right_gaze_x + left_gaze_x)//2, (right_gaze_y + left_gaze_y)//2), 
                                10, (255, 255, 255), 2)

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

            elif MODE == CALLIBRATION:
                callibration_return = callibration()
                print(callibration_return)
                for p in callibration_return:
                    plt.plot(p[0], p[2], 'or')
                    plt.plot(p[1], p[3], 'ob')
                plt.plot(callibration_return[0][0], callibration_return[0][2], '^c')
                plt.plot(callibration_return[0][1], callibration_return[0][3], '^c')
                plt.show()
                MODE = IDLE
                cv2.destroyAllWindows()

        
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('c'):
            MODE = CALLIBRATION
            cv2.destroyAllWindows()
        elif key == ord('e'):
            MODE = EVALUATION
            cv2.destroyAllWindows()
        loop_end = time.time()
        duration = loop_end - loop_start
        process_duration = loop_end - process_start


cap.release
cv2.destroyAllWindows()

# left_gaze_arr = np.array(left_gaze_arr)
# left_blink_arr = np.array(left_blink_arr)
# right_gaze_arr = np.array(right_gaze_arr)
# right_blink_arr = np.array(right_blink_arr)

# plt.figure("x")
# plt.plot(left_gaze_arr[:, 0], left_gaze_arr[:, 1])
# plt.plot(right_gaze_arr[:, 0], right_gaze_arr[:, 1])
# if left_blink_arr.size != 0:
#     plt.plot(left_blink_arr[:, 0], left_blink_arr[:, 1], 'o')
# if right_blink_arr.size != 0:
#     plt.plot(right_blink_arr[:, 0], right_blink_arr[:, 1], 'o')
# plt.legend(['left', 'right'])
# plt.figure("y")
# plt.plot(left_gaze_arr[:, 0], left_gaze_arr[:, 2])
# plt.plot(right_gaze_arr[:, 0], right_gaze_arr[:, 2])
# if left_blink_arr.size != 0:
#     plt.plot(left_blink_arr[:, 0], left_blink_arr[:, 2], 'o')
# if right_blink_arr.size != 0:
#     plt.plot(right_blink_arr[:, 0], right_blink_arr[:, 2], 'o')
# plt.legend(['left', 'right'])

# plt.figure("FPS")
# plt.plot(fps)

# plt.show()