import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from math import hypot, inf, cos, atan, sin, tan, asin
from face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis

def length(p1, p2):
    return hypot(p1[0] - p2[0], p1[1] - p2[1])

def slope(src, dest):
    dy = dest[1] - src[1]
    dx = dest[0] - src[0]
    if dx == 0:
        return inf if dy>0 else -inf
    return dy/dx

def project(common_point, point, land_point):
    vect1 = [point[0] - common_point[0], point[1] - common_point[1]]
    vect2 = [land_point[0] - common_point[0], land_point[1] - common_point[1]]
    dot_product = vect1[0] * vect2[0] + vect1[1] * vect2[1]
    l = dot_product / length(common_point, land_point)
    m = slope(common_point, land_point)
    dx = abs(l * cos(atan(m)))
    if point[0] < common_point[0]:
        dx = -dx
    dy = abs(l * sin(atan(m)))
    if point[1] < common_point[1]:
        dy = -dy
    return [int(common_point[0] + dx), int(common_point[1] + dy)]

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
    return t


cap = cv2.VideoCapture(0)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# pseudo camera internals
focal_length = frame_width
center = (frame_width/2, frame_height/2)
camera_matrix = np.array(
                        [[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")
pcf = PCF(near=1,far=10000,frame_height=frame_height,frame_width=frame_width,fy=camera_matrix[1,1])
dist_coeff = np.zeros((4, 1))

duration = time.time()
start = time.time()

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
gaze = []
gaze_t = []
blink = []
blink_t = []

buffer = np.zeros((BUFFER_LENGTH, 2))
buffer_index = 0

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh: 

    while True: 
        loop_start = time.time()
        _, img = cap.read()
        img_h, img_w = img.shape[:2]
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img)
        top = np.zeros((480, 640, 3), np.uint8)
        board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH, 3), np.uint8)

        if results.multi_face_landmarks:
            mesh_points = np.array([[int(p.x * img_w), int(p.y * img_h)] for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3d = np.array([(p.x, p.y, p.z) for p in results.multi_face_landmarks[0].landmark])[:468].T

            # blink detection
            l_h = length(mesh_points[LEFT_EYE[0]], mesh_points[LEFT_EYE[8]])
            r_h = length(mesh_points[RIGHT_EYE[0]], mesh_points[RIGHT_EYE[8]])
            l_v = length(mesh_points[LEFT_EYE[4]], mesh_points[LEFT_EYE[12]])
            r_v = length(mesh_points[RIGHT_EYE[4]], mesh_points[RIGHT_EYE[12]])
            l_ratio = l_v/l_h
            r_ratio = r_v/r_h
            if l_ratio < L_BLINK_RATIO:
                cv2.putText(img, 'left blink', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if r_ratio < R_BLINK_RATIO:
                cv2.putText(img, 'right blink', (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            # calculate coordinates
            left_iris_center = mesh_points[LEFT_CENTER]
            right_iris_center = mesh_points[RIGHT_CENTER]

            left_projected_center = project(mesh_points[LEFT_EYE[0]], left_iris_center, mesh_points[LEFT_EYE[8]])
            right_projected_center = project(mesh_points[RIGHT_EYE[0]], right_iris_center, mesh_points[RIGHT_EYE[8]])

            left_iris_x = length(left_iris_center, mesh_points[LEFT_EYE[0]]) / l_h
            # left_iris_y = length(left_projected_center, mesh_points[LEFT_EYE[0]]) / l_h
            right_iris_x = length(right_projected_center, mesh_points[RIGHT_EYE[0]]) / r_h
            # right_iris_x = length(right_projected_center, mesh_points[RIGHT_EYE[0]]) / r_h

            # calculate head orientations
            metric_landmarks, pose_transform_mat = get_metric_landmarks(mesh_points_3d.copy(), pcf)
            model_points = metric_landmarks[0:3, points_idx].T
            image_points = mesh_points_3d[0:2, points_idx].T * np.array([frame_width, frame_height])[None,:]

            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 25.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeff)
          
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            print(get_head_orientation(rotation_vector, translation_vector))

            # calculate gaze
            left_gaze = int(-14000 * tan(asin(np.mean(buffer, 0)[0] - 0.56)) + BOARD_WIDTH/2)

            # show orientation calculations
            # for ii in points_idx: # range(landmarks.shape[1]):
            #     pos = np.array((frame_width*mesh_points_3d[0, ii], frame_height*mesh_points_3d[1, ii])).astype(np.int32)
            #     cv2.circle(img, tuple(pos), 2, (0, 255, 0), -1)
            # cv2.arrowedLine(img, p1, p2, (0,0,200), 2)

            # show axis and coordinates
            # cv2.line(img, mesh_points[LEFT_EYE[0]], mesh_points[LEFT_EYE[8]], (127, 127, 127), 1)
            # cv2.line(img, mesh_points[RIGHT_EYE[0]], mesh_points[RIGHT_EYE[8]], (127, 127, 127), 1)
            # cv2.line(img, mesh_points[LEFT_EYE[0]], left_iris_center, (255, 0, 255), 1)
            # cv2.line(img, mesh_points[RIGHT_EYE[0]], right_iris_center, (255, 0, 255), 1)
            # cv2.circle(img, mesh_points[RIGHT_EYE[0]], 2, (255, 0, 0), cv2.FILLED)
            # cv2.circle(img, mesh_points[LEFT_EYE[0]], 2, (255, 0, 0), cv2.FILLED)
            # cv2.circle(img, left_projected_center, 2, (0, 0, 255), cv2.FILLED)
            # cv2.circle(img, right_projected_center, 2, (0, 0, 255), cv2.FILLED)

            # show horizontal and vertical lines
            # cv2.line(img, mesh_points[LEFT_EYE[0]], mesh_points[LEFT_EYE[8]], (0, 255, 0), 1)
            # cv2.line(img, mesh_points[LEFT_EYE[4]], mesh_points[LEFT_EYE[12]], (0, 255, 0), 1)
            # cv2.line(img, mesh_points[RIGHT_EYE[0]], mesh_points[RIGHT_EYE[8]], (0, 255, 0), 1)
            # cv2.line(img, mesh_points[RIGHT_EYE[4]], mesh_points[RIGHT_EYE[12]], (0, 255, 0), 1)

            # storing data
            # fps += [1/duration]
            
            if l_ratio < L_BLINK_RATIO and r_ratio < R_BLINK_RATIO:
                blink += [np.mean(buffer, 0)]
                blink_t += [time.time() - start]
            else:
                # filter
                buffer[buffer_index] = [left_iris_x, right_iris_x]
                buffer_index = (buffer_index + 1) % BUFFER_LENGTH

                gaze_t += [time.time() - start]
                gaze += [np.mean(buffer, 0)]

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
            cv2.circle(board, (left_gaze, BOARD_HEIGHT//2), 10, (0, 255, 255), 2)


        cv2.putText(img, str(int(duration * 1000)) + ' ms, ' + f"{(1 / duration):.1f}" + ' FPS',
                     (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Cam", img)
        # cv2.imshow("Top", top)
        # cv2.imshow("Board", board)
        
        if cv2.waitKey(1) == 27:
            break
        loop_end = time.time()
        duration = loop_end - loop_start


cap.release
cv2.destroyAllWindows()

# plt.plot(gaze_t, gaze)
# plt.plot(blink_t, blink, 'o')
# plt.plot(fps)
# plt.legend(['left', 'right'])
# plt.show()