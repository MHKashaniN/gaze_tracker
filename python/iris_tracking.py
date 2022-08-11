import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
duration = time.time()
start = time.time()

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173]

LEFT_IRIS = range(474, 478)
RIGHT_IRIS = range(469, 473)

LEFT_CENTER = 473
RIGHT_CENTER = 468

fps = []
gaze = []
gaze_t = []

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh: 

    while True: 
        loop_start = time.time()
        _, img = cap.read()
        img_h, img_w = img.shape[:2]
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img)

        if results.multi_face_landmarks:
            mesh_points = np.array([[int(p.x * img_w), int(p.y * img_h)] for p in results.multi_face_landmarks[0].landmark])

            gaze_t += [time.time() - start]
            gaze += [mesh_points[LEFT_CENTER]]

            # show landmarks
            # for p in results.multi_face_landmarks[0].landmark:
            #     cv2.circle(img, (int(p.x * img_w), int(p.y * img_h)), 1, (255, 0, 255), cv2.FILLED) 

            # show iris circle
            (l_x, l_y), l_r = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_x, r_y), r_r = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            (l_x, l_y), l_r = (int(l_x), int(l_y)), int(l_r)
            (r_x, r_y), r_r = (int(r_x), int(r_y)), int(r_r)
            cv2.circle(img, (l_x, l_y), l_r, (0, 255, 0), 1)
            cv2.circle(img, (r_x, r_y), r_r, (0, 255, 0), 1)

            # show middle points
            cv2.circle(img, mesh_points[LEFT_CENTER], 2, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, mesh_points[RIGHT_CENTER], 2, (255, 0, 255), cv2.FILLED)

            # show eyes polygons
            cv2.polylines(img, [mesh_points[LEFT_EYE]], True, (0, 0, 255), 1)
            cv2.polylines(img, [mesh_points[RIGHT_EYE]], True, (0, 0, 255), 1)


        cv2.putText(img, str(int(duration * 1000)) + ' ms, ' + f"{(1 / duration):.1f}" + ' FPS',
                     (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        fps += [int(1 / duration)]
        cv2.imshow("Cam", img)
        
        if cv2.waitKey(1) == 27:
            break
        loop_end = time.time()
        duration = loop_end - loop_start


cap.release
cv2.destroyAllWindows()

plt.plot(gaze_t, gaze)
# plt.plot(blink_t, blink, 'o')
plt.show()