import cv2
import time

cap = cv2.VideoCapture(0)
duration = time.time()

while True:
	start = time.time()
	ret, img = cap.read()
	cv2.putText(img, str((int) (duration * 1000)) + ' ms, ' + str((int) (1 / duration)) + ' FPS', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
	cv2.imshow("cam", img)
	if cv2.waitKey(1) == 27:
		break
	end = time.time()
	duration = end - start

cap.release
cv2.destroyAllWindows()