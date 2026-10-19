import cv2


cap = cv2.VideoCapture("test-videos/mona-lisa.avi")
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Не удалось прочитать первый кадр")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=1000)
keypoints, descriptors = orb.detectAndCompute(gray, None)

if descriptors is None:
    raise RuntimeError("Ключевые точки не обнаружены")

result = cv2.drawKeypoints(
    frame, keypoints, None, color=(0, 255, 0)
)

cv2.imwrite("keypoints.jpg", result)

print(f"Найдено точек: {len(keypoints)}")
print(f"Размер дескрипторов: {descriptors.shape}")
