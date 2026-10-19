import cv2
import numpy as np


cap = cv2.VideoCapture("test-videos/mona-lisa.avi")

if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

success, reference = cap.read()

if not success:
    cap.release()
    raise RuntimeError("Не удалось прочитать первый кадр")

height, width = reference.shape[:2]
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Не удалось определить FPS видео")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    "tracked_video.mp4", fourcc, fps, (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Не удалось создать выходное видео")

sift = cv2.SIFT_create(nfeatures=1000)
kp1, des1 = sift.detectAndCompute(
    cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None
)

if des1 is None:
    cap.release()
    writer.release()
    raise RuntimeError("На эталоне не найдены дескрипторы")

matcher = cv2.BFMatcher(cv2.NORM_L2)

corners = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
]).reshape(-1, 1, 2)


def detect_object(frame):
    kp2, des2 = sift.detectAndCompute(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
    )

    if des2 is None:
        return None

    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return None

    src_points = np.float32([
        kp1[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        kp2[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    if H is None or mask is None:
        return None

    inliers = np.count_nonzero(mask)

    if inliers < 10 or inliers / len(good_matches) < 0.5:
        return None

    projected = cv2.perspectiveTransform(corners, H)

    if not np.isfinite(projected).all():
        return None

    quad = projected.reshape(4, 2)

    if not cv2.isContourConvex(quad):
        return None

    area = cv2.contourArea(quad)
    frame_area = width * height

    if not 0.005 * frame_area <= area <= 1.2 * frame_area:
        return None

    x, y = quad[:, 0], quad[:, 1]

    if (
            (x < -0.2 * width).any()
            or (x > 1.2 * width).any()
            or (y < -0.2 * height).any()
            or (y > 1.2 * height).any()
    ):
        return None

    return projected


processed = 0
detected = 0

try:
    frame = reference

    while True:
        projected = detect_object(frame)

        if projected is not None:
            detected += 1

            cv2.polylines(
                frame,
                [np.int32(projected)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=3
            )

            x, y = projected[0, 0].astype(int)

            cv2.putText(
                frame,
                "Object",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        writer.write(frame)
        processed += 1

        success, frame = cap.read()

        if not success:
            break
finally:
    cap.release()
    writer.release()

print(f"Обработано кадров: {processed}")
print(f"Объект обнаружен на: {detected}")
print(f"Доля обнаружений: {detected / processed:.1%}")
print("Результат: tracked_video.mp4")
