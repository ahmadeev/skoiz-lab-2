import cv2
import numpy as np


INPUT_VIDEO = "test-videos/mona-lisa.avi"
OUTPUT_VIDEO = "tracked_video.mp4"
VIDEO_CODEC = "mp4v"

SIFT_FEATURES = 1000
KNN_NEIGHBORS = 2
LOWE_RATIO = 0.75
MIN_MATCHES = 10
MIN_INLIERS = 10
MIN_INLIER_RATIO = 0.5
RANSAC_REPROJECTION_THRESHOLD = 5.0

MIN_AREA_RATIO = 0.005
MAX_AREA_RATIO = 1.2
FRAME_MARGIN_RATIO = 0.2

BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 3
POINT_COLOR = (0, 0, 255)
POINT_RADIUS = 3
LABEL = "Object"
LABEL_FONT_SCALE = 0.7
LABEL_THICKNESS = 2
LABEL_MIN_Y = 20
LABEL_Y_OFFSET = 10


cap = cv2.VideoCapture(INPUT_VIDEO)
writer = None

try:
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    success, reference = cap.read()

    if not success:
        raise RuntimeError("Не удалось прочитать первый кадр")

    height, width = reference.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Не удалось определить FPS видео")

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO, fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Не удалось создать выходное видео")

    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kp1, des1 = sift.detectAndCompute(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None
    )

    if des1 is None:
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

        matches = matcher.knnMatch(
            des1, des2, k=KNN_NEIGHBORS
        )

        good_matches = []

        for pair in matches:
            if len(pair) < KNN_NEIGHBORS:
                continue

            m, n = pair

            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)

        if len(good_matches) < MIN_MATCHES:
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
            ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD
        )

        if H is None or mask is None:
            return None

        inliers = np.count_nonzero(mask)

        if (
            inliers < MIN_INLIERS
            or inliers / len(good_matches) < MIN_INLIER_RATIO
        ):
            return None

        projected = cv2.perspectiveTransform(corners, H)

        if not np.isfinite(projected).all():
            return None

        quad = projected.reshape(4, 2)

        if not cv2.isContourConvex(quad):
            return None

        area = cv2.contourArea(quad)
        frame_area = width * height

        if not (
            MIN_AREA_RATIO * frame_area
            <= area
            <= MAX_AREA_RATIO * frame_area
        ):
            return None

        x, y = quad[:, 0], quad[:, 1]
        max_coordinate_ratio = 1 + FRAME_MARGIN_RATIO

        if (
            (x < -FRAME_MARGIN_RATIO * width).any()
            or (x > max_coordinate_ratio * width).any()
            or (y < -FRAME_MARGIN_RATIO * height).any()
            or (y > max_coordinate_ratio * height).any()
        ):
            return None

        inlier_points = [
            kp2[m.trainIdx].pt
            for m, valid in zip(good_matches, mask.ravel())
            if valid
        ]

        return projected, inlier_points

    processed = 0
    detected = 0
    frame = reference

    while True:
        detection = detect_object(frame)

        if detection is not None:
            projected, points = detection
            detected += 1

            for x, y in points:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    POINT_RADIUS,
                    POINT_COLOR,
                    -1
                )

            cv2.polylines(
                frame,
                [np.int32(projected)],
                isClosed=True,
                color=BOX_COLOR,
                thickness=BOX_THICKNESS
            )

            x, y = projected[0, 0].astype(int)

            cv2.putText(
                frame,
                LABEL,
                (x, max(LABEL_MIN_Y, y - LABEL_Y_OFFSET)),
                cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE,
                BOX_COLOR,
                LABEL_THICKNESS
            )

        writer.write(frame)
        processed += 1

        success, frame = cap.read()

        if not success:
            break
finally:
    cap.release()
    if writer is not None:
        writer.release()

print(f"Обработано кадров: {processed}")
print(f"Объект обнаружен на: {detected}")
print(f"Доля обнаружений: {detected / processed:.1%}")
print(f"Результат: {OUTPUT_VIDEO}")
