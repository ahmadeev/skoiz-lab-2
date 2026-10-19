import cv2
import numpy as np


cap = cv2.VideoCapture("test-videos/mona-lisa.avi")

success1, reference = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
success2, frame = cap.read()

cap.release()

if not success1 or not success2:
    raise RuntimeError("Не удалось прочитать кадры")

orb = cv2.SIFT_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(
    cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None
)

kp2, des2 = orb.detectAndCompute(
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
)

if des1 is None or des2 is None:
    raise RuntimeError("Дескрипторы не найдены")

matcher = cv2.BFMatcher(cv2.NORM_L2)
matches = matcher.knnMatch(des1, des2, k=2)

good_matches = []

for pair in matches:
    if len(pair) < 2:
        continue

    m, n = pair

    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

good_matches.sort(key=lambda m: m.distance)

if len(good_matches) < 4:
    raise RuntimeError("Недостаточно соответствий для RANSAC")

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
    raise RuntimeError("Не удалось вычислить гомографию")

inlier_matches = [
    m for m, is_inlier in zip(good_matches, mask.ravel())
    if is_inlier
]

result = cv2.drawMatches(
    reference, kp1,
    frame, kp2,
    inlier_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite("ransac_matches.jpg", result)

print(f"Соответствий после BFMatcher: {len(good_matches)}")
print(f"Inliers: {len(inlier_matches)}")
print(f"Outliers: {len(good_matches) - len(inlier_matches)}")
print(f"Доля inliers: {len(inlier_matches) / len(good_matches):.1%}")
