import cv2


cap = cv2.VideoCapture("test-videos/mona-lisa.avi")

success1, reference = cap.read()

cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
success2, frame = cap.read()

cap.release()

if not success1 or not success2:
    raise RuntimeError("Не удалось прочитать кадры")

orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(
    cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None
)

kp2, des2 = orb.detectAndCompute(
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
)

if des1 is None or des2 is None:
    raise RuntimeError("Дескрипторы не найдены")

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = matcher.knnMatch(des1, des2, k=2)

good_matches = []

for pair in matches:
    if len(pair) < 2:
        continue

    m, n = pair

    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

good_matches.sort(key=lambda m: m.distance)

result = cv2.drawMatches(
    reference, kp1,
    frame, kp2,
    good_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite("matches.jpg", result)

print(f"Точек на эталоне: {len(kp1)}")
print(f"Точек на кадре: {len(kp2)}")
print(f"Всего соответствий: {len(matches)}")
print(f"После фильтрации: {len(good_matches)}")
