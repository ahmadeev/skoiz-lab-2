from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


VIDEO_INPUT_DIR = Path("test-videos/input")
VIDEO_OUTPUT_DIR = Path("test-videos/output")
INPUT_VIDEOS = [
    # VIDEO_INPUT_DIR / "mona-lisa.avi",
    # VIDEO_INPUT_DIR / "mona-lisa-blur.avi",
    # VIDEO_INPUT_DIR / "mona-lisa-blur-extra-credit.avi",
    VIDEO_INPUT_DIR / "video.MOV",
]
VIDEO_CODEC = "mp4v"
SIFT_FEATURES = 1000

REFERENCE_BOX_COLOR = (0, 255, 0)  # BGR
PREVIOUS_BOX_COLOR = (0, 165, 255)  # BGR
BOX_THICKNESS = 3
POINT_COLOR = (0, 0, 255)
POINT_RADIUS = 3
LABEL = "Object"
LABEL_FONT_SCALE = 0.7
LABEL_THICKNESS = 2
LABEL_MIN_Y = 20
LABEL_Y_OFFSET = 10


@dataclass(frozen=True)
class DetectionConfig:
    lowe_ratio: float = 0.75
    min_matches: int = 10
    min_inliers: int = 10
    min_inlier_ratio: float = 0.5
    ransac_reprojection_threshold: float = 5.0
    min_area_ratio: float = 0.005
    max_area_ratio: float = 1.2
    frame_margin_ratio: float = 0.2


@dataclass(frozen=True)
class FrameFeatures:
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: np.ndarray | None


@dataclass(frozen=True)
class DetectionResult:
    projected_corners: np.ndarray
    inlier_points: list[tuple[float, float]]


@dataclass
class ProcessingStats:
    processed_frames: int = 0
    reference_detections: int = 0
    previous_state_detections: int = 0

    @property
    def total_detections(self) -> int:
        """Возвращает общее количество успешных обнаружений."""
        return self.reference_detections + self.previous_state_detections

    @property
    def detection_rate(self) -> float:
        """Вычисляет долю кадров с обнаруженным объектом."""
        if self.processed_frames == 0:
            return 0.0

        return self.total_detections / self.processed_frames


def extract_features(frame: np.ndarray, feature_detector) -> FrameFeatures:
    """Извлекает ключевые точки и дескрипторы из кадра."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = feature_detector.detectAndCompute(
        gray_frame, None
    )
    return FrameFeatures(keypoints, descriptors)


def detect_object(
    source_features: FrameFeatures,
    source_corners: np.ndarray,
    target_features: FrameFeatures,
    matcher,
    frame_size: tuple[int, int],
    config: DetectionConfig,
) -> DetectionResult | None:
    """Обнаруживает объект на кадре по признакам выбранного источника."""
    if (
        source_features.descriptors is None
        or target_features.descriptors is None
    ):
        return None

    matches = matcher.knnMatch(
        source_features.descriptors,
        target_features.descriptors,
        k=2,
    )
    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue

        first_match, second_match = pair[:2]

        if first_match.distance < config.lowe_ratio * second_match.distance:
            good_matches.append(first_match)

    if len(good_matches) < config.min_matches:
        return None

    source_points = np.float32([
        source_features.keypoints[match.queryIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)
    target_points = np.float32([
        target_features.keypoints[match.trainIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        ransacReprojThreshold=config.ransac_reprojection_threshold,
    )

    if homography is None or mask is None:
        return None

    inliers = np.count_nonzero(mask)

    if (
        inliers < config.min_inliers
        or inliers / len(good_matches) < config.min_inlier_ratio
    ):
        return None

    projected_corners = cv2.perspectiveTransform(
        source_corners, homography
    )

    if not np.isfinite(projected_corners).all():
        return None

    quadrilateral = projected_corners.reshape(4, 2)

    if not cv2.isContourConvex(quadrilateral):
        return None

    width, height = frame_size
    area = cv2.contourArea(quadrilateral)
    frame_area = width * height

    if not (
        config.min_area_ratio * frame_area
        <= area
        <= config.max_area_ratio * frame_area
    ):
        return None

    x_coordinates = quadrilateral[:, 0]
    y_coordinates = quadrilateral[:, 1]
    max_coordinate_ratio = 1 + config.frame_margin_ratio

    if (
        (x_coordinates < -config.frame_margin_ratio * width).any()
        or (x_coordinates > max_coordinate_ratio * width).any()
        or (y_coordinates < -config.frame_margin_ratio * height).any()
        or (y_coordinates > max_coordinate_ratio * height).any()
    ):
        return None

    inlier_points = [
        target_features.keypoints[match.trainIdx].pt
        for match, valid in zip(good_matches, mask.ravel())
        if valid
    ]
    return DetectionResult(projected_corners, inlier_points)


def select_object_features(
    features: FrameFeatures,
    projected_corners: np.ndarray,
    min_matches: int,
) -> FrameFeatures | None:
    """Отбирает признаки объекта для отслеживания на следующем кадре."""
    if features.descriptors is None:
        return None

    contour = projected_corners.reshape(-1, 2).astype(np.float32)
    indexes = [
        index
        for index, keypoint in enumerate(features.keypoints)
        if cv2.pointPolygonTest(
            contour, keypoint.pt, measureDist=False
        ) >= 0
    ]

    if len(indexes) < min_matches:
        return None

    keypoints = [features.keypoints[index] for index in indexes]
    descriptors = features.descriptors[indexes]
    return FrameFeatures(keypoints, descriptors)


def draw_detection(
    frame: np.ndarray,
    detection: DetectionResult,
    box_color: tuple[int, int, int],
) -> None:
    """Наносит результат обнаружения объекта на кадр."""
    for x_coordinate, y_coordinate in detection.inlier_points:
        cv2.circle(
            frame,
            (int(x_coordinate), int(y_coordinate)),
            POINT_RADIUS,
            POINT_COLOR,
            -1,
        )

    cv2.polylines(
        frame,
        [np.int32(detection.projected_corners)],
        isClosed=True,
        color=box_color,
        thickness=BOX_THICKNESS,
    )

    x_coordinate, y_coordinate = detection.projected_corners[0, 0].astype(
        int
    )
    cv2.putText(
        frame,
        LABEL,
        (
            x_coordinate,
            max(LABEL_MIN_Y, y_coordinate - LABEL_Y_OFFSET),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        box_color,
        LABEL_THICKNESS,
    )


def print_processing_stats(
    stats: ProcessingStats,
    output_video: str | Path,
) -> None:
    """Выводит итоговую статистику обработки видео."""
    print(f"Обработано кадров: {stats.processed_frames}")
    print(f"Обнаружено по эталону: {stats.reference_detections}")
    print(
        "Обнаружено по предыдущему состоянию: "
        f"{stats.previous_state_detections}"
    )
    print(f"Всего обнаружений: {stats.total_detections}")
    print(f"Доля обнаружений: {stats.detection_rate:.1%}")
    print(f"Результат: {output_video}")


def format_tracking_table(
    results: Sequence[tuple[str | Path, ProcessingStats]],
) -> str:
    """Формирует Markdown-таблицу с результатами трекинга."""
    lines = [
        "| Видео | Кадров | По эталону | По предыдущему состоянию | Доля обнаружений |",
        "|---|---:|---:|---:|---:|",
    ]

    for input_video, stats in results:
        name = Path(input_video).name

        if name == "video.avi":
            name = "Собственное видео"

        lines.append(
            f"| {name} | {stats.processed_frames} | "
            f"{stats.reference_detections} | "
            f"{stats.previous_state_detections} | "
            f"{stats.detection_rate:.1%} |"
        )

    return "\n".join(lines)


def process_video(
    input_video: str | Path,
    feature_detector,
    matcher,
    detection_config: DetectionConfig,
) -> ProcessingStats:
    """Выполняет полный цикл обработки одного видео."""
    output_video = VIDEO_OUTPUT_DIR / f"tracked_{Path(input_video).stem}.mp4"
    capture = cv2.VideoCapture(str(input_video))
    writer = None
    stats = ProcessingStats()

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {input_video}")

        success, reference_frame = capture.read()

        if not success:
            raise RuntimeError(
                f"Не удалось прочитать первый кадр: {input_video}"
            )

        height, width = reference_frame.shape[:2]
        frame_size = (width, height)
        fps = capture.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            raise RuntimeError(
                f"Не удалось определить FPS видео: {input_video}"
            )

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(output_video), fourcc, fps, frame_size
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Не удалось создать выходное видео: {output_video}"
            )

        reference_features = extract_features(
            reference_frame, feature_detector
        )

        if reference_features.descriptors is None:
            raise RuntimeError(
                f"На эталоне не найдены дескрипторы: {input_video}"
            )

        reference_corners = np.float32([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]).reshape(-1, 1, 2)

        frame = reference_frame
        previous_features = None
        previous_corners = None

        while True:
            frame_features = extract_features(frame, feature_detector)
            box_color = REFERENCE_BOX_COLOR
            detection = detect_object(
                reference_features,
                reference_corners,
                frame_features,
                matcher,
                frame_size,
                detection_config,
            )

            if detection is not None:
                stats.reference_detections += 1
            elif (
                previous_features is not None
                and previous_corners is not None
            ):
                detection = detect_object(
                    previous_features,
                    previous_corners,
                    frame_features,
                    matcher,
                    frame_size,
                    detection_config,
                )

                if detection is not None:
                    stats.previous_state_detections += 1
                    box_color = PREVIOUS_BOX_COLOR

            if detection is not None:
                object_features = select_object_features(
                    frame_features,
                    detection.projected_corners,
                    detection_config.min_matches,
                )

                if object_features is not None:
                    previous_features = object_features
                    previous_corners = detection.projected_corners

                draw_detection(frame, detection, box_color)

            writer.write(frame)
            stats.processed_frames += 1
            success, frame = capture.read()

            if not success:
                break
    finally:
        capture.release()

        if writer is not None:
            writer.release()

    print_processing_stats(stats, output_video)
    return stats


def process_videos(input_videos: Sequence[str | Path]) -> None:
    """Последовательно запускает обработку набора видео."""
    feature_detector = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    detection_config = DetectionConfig()
    results = []

    for input_video in input_videos:
        print(f"\nОбработка видео: {input_video}")
        stats = process_video(
            input_video,
            feature_detector,
            matcher,
            detection_config,
        )
        results.append((input_video, stats))

    print(f"\n{format_tracking_table(results)}")


if __name__ == "__main__":
    process_videos(INPUT_VIDEOS)
