# Простейший алгоритм трекинга объекта на видео на основе ключевых точек.

[Текст задания](./docs/task.md)

## Теоретическая база

- SIFT --- Scale-Invariant Feature Transform, преобразование признаков, инвариантных к масштабу.
- KNN --- K-Nearest Neighbors, K ближайших соседей.
- LRT --- Lowe Ratio Test, проверка отношения расстояний Лоу.
- RANSAC --- Random Sample Consensus, метод случайных выборок для поиска согласованной модели.
- BFMatcher --- Brute-Force Matcher, сопоставление полным перебором.

Трекинг объекта основан на сопоставлении характерных точек изображения между кадрами. 
SIFT позволяет находить признаки, устойчивые к изменению масштаба и повороту. 
По соответствующим точкам с помощью RANSAC вычисляется матрица гомографии, которая позволяет определить положение плоского объекта на текущем кадре.

- SIFT использует многоэтапное попарное вычитание изображений, размытых по Гауссу разными фильтрами, для получения экстремумов (потенциальных характерных точек).
- KNN находит ближайшие точки двух изобажений на основании их дескрипторов с помощью некоторой метрики расстояния (например, евклидово расстояние).
- LRT сравнивает два расстояния с некоторой константой. Точка принимается, если разница расстояний значительна.
- RANSAC многократно берет n случайных соответствия для вычисления гомографии, затем проверяет, сколько точек из общего числа согласуется с этой моделью.

## Описание разработанной системы (алгоритмы, принципы работы, архитектура)

В системе выстроен следующий пайплайн обработки видео:

<details>
    <summary>SIFT --- выделение признаков</summary>
    SIFT.detectAndCompute() -> ключевые точки (KeyPoint) и их 128-мерные дескрипторы (ndarray).
</details>
<details>
    <summary>KNN + Lowe Ratio Test --- поиск соответствий</summary>
    knnMatch() -> для каждого дескриптора источника список из двух ближайших совпадений (DMatch).
    Сравнение осуществляется сначала с эталоном, затем с предыдущим состоянием.
</details>
<details>
    <summary>RANSAC + Homography</summary>
    findHomography() -> матрица гомографии 3*3 и маска инлайеров.
</details>
<details>
    <summary>Геометрические проверки</summary>
    Проверяются количество и доля инлайеров, конечность координат углов, выпуклость рамки, её площадь и допустимое отклонение от границ кадра.
</details>

В качестве эталона используется первый кадр видео. 
На каждом следующем кадре объект сначала ищется по эталону, а при неудаче --- по признакам последнего успешного обнаружения. 
После обнаружения обновляются признаки объекта и координаты его рамки.

## Результаты работы и тестирования системы

<figure>
  <img src="./docs/images/keypoints.jpg" alt="Описание фото для поисковиков">
  <figcaption>Ключевые точки</figcaption>
</figure>
<figure>
  <img src="./docs/images/matches.jpg" alt="Описание фото для поисковиков">
  <figcaption>Соответствия</figcaption>
</figure>
<figure>
  <img src="./docs/images/ransac_matches.jpg" alt="Описание фото для поисковиков">
  <figcaption>Инлайеры RANSAC</figcaption>
</figure>
<figure>
  <img src="./docs/images/tracked_frame.jpg" alt="Описание фото для поисковиков">
  <figcaption>Рамка объекта</figcaption>
</figure>

<!-- TRACKING TABLE START -->

| Видео                           | Кадров | По эталону | По предыдущему состоянию | Доля обнаружений |
|---------------------------------|-------:|-----------:|-------------------------:|-----------------:|
| mona-lisa.avi                   |    558 |        231 |                      327 |           100.0% |
| mona-lisa-blur.avi              |    558 |        212 |                      327 |            96.6% |
| mona-lisa-blur-extra-credit.avi |    558 |        210 |                      316 |            94.3% |
| video.MOV                       |    488 |        171 |                      298 |            96.1% |

<!-- TRACKING TABLE END -->

| mona-lisa.avi                            | mona-lisa-blur.avi                            |
|------------------------------------------|-----------------------------------------------|
| ![](./docs/images/tracked_mona-lisa.gif) | ![](./docs/images/tracked_mona-lisa-blur.gif) |

| mona-lisa-blur-extra-credit.avi                            | video.mov                            |
|------------------------------------------------------------|--------------------------------------|
| ![](./docs/images/tracked_mona-lisa-blur-extra-credit.gif) | ![](./docs/images/tracked_video.gif) |

## Выводы по работе

1. Реализован алгоритм трекинга плоского объекта с использованием SIFT, BFMatcher, Lowe Ratio Test и RANSAC.
2. Использование признаков последнего успешного обнаружения позволило добиться почти абсолютного распознавания для выбранных видео.
3. На собственном видео объект обнаружен в 96,1% кадров. Алгоритм показал хорошие результаты на всех четырёх видео, хотя в сложных условиях возможна потеря объекта.

## Использованные источники

1. OpenCV. Introduction to SIFT (Scale-Invariant Feature Transform). --- [Документация OpenCV](https://docs.opencv.org/5.0/py_tutorials/py_features/py_sift_intro/py_sift_intro.html#sift-in-opencv).
2. OpenCV. Feature Matching (BFMatcher, KNN, Lowe Ratio Test). --- [Документация OpenCV](https://docs.opencv.org/5.0/py_tutorials/py_features/py_matcher/py_matcher.html#basics-of-brute-force-matcher).
3. OpenCV. Feature Matching + Homography to find Objects (RANSAC, findHomography). --- [Документация OpenCV](https://docs.opencv.org/5.0/py_tutorials/py_features/py_feature_homography/py_feature_homography.html#code).
