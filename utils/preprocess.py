from typing import Callable

import cv2


def preprocess(
    image_size: tuple[int, int],
    preprocess_func: Callable = None,
    normalize: bool = False
) -> Callable[[cv2.Mat], cv2.Mat]:
    def func(image: cv2.Mat) -> cv2.Mat:
        resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        preprocessed = preprocess_func(resized) if preprocess_func else resized
        return preprocessed if not normalize else preprocessed / 255

    return func
