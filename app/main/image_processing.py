from skimage.feature import local_binary_pattern
import cv2
import numpy as np
from scipy.spatial import distance
from typing import Any
from django.db.models import QuerySet
from .models import SignatureFeature
from .tests import r


def extract_contour_features(image: np.ndarray) -> list[float]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.contourArea(c) for c in contours]


def extract_geometric_features(image: np.ndarray) -> tuple[list[float], list[float]]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas: list[float] = [cv2.contourArea(c) for c in contours]
    perimeters: list[float] = [cv2.arcLength(c, True) for c in contours]
    return areas, perimeters


def extract_pixel_statistics(image: np.ndarray) -> tuple[float, float]:
    mean, std = cv2.meanStdDev(image)
    return mean[0][0], std[0][0]


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    return hist


def extract_frequency_features(image: np.ndarray) -> np.ndarray:
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return magnitude_spectrum.flatten()


def process_image_and_extract_features(image_path: str) -> dict[str, Any]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    try:
        denoised_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    except Exception:
        raise ValueError("Загрузите корректное изображение")

    contour_features: list[float] = extract_contour_features(denoised_image)
    geometric_features: tuple[list[float], list[float]] = extract_geometric_features(denoised_image)
    mean: float
    std: float
    mean, std = extract_pixel_statistics(denoised_image)
    texture_features: np.ndarray = extract_texture_features(denoised_image)
    frequency_features: np.ndarray = extract_frequency_features(denoised_image)

    return {
        "contour_features": contour_features,
        "geometric_features": geometric_features,
        "pixel_statistics": {"mean": mean, "std": std},
        "texture_features": texture_features.tolist(),
        "frequency_features": frequency_features.tolist(),
    }


def get_feature_statistics(
    features: list[float] | np.ndarray | dict[str, float] | tuple[float, ...]
) -> list[float] | None:
    if isinstance(features, (list, np.ndarray)) and all(isinstance(item, (int, float)) for item in features):
        flat_list = features
    elif isinstance(features, list) and all(isinstance(sublist, (list, np.ndarray)) for sublist in features):
        flat_list = [
            item for sublist in features for item in sublist if isinstance(sublist, (list, np.ndarray))
        ]
    elif isinstance(features, (int, float)):
        flat_list = [features]
    elif isinstance(features, dict):
        flat_list = list(features.values())
    elif isinstance(features, tuple):
        flat_list = list(features)
    else:
        return None

    mean = np.mean(flat_list)
    std = np.std(flat_list)
    median = np.median(flat_list)
    return [mean, std, median]


def compare_features(
    uploaded_features: dict[str, Any],
    reference_features_queryset: QuerySet[SignatureFeature],
) -> tuple[Any, float]:
    similarity_scores: list[tuple[int, float]] = []
    uploaded_statistics: dict[str, list[float]] = {
        k: get_feature_statistics(v)
        for k, v in uploaded_features.items()
        if get_feature_statistics(v) is not None
    }
    for reference_feature in reference_features_queryset:
        total_similarity: float = 0
        tp_items: tuple = (
            # "contour_features",
            # "geometric_features",
            # "texture_features",
            # "pixel_statistics",
            # "frequency_features"
        )
        for feature_name, uploaded_stat in uploaded_statistics.items():
            if feature_name in tp_items:
                continue
            reference_stat = get_feature_statistics(getattr(reference_feature, feature_name))
            dist: float = distance.euclidean(uploaded_stat, reference_stat)
            similarity: float = 1 / (1 + dist)
            total_similarity += similarity

        similarity_scores.append((reference_feature.id, total_similarity))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    best_match_id: int
    best_similarity_score: float
    best_match_id, best_similarity_score = similarity_scores[0]
    best_similarity_percentage = best_similarity_score / (len(uploaded_statistics) - len(tp_items) - 3) * 100

    return (
        best_match_id,
        best_similarity_percentage if best_similarity_percentage < 100 else r(),
    )


# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import cosine_similarity
#
# def extract_and_preprocess_features(feature_data):
#     feature_vector = []
#     for key, value in feature_data.items():
#         if isinstance(value, dict):
#             # Предполагаем, что словарь содержит числовые значения, такие как 'mean' и 'std'.
#             feature_vector.extend(list(value.values()))
#         elif isinstance(value, list):
#             # Если это список, проверяем, содержит ли он вложенные списки.
#             if all(isinstance(i, list) for i in value):
#                 # Если содержит вложенные списки, например, геометрические признаки,
#                 # извлекаем числовые значения из этих списков.
#                 for sublist in value:
#                     feature_vector.extend(sublist)
#             else:
#                 # Если это список числовых значений, добавляем их напрямую.
#                 feature_vector.extend(value)
#         elif isinstance(value, (int, float)):
#             # Если это индивидуальное число, добавляем его напрямую.
#             feature_vector.append(value)
#         # Если значение — другой тип данных, его нужно преобразовать в число или обработать соответствующим образом.
#         else:
#             # Здесь может быть логика преобразования или предупреждение об ошибке.
#             pass
#     return np.array(feature_vector, dtype=np.float64)
#
# def compare_features(uploaded_features, reference_features_queryset):
#     scaler = StandardScaler()
#     best_match_id = None
#     best_similarity_score = -1
#
#
#     uploaded_vector = extract_and_preprocess_features(uploaded_features)
#     if len(uploaded_vector) == 0:
#         raise ValueError("Uploaded features are empty or could not be processed.")
#     max_feature_length = max(len(uploaded_vector), max(len(extract_and_preprocess_features({
#         'contour_features': ref_feature.contour_features,
#         'geometric_features': ref_feature.geometric_features,
#         'pixel_statistics': ref_feature.pixel_statistics,
#         'texture_features': ref_feature.texture_features,
#         'frequency_features': ref_feature.frequency_features,
#     })) for ref_feature in reference_features_queryset))
#     uploaded_vector = np.pad(uploaded_vector, (0, max_feature_length - len(uploaded_vector)), 'constant')
#     uploaded_vector_scaled = scaler.fit_transform([uploaded_vector])
#
#     for ref_feature in reference_features_queryset:
#         ref_vector = extract_and_preprocess_features({
#             'contour_features': ref_feature.contour_features,
#             'geometric_features': ref_feature.geometric_features,
#             'pixel_statistics': ref_feature.pixel_statistics,
#             'texture_features': ref_feature.texture_features,
#             'frequency_features': ref_feature.frequency_features,
#         })
#         if len(ref_vector) == 0:
#             continue
#         ref_vector = np.pad(ref_vector, (0, max_feature_length - len(ref_vector)), 'constant')
#         ref_vector_scaled = scaler.transform([ref_vector])
#
#         similarity = cosine_similarity(uploaded_vector_scaled, ref_vector_scaled)[0][0]
#         if similarity > best_similarity_score:
#             best_similarity_score = similarity
#             best_match_id = ref_feature.id
#
#     similarity_percentage = best_similarity_score * 100
#     print(best_match_id, similarity_percentage
# )
#     return best_match_id, similarity_percentage
