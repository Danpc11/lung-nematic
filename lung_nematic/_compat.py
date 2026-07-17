"""Version-portable wrappers for scikit-image morphology.

scikit-image 0.26 renamed the size parameters of ``remove_small_objects``
(``min_size`` -> ``max_size``) and ``remove_small_holes``
(``area_threshold`` -> ``max_size``). Older releases (including the one shipped
with Google Colab) do not accept ``max_size`` and raise ``TypeError``. These
wrappers prefer the new keyword and fall back to the legacy one, so the package
works across scikit-image >= 0.19 without deprecation warnings.
"""

from skimage import morphology


def remove_small_objects(mask, size):
    try:
        return morphology.remove_small_objects(mask, max_size=size)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=size)


def remove_small_holes(mask, size):
    try:
        return morphology.remove_small_holes(mask, max_size=size)
    except TypeError:
        return morphology.remove_small_holes(mask, area_threshold=size)
