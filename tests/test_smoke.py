from pathlib import Path

import numpy as np
from PIL import Image

from lung_nematic.config import AnalysisConfig
from lung_nematic.pipeline import analyze_image


def test_pipeline_smoke(tmp_path: Path):
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    image[60:200, 60:200] = [220, 150, 200]
    image_path = tmp_path / "synthetic.png"
    Image.fromarray(image).save(image_path)

    config = AnalysisConfig(
        sigmas_px=(10.0, 15.0),
        min_scales_for_persistence=1,
        min_edge_distance_px=5,
    )
    metadata = {
        "filename": image_path.name,
        "image_id": "synthetic",
        "group": "test",
        "microns_per_pixel": None,
    }

    summary = analyze_image(
        image_path,
        metadata,
        tmp_path / "results",
        config,
    )

    assert summary["image_id"] == "synthetic"
    assert (tmp_path / "results" / "synthetic").exists()
