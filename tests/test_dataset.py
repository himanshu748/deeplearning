import tempfile
import unittest
from pathlib import Path

from src.dataset import find_data_dirs, get_mask_path, load_pairs


class DatasetDiscoveryTest(unittest.TestCase):
    def test_finds_explicit_image_and_mask_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()

            self.assertEqual(find_data_dirs(root), (images, masks))

    def test_pairs_common_mask_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            image = images / "plot_sat_01.png"
            mask = masks / "plot_mask_01.tiff"
            image.touch()
            mask.touch()

            self.assertEqual(get_mask_path(image, masks), mask)
            self.assertEqual(load_pairs(root), [(image, mask)])

    def test_load_pairs_rejects_unmatched_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()
            (images / "plot.png").touch()

            with self.assertRaisesRegex(FileNotFoundError, "No valid image-mask pairs"):
                load_pairs(root)


if __name__ == "__main__":
    unittest.main()
