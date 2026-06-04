import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset import find_data_dirs, get_mask_path, load_pairs, validate_pair_files


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

    def test_validate_pair_files_accepts_readable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            mask = root / "mask.png"
            write_png(image)
            write_png(mask)

            validate_pair_files([(image, mask)])

    def test_validate_pair_files_rejects_empty_paired_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            mask = root / "mask.png"
            image.write_bytes(b"image")
            mask.touch()

            with self.assertRaisesRegex(ValueError, "mask file is empty"):
                validate_pair_files([(image, mask)])

    def test_validate_pair_files_rejects_corrupt_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            mask = root / "mask.png"
            image.write_bytes(b"not an image")
            write_png(mask)

            with self.assertRaisesRegex(ValueError, "not a readable image"):
                validate_pair_files([(image, mask)])

    def test_validate_pair_files_rejects_mismatched_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            mask = root / "mask.png"
            write_png(image, size=(4, 4))
            write_png(mask, size=(2, 4))

            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                validate_pair_files([(image, mask)])


def write_png(path: Path, size: tuple[int, int] = (2, 2)) -> None:
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


if __name__ == "__main__":
    unittest.main()
