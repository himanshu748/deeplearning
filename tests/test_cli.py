import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import train
from src.config import Config
from src.models import build_model


class TrainCliTest(unittest.TestCase):
    def test_parser_accepts_dry_run(self):
        args = train.build_parser().parse_args(
            ["--data-dir", "/tmp/forest-data", "--dry-run", "--models", "unet"]
        )

        self.assertTrue(args.dry_run)
        self.assertEqual(args.models, ["unet"])

    def test_resolve_dataset_path_rejects_missing_local_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "Dataset path does not exist"):
            train.resolve_dataset_path("/tmp/definitely-missing-forest-dataset")

    def test_resolve_dataset_path_requires_explicit_download(self):
        with self.assertRaisesRegex(RuntimeError, "--download-data"):
            train.resolve_dataset_path(None)

    def test_summarize_dataset_returns_deterministic_split_counts(self):
        pairs = [(Path(f"image-{idx}.png"), Path(f"mask-{idx}.png")) for idx in range(10)]
        cfg = Config(seed=7)

        summary = train.summarize_dataset(pairs, cfg)

        self.assertEqual(
            summary,
            {
                "total_pairs": 10,
                "train_pairs": 7,
                "val_pairs": 2,
                "test_pairs": 1,
            },
        )

    def test_selected_models_defaults_to_all_model_choices(self):
        self.assertEqual(train.selected_models(None), list(train.MODEL_CHOICES))

    def test_dry_run_summary_reports_selected_models(self):
        pairs = [(Path(f"image-{idx}.png"), Path(f"mask-{idx}.png")) for idx in range(10)]
        cfg = Config(seed=7)

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            train.print_dry_run_summary(
                Path("/tmp/forest-data"),
                pairs,
                cfg,
                models=["unet"],
            )

        output = stdout.getvalue()
        self.assertIn("- models: unet", output)
        self.assertNotIn("deeplabv3plus", output)

    def test_split_manifest_records_relative_deterministic_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            pairs = [
                (
                    dataset / "images" / f"image-{idx}.png",
                    dataset / "masks" / f"mask-{idx}.png",
                )
                for idx in range(10)
            ]
            cfg = Config(seed=7)

            manifest = train.build_split_manifest(dataset, pairs, cfg, models=["unet"])

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["seed"], 7)
        self.assertEqual(manifest["models"], ["unet"])
        self.assertEqual(len(manifest["splits"]["train"]), 7)
        self.assertEqual(len(manifest["splits"]["val"]), 2)
        self.assertEqual(len(manifest["splits"]["test"]), 1)
        self.assertTrue(manifest["splits"]["train"][0]["image"].startswith("images/"))

    def test_write_split_manifest_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            output_path = root / "artifacts" / "splits.json"
            pairs = [
                (
                    dataset / "images" / f"image-{idx}.png",
                    dataset / "masks" / f"mask-{idx}.png",
                )
                for idx in range(3)
            ]

            train.write_split_manifest(
                output_path,
                dataset,
                pairs,
                Config(seed=1),
                ["unet"],
            )

            manifest = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(set(manifest["splits"]), {"train", "val", "test"})
        self.assertEqual(manifest["models"], ["unet"])

    def test_resolve_dataset_path_expands_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = train.resolve_dataset_path(tmp)

        self.assertEqual(resolved, Path(tmp).resolve())

    def test_download_data_uses_kagglehub_when_explicit(self):
        with patch("train.require_kagglehub") as require_kagglehub, patch("sys.stdout"):
            require_kagglehub.return_value.dataset_download.return_value = "/tmp/kaggle-dataset"

            resolved = train.resolve_dataset_path(None, download_data=True)

        self.assertEqual(resolved, Path("/tmp/kaggle-dataset"))
        require_kagglehub.return_value.dataset_download.assert_called_once_with(
            "quadeer15sh/augmented-forest-segmentation"
        )

    def test_cli_entry_returns_failure_without_traceback_for_bad_path(self):
        with patch("sys.stderr") as stderr:
            result = train.cli_entry(["--data-dir", "/tmp/definitely-missing-forest-dataset"])

        self.assertEqual(result, 1)
        stderr.write.assert_called()

    def test_invalid_model_name_does_not_require_training_stack(self):
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            build_model("not-a-model")


if __name__ == "__main__":
    unittest.main()
