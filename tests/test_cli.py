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

    def test_resolve_dataset_path_expands_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = train.resolve_dataset_path(tmp)

        self.assertEqual(resolved, Path(tmp).resolve())

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
