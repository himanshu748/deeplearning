import unittest

from src.config import Config


class ConfigValidationTest(unittest.TestCase):
    def test_config_validation_does_not_initialize_training_runtime(self):
        cfg = Config()

        self.assertIsNone(cfg.device)
        self.assertFalse(cfg.use_amp)

    def test_rejects_invalid_split_ratios(self):
        with self.assertRaisesRegex(ValueError, "train_ratio \\+ val_ratio"):
            Config(train_ratio=0.8, val_ratio=0.2)

    def test_rejects_invalid_loss_weights(self):
        with self.assertRaisesRegex(ValueError, "dice_weight \\+ bce_weight"):
            Config(dice_weight=0, bce_weight=0)

        with self.assertRaisesRegex(ValueError, "dice_weight"):
            Config(dice_weight=-0.1)

        with self.assertRaisesRegex(ValueError, "bce_weight"):
            Config(bce_weight=-0.1)


if __name__ == "__main__":
    unittest.main()
