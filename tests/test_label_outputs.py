import csv
import os
import tempfile
import unittest

import numpy as np

from label_outputs import materialize_row_labels
from metric import hungarian_match_labels
from run_directory_pipeline import split_round_robin


class RAVWorkflowTests(unittest.TestCase):
    def test_hungarian_alignment_recovers_permuted_labels(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        predictions = np.array([2, 2, 0, 0, 1, 1])

        aligned = hungarian_match_labels(labels, predictions)

        np.testing.assert_array_equal(aligned, labels)

    def test_existing_labels_are_copied_to_canonical_seed_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_dir = os.path.join(temp_dir, "run")
            os.makedirs(result_dir)
            source_path = os.path.join(result_dir, "labels.csv")
            with open(source_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["index", "label", "pred", "aligned_pred"]
                )
                writer.writeheader()
                writer.writerow({"index": 0, "label": 1, "pred": 0, "aligned_pred": 1})

            row = {
                "dataset": "sample",
                "seed": "42",
                "result_dir": result_dir,
            }
            materialize_row_labels(row, temp_dir)

            expected_path = os.path.join(temp_dir, "labels", "sample", "seed_42.csv")
            self.assertEqual(row["labels_path"], expected_path)
            with open(expected_path, newline="") as f:
                copied = list(csv.DictReader(f))
            self.assertEqual(copied[0]["aligned_pred"], "1")

    def test_directory_pipeline_balances_datasets_across_gpus(self):
        groups = split_round_robin(["a", "b", "c", "d", "e"], 2)

        self.assertEqual(groups, [["a", "c", "e"], ["b", "d"]])


if __name__ == "__main__":
    unittest.main()
