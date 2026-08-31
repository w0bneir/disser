from __future__ import annotations

import unittest

from analyze_natural_pool_ratings import decode_ratings


class NaturalPoolRatingsAnalysisTests(unittest.TestCase):
    def make_key(self) -> dict:
        return {
            "pairwise_key": {
                "Q01": {
                    "hypothesis": "H1_scheduler_vs_shuffle",
                    "A_method": "perceptual_full",
                    "B_method": "shuffle_full",
                    "A_blind_id": "P03",
                    "B_blind_id": "P01",
                }
            }
        }

    def make_document(self) -> dict:
        return {
            "protocol": "natural_pool_pairwise_v1",
            "session_id": "session-001",
            "pairs": {
                "Q01": {
                    "blind_A": "P03",
                    "blind_B": "P01",
                    "less_repetitive": "A",
                    "more_natural": "same",
                    "more_consistent": "B",
                    "preferred": "A",
                    "confidence": "4",
                    "comment": "",
                }
            },
        }

    def test_decodes_blind_sides_into_methods(self) -> None:
        report = decode_ratings(self.make_key(), [self.make_document()])
        self.assertEqual(report["valid_documents"], 1)
        self.assertEqual(report["warnings"], [])
        by_criterion = {item["criterion"]: item for item in report["comparisons"]}
        repetition = by_criterion["less_repetitive"]
        self.assertEqual(repetition["method_1"], "perceptual_full")
        self.assertEqual(repetition["method_1_wins"], 1)
        self.assertEqual(repetition["method_2_wins"], 0)
        naturalness = by_criterion["more_natural"]
        self.assertEqual(naturalness["ties"], 1)
        consistency = by_criterion["more_consistent"]
        self.assertEqual(consistency["method_2_wins"], 1)
        self.assertEqual(consistency["mean_confidence"], 4.0)

    def test_rejects_wrong_blind_ids_without_decoding_answer(self) -> None:
        document = self.make_document()
        document["pairs"]["Q01"]["blind_A"] = "P99"
        report = decode_ratings(self.make_key(), [document])
        self.assertEqual(report["valid_documents"], 0)
        self.assertEqual(report["invalid_documents"], 1)
        self.assertTrue(report["warnings"])
        self.assertEqual(report["comparisons"], [])

    def test_accepts_client_side_ab_swap_and_decodes_actual_side(self) -> None:
        document = self.make_document()
        row = document["pairs"]["Q01"]
        row["blind_A"], row["blind_B"] = row["blind_B"], row["blind_A"]
        row["less_repetitive"] = "A"
        report = decode_ratings(self.make_key(), [document])
        comparison = next(
            item for item in report["comparisons"] if item["criterion"] == "less_repetitive"
        )
        self.assertEqual(comparison["method_1"], "perceptual_full")
        self.assertEqual(comparison["method_1_wins"], 0)
        self.assertEqual(comparison["method_2_wins"], 1)

    def test_partial_document_is_not_aggregated(self) -> None:
        document = self.make_document()
        document["pairs"]["Q01"]["preferred"] = ""
        report = decode_ratings(self.make_key(), [document])
        self.assertEqual(report["valid_documents"], 0)
        self.assertEqual(report["partial_documents"], 1)
        self.assertEqual(report["comparisons"], [])

    def test_duplicate_session_is_counted_once(self) -> None:
        first = self.make_document()
        second = self.make_document()
        second["saved_at"] = "later"
        report = decode_ratings(self.make_key(), [first, second])
        self.assertEqual(report["valid_documents"], 1)
        self.assertEqual(report["duplicate_documents"], 1)

    def test_corrected_export_after_partial_same_session_is_accepted(self) -> None:
        partial = self.make_document()
        partial["pairs"]["Q01"]["preferred"] = ""
        corrected = self.make_document()
        report = decode_ratings(self.make_key(), [partial, corrected])
        self.assertEqual(report["partial_documents"], 1)
        self.assertEqual(report["valid_documents"], 1)
        self.assertEqual(report["duplicate_documents"], 0)

    def test_unknown_protocol_is_not_counted(self) -> None:
        document = self.make_document()
        document["protocol"] = "unknown"
        report = decode_ratings(self.make_key(), [document])
        self.assertEqual(report["valid_documents"], 0)
        self.assertEqual(report["invalid_documents"], 1)
        self.assertTrue(report["warnings"])


if __name__ == "__main__":
    unittest.main()
