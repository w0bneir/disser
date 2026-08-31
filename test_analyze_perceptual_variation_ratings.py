import copy
import unittest

from analyze_perceptual_variation_ratings import decode_ratings


def _fixture():
    mapping = {
        "A1": "exact_copy_A:reference", "A2": "exact_copy_B:reference",
        "L1": "synthetic_low_A:reference", "L2": "synthetic_low_B:synthetic_low",
        "M1": "synthetic_mid_A:reference", "M2": "synthetic_mid_B:synthetic_mid",
        "H1": "synthetic_high_A:reference", "H2": "synthetic_high_B:synthetic_high",
        "N1": "natural_ceiling_A:reference", "N2": "natural_ceiling_B:natural_donor",
        "R": "repeat_reference", "C": "synthetic_cycle",
    }
    key = {
        "protocol": "perceptual_variation_draft_v0", "blind_mapping": mapping,
        "direct_assignment": {"D1": "exact_copy", "D2": "synthetic_low", "D3": "synthetic_mid", "D4": "synthetic_high", "D5": "natural_ceiling"},
        "loop_truth": {"repeat": "R", "synthetic_cycle": "C"},
        "reference_name": "ref.wav", "natural_donor_name": "donor.wav",
    }
    def direct(a, b, difference="same", useful="not_applicable"):
        return {"blind_A": a, "blind_B": b, "same_or_different": difference, "same_event": "yes", "useful_difference": useful, "more_natural": "same", "artifacts": "neither", "confidence": "5", "comment": ""}
    document = {
        "protocol": "perceptual_variation_draft_v0", "session_id": "x", "saved_at": "now",
        "direct": {
            "D1": direct("A1", "A2"), "D2": direct("L1", "L2"), "D3": direct("M1", "M2"),
            "D4": direct("H2", "H1", "different", "slight"), "D5": direct("N1", "N2", "different", "slight"),
        },
        "loops": {"L01": {"blind_A": "C", "blind_B": "R", "less_repetitive": "A", "more_natural": "same", "artifacts": "neither", "preferred": "A", "confidence": "5", "comment": ""}},
    }
    manifest = {"candidate_metrics": {"high": {"profile_distance_from_reference": 3.0}}}
    return key, document, manifest


class PerceptualVariationRatingsTests(unittest.TestCase):
    def test_decodes_swaps_and_identifies_successful_high_dose(self):
        key, document, manifest = _fixture()
        report = decode_ratings(key, document, manifest)
        self.assertTrue(report["decision"]["control_passed"])
        self.assertEqual(report["decision"]["successful_synthetic_doses"], ["synthetic_high"])
        self.assertTrue(report["decision"]["synthetic_cycle_reduces_repetition"])
        self.assertTrue(report["decision"]["gate_passed"])

    def test_non_useful_detected_dose_fails_gate(self):
        key, document, manifest = _fixture()
        document = copy.deepcopy(document)
        document["direct"]["D4"]["useful_difference"] = "none"
        report = decode_ratings(key, document, manifest)
        self.assertEqual(report["decision"]["detected_synthetic_doses"], ["synthetic_high"])
        self.assertFalse(report["decision"]["gate_passed"])

    def test_rejects_tampered_blind_id(self):
        key, document, manifest = _fixture()
        document = copy.deepcopy(document)
        document["direct"]["D2"]["blind_A"] = "N1"
        with self.assertRaisesRegex(ValueError, "Blind IDs"):
            decode_ratings(key, document, manifest)


if __name__ == "__main__":
    unittest.main()
