import copy
import unittest

from analyze_take_discriminability_ratings import decode_gate


def _fixture():
    key = {
        "blind_mapping": {
            "S1": "direct_same_control_A", "S2": "direct_same_control_B",
            "N1": "direct_near_pair_A", "N2": "direct_near_pair_B",
            "M1": "direct_median_pair_A", "M2": "direct_median_pair_B",
            "F1": "direct_far_pair_A", "F2": "direct_far_pair_B",
            "R": "loop_repeat_raw", "A": "loop_alternate_far_raw",
            "Q": "loop_shuffle_raw", "C": "loop_shuffle_clip_matched",
        },
        "direct_truth": {
            name: {"sources": [name + "a", name + "b"], "distance": distance}
            for name, distance in (("same_control", 0.0), ("near_pair", 0.7), ("median_pair", 1.0), ("far_pair", 1.4))
        },
        "direct_comparisons": {"D1": "same_control", "D2": "near_pair", "D3": "median_pair", "D4": "far_pair"},
        "loop_asset_codes": {"repeat_raw": "R", "alternate_far_raw": "A", "shuffle_raw": "Q", "shuffle_clip_matched": "C"},
        "loop_comparisons": {"L1": ["repeat_raw", "shuffle_raw"], "L2": ["shuffle_raw", "shuffle_clip_matched"], "L3": ["repeat_raw", "alternate_far_raw"]},
    }
    def direct(a, b, result):
        return {"blind_A": a, "blind_B": b, "same_or_different": result, "useful_difference": "none" if result == "different" else "not_applicable", "same_event": "yes", "confidence": "5", "comment": ""}
    def loop(a, b, winner):
        return {"blind_A": a, "blind_B": b, "less_repetitive": winner, "clearer_variation": winner, "more_natural": "same", "preferred": winner, "confidence": "5", "comment": ""}
    document = {
        "protocol": "take_discriminability_gate_v1", "session_id": "test", "saved_at": "now",
        "direct": {"D1": direct("S1", "S2", "same"), "D2": direct("N1", "N2", "different"), "D3": direct("M1", "M2", "same"), "D4": direct("F1", "F2", "different")},
        "loops": {
            "L1": loop("Q", "R", "A"),
            "L2": {**loop("C", "Q", "B"), "less_repetitive": "same", "preferred": "same"},
            "L3": loop("R", "A", "B"),
        },
    }
    return key, document


class TakeDiscriminabilityRatingsTests(unittest.TestCase):
    def test_decodes_client_side_swaps_and_passes_gate(self):
        key, document = _fixture()
        report = decode_gate(key, document)
        self.assertTrue(report["decision"]["control_passed"])
        self.assertTrue(report["decision"]["raw_shuffle_beats_repeat"])
        self.assertTrue(report["decision"]["alternate_far_beats_repeat"])
        self.assertTrue(report["decision"]["clip_matching_reduces_clear_variation"])
        self.assertTrue(report["decision"]["gate_passed"])
        self.assertTrue(report["decision"]["acoustic_distance_ranking_is_not_perceptually_monotonic"])

    def test_rejects_tampered_blind_ids(self):
        key, document = _fixture()
        document = copy.deepcopy(document)
        document["loops"]["L1"]["blind_A"] = "F1"
        with self.assertRaisesRegex(ValueError, "Blind IDs"):
            decode_gate(key, document)

    def test_rejects_partial_answer(self):
        key, document = _fixture()
        document = copy.deepcopy(document)
        del document["direct"]["D2"]["same_event"]
        with self.assertRaisesRegex(ValueError, "same_event"):
            decode_gate(key, document)


if __name__ == "__main__":
    unittest.main()
