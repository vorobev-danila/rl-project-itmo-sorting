import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import lab3_audit as audit


ROOT = Path(__file__).resolve().parents[1]


class UnitConversionTest(unittest.TestCase):
    def test_physical_order_differs_from_numeric_order(self):
        obj = {"items": ["2kg of metal", "30g of glass", "4mg of rubber"],
               "proposed_order": ["4mg of rubber", "30g of glass", "2kg of metal"]}
        self.assertEqual(audit.proxy_verdict(obj), "CORRECT")
        self.assertEqual(audit.proxy_verdict(obj, numeric_only=True), "INCORRECT")
        obj["proposed_order"] = ["2kg of metal", "4mg of rubber", "30g of glass"]
        self.assertEqual(audit.proxy_verdict(obj), "INCORRECT")
        self.assertEqual(audit.proxy_verdict(obj, numeric_only=True), "CORRECT")

    def test_exact_decimal_conversion_and_rejected_ambiguity(self):
        self.assertEqual(audit.parse_item("0.1cm of metal")[3], 1)
        for items in (["1g of glass", "1m of rubber"], ["1g of glass", "1000mg of rubber"]):
            with self.assertRaises(ValueError):
                audit.describe(items)

    def test_missing_object_is_not_silently_sorted(self):
        with self.assertRaises(ValueError):
            audit.proxy_verdict({"items": ["1 apple", "2 pears"], "proposed_order": ["1 apple"]})


class PreparedFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.objects = audit.read_jsonl(ROOT / "audit/slice.jsonl")

    def test_distinct_tasks_and_no_embedded_answers(self):
        audit.validate_slice(self.objects)
        self.assertEqual(len({tuple(sorted(obj["items"])) for obj in self.objects}), 52)
        self.assertEqual(len(self.objects), 52)
        allowed = {"id", "group", "seed", "game", "fold", "items", "proposed_order"}
        for obj in self.objects:
            self.assertEqual(set(obj), allowed)
        source = json.loads((ROOT / "docs/preliminary_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit.build_slice(source["records"]), self.objects)

    def test_each_person_has_all_core_and_disjoint_second(self):
        rows = audit.read_csv(ROOT / "audit/annotations.csv")
        expected = audit.annotation_rows(self.objects)

        def check_assignments(actual, templates):
            # Labels and times may be filled independently before merging packets.
            assignment_fields = ("id", "annotator", "order")
            self.assertEqual(
                [tuple(row[field] for field in assignment_fields) for row in actual],
                [tuple(row[field] for field in assignment_fields) for row in templates],
            )
            for row in actual:
                with self.subTest(annotator=row["annotator"], object_id=row["id"]):
                    self.assertEqual(set(row), set(audit.ANNOTATION_FIELDS))
                    self.assertIn(row["instruction_version"], ("1.0", "1.1"))
                    if row["label"]:
                        self.assertIn(row["label"], audit.LABELS)
                        self.assertTrue(0 <= float(row["elapsed_seconds"]) < float("inf"))
                    else:
                        self.assertEqual(row["elapsed_seconds"], "")

        self.assertEqual(len(rows), 112)
        check_assignments(rows, expected)
        groups = {obj["id"]: obj["group"] for obj in self.objects}
        second_owners = Counter()
        for person in audit.ANNOTATORS:
            personal = audit.read_csv(ROOT / "annotation_packets" / f"{person}.csv")
            assigned = [row for row in expected if row["annotator"] == person]
            check_assignments(personal, assigned)
            self.assertEqual(Counter(groups[row["id"]] for row in personal), {"core": 20, "second": 8})
            second_owners.update(row["id"] for row in personal if groups[row["id"]] == "second")
        self.assertEqual(len(second_owners), 32)
        self.assertTrue(all(value == 1 for value in second_owners.values()))

    def test_prepare_refuses_to_overwrite_human_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audit").mkdir()
            path = root / "audit/annotations.csv"
            path.write_text("human work", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                audit.prepare(root)
            self.assertEqual(path.read_text(encoding="utf-8"), "human work")

    def test_release_blocks_empty_annotation_before_reading_private_data(self):
        with self.assertRaisesRegex(ValueError, "core judgments"):
            audit.release(ROOT, ROOT / "not-a-private-directory")
        self.assertFalse((ROOT / "audit/scorer.jsonl").exists())
        self.assertEqual(audit.read_csv(ROOT / "audit/probe_results.csv"), [])

    def test_minimum_human_protocol_and_invalid_time(self):
        # Synthetic judgments test the validator only; never written to submission files.
        rows = copy.deepcopy(audit.annotation_rows(self.objects))
        groups = {obj["id"]: obj["group"] for obj in self.objects}
        completed_second = Counter()
        for row in rows:
            if groups[row["id"]] == "core" or completed_second[row["annotator"]] < 4:
                row["label"], row["elapsed_seconds"] = "CORRECT", "12.5"
                if groups[row["id"]] == "second":
                    completed_second[row["annotator"]] += 1
        audit.validate_humans(self.objects, rows)
        rows[0]["elapsed_seconds"] = "nan"
        with self.assertRaises(ValueError):
            audit.validate_humans(self.objects, rows)


class ExecutionProtocolTest(unittest.TestCase):
    class FailureEnv:
        def __init__(self):
            self.calls = []

        def reset(self, **kwargs):
            self.reset_parameters = kwargs
            return "start", {"validActions": ["take 2 pears", "take 1 apple"]}

        def step(self, action):
            self.calls.append(action)
            if len(self.calls) == 1:
                return "taken", 0, False, {"validActions": ["put 2 pears in box"], "score": 0}
            if len(self.calls) == 2:
                return "failed", -1, True, {"validActions": [], "score": -1, "taskfailure": True}
            raise AssertionError("step called after done=True")

    def test_stops_at_first_done_and_disables_gold_path(self):
        env = self.FailureEnv()
        obj = {"id": "test", "seed": 0, "items": ["1 apple", "2 pears"],
               "proposed_order": ["2 pears", "1 apple"]}
        verdict, trace = audit.run_candidate(env, obj)
        self.assertEqual(verdict, "INCORRECT")
        self.assertEqual(len(trace), 2)
        self.assertEqual(env.calls, ["take 2 pears", "put 2 pears in box"])
        self.assertEqual(env.reset_parameters, {
            "seed": 0, "gameFold": "test", "gameName": "sorting", "generateGoldPath": False})

    def test_source_mismatch_aborts_before_actions(self):
        env = self.FailureEnv()
        obj = {"id": "test", "seed": 0, "items": ["99 apples"], "proposed_order": ["99 apples"]}
        with self.assertRaisesRegex(RuntimeError, "Source objects differ"):
            audit.run_candidate(env, obj)
        self.assertEqual(env.calls, [])


if __name__ == "__main__":
    unittest.main()
