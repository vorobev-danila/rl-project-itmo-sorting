"""Prepare Lab 3 human annotation; keep environment judgments outside the repository."""

import argparse
import csv
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parent
ANNOTATORS = ("A1", "A2", "A3", "A4")
ANNOTATION_FIELDS = ("id", "annotator", "label", "order", "instruction_version", "elapsed_seconds")
PROBE_FIELDS = ("probe", "n", "metric", "value", "interpretation")
LABELS = {"CORRECT", "INCORRECT"}
UNITS = {
    "mg": ("mass", 1), "g": ("mass", 1000), "kg": ("mass", 1000000),
    "mm": ("length", 1), "cm": ("length", 10), "m": ("length", 1000),
    "ml": ("volume", 1), "l": ("volume", 1000),
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_item(item):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(mg|kg|g|mm|cm|m|ml|l)?\s+(.+)", item)
    if not match:
        raise ValueError(f"Cannot parse object: {item!r}")
    number, unit, _ = match.groups()
    dimension, factor = UNITS[unit] if unit else ("count", 1)
    return Fraction(number), unit, dimension, Fraction(number) * factor


def describe(items):
    parsed = [parse_item(item) for item in items]
    dimensions = {x[2] for x in parsed}
    if len(dimensions) != 1:
        raise ValueError("Incomparable dimensions")
    values = [x[3] for x in parsed]
    if len(set(values)) != len(values):
        raise ValueError("Tied values require a separate protocol")
    units = {x[1] for x in parsed}
    kind = "unitless" if units == {None} else "single-unit" if len(units) == 1 else "mixed-units"
    return next(iter(dimensions)), kind


def proxy_verdict(obj, numeric_only=False):
    items, order = obj["items"], obj["proposed_order"]
    if Counter(items) != Counter(order):
        raise ValueError("Proposed order must contain every object exactly once")
    describe(items)
    values = [parse_item(item)[0 if numeric_only else 3] for item in order]
    return "CORRECT" if all(a <= b for a, b in zip(values, values[1:])) else "INCORRECT"


def build_slice(records):
    # Sample only source objects, never source scores or previously normalized values.
    buckets = defaultdict(list)
    signatures, seeds = set(), set()
    for record in sorted(records, key=lambda x: x["seed"]):
        seed = record["seed"]
        if not 0 <= seed < 100:
            continue
        items = list(record["items"])
        signature = tuple(sorted(items))
        if seed in seeds or signature in signatures:
            raise ValueError("Duplicate source task")
        seeds.add(seed)
        signatures.add(signature)
        dimension, kind = describe(items)
        buckets[(dimension, len(items), kind)].append((seed, items))
    rng = random.Random(2026)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < 52:
        before = len(selected)
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                if len(selected) == 52:
                    break
        if len(selected) == before:
            raise ValueError("Need at least 52 distinct comparable source tasks")
    # Enrich both groups with valid and invalid candidates; IDs contain no labels.
    valid_indices = set(rng.sample(range(20), 10) + rng.sample(range(20, 52), 16))
    result = []
    for index, (seed, items) in enumerate(selected):
        physical_order = sorted(items, key=lambda item: parse_item(item)[3])
        if index in valid_indices:
            candidate = physical_order
        else:
            candidate = sorted(items, key=lambda item: parse_item(item)[0])
            if candidate == physical_order:
                candidate = physical_order.copy()
                j = rng.randrange(len(candidate) - 1)
                candidate[j], candidate[j + 1] = candidate[j + 1], candidate[j]
        display_items = items.copy()
        rng.shuffle(display_items)
        result.append({
            "id": f"K02-S{index + 1:03d}",
            "group": "core" if index < 20 else "second",
            "seed": seed, "game": "sorting", "fold": "test",
            "items": display_items, "proposed_order": candidate,
        })
    return result


def annotation_rows(objects):
    core = [obj["id"] for obj in objects if obj["group"] == "core"]
    second = [obj["id"] for obj in objects if obj["group"] == "second"]
    # Match the supplied notebook's non-pairwise generator exactly.
    random.Random(2026).shuffle(second)
    rows = []
    for person in ANNOTATORS:
        for object_id in core:
            rows.append(dict(zip(ANNOTATION_FIELDS, (object_id, person, "", "NA", "1.0", ""))))
    for j, person in enumerate(ANNOTATORS):
        for object_id in second[j * 8:(j + 1) * 8]:
            rows.append(dict(zip(ANNOTATION_FIELDS, (object_id, person, "", "NA", "1.0", ""))))
    return rows


def prepare(root):
    audit, packets = root / "audit", root / "annotation_packets"
    protected = [audit / name for name in ("slice.jsonl", "annotations.csv", "baseline_predictions.csv", "probe_results.csv")]
    protected += [packets / (person + suffix) for person in ANNOTATORS for suffix in (".csv", ".md")]
    if any(path.exists() for path in protected):
        raise FileExistsError("Preparation files already exist; refusing to overwrite annotation work")
    source = json.loads((root / "docs/preliminary_audit.json").read_text(encoding="utf-8"))
    objects = build_slice(source["records"])
    audit.mkdir(exist_ok=True)
    packets.mkdir(exist_ok=True)
    write_jsonl(audit / "slice.jsonl", objects)
    rows = annotation_rows(objects)
    write_csv(audit / "annotations.csv", ANNOTATION_FIELDS, rows)
    write_csv(audit / "baseline_predictions.csv", ("id", "verdict"), [
        {"id": obj["id"], "verdict": proxy_verdict(obj, numeric_only=True)} for obj in objects
    ])
    # Probe outcomes are withheld with the scorer until all human annotation ends.
    write_csv(audit / "probe_results.csv", PROBE_FIELDS, [])
    by_id = {obj["id"]: obj for obj in objects}
    for person in ANNOTATORS:
        assigned = [row for row in rows if row["annotator"] == person]
        write_csv(packets / f"{person}.csv", ANNOTATION_FIELDS, assigned)
        lines = [f"# K02 {person} — объекты для независимой разметки", "",
                 "Читайте audit/instruction.md. Заполняйте только свой CSV; не открывайте ответы других участников.", "",
                 "Первые 10 объектов — пилот. Все 20 core обязательны; далее назначено 8 second.", ""]
        for row in assigned:
            obj = by_id[row["id"]]
            lines.extend([f"## {obj['id']} ({obj['group']})", "",
                          "Исходные объекты: " + "; ".join(obj["items"]), "",
                          "Предложенный порядок: " + " → ".join(obj["proposed_order"]), ""])
        (packets / f"{person}.md").write_text("\n".join(lines), encoding="utf-8")
    print("Prepared 52 objects, 112 empty annotation rows and four annotation packets.")


def validate_slice(objects):
    if len(objects) != 52 or len({obj["id"] for obj in objects}) != 52:
        raise ValueError("Need 52 unique IDs")
    if Counter(obj["group"] for obj in objects) != {"core": 20, "second": 32}:
        raise ValueError("Need 20 core and 32 second")
    if len({obj["seed"] for obj in objects}) != 52:
        raise ValueError("Tasks must use distinct seeds")
    for obj in objects:
        if obj["game"] != "sorting" or obj["fold"] != "test":
            raise ValueError("Unexpected environment configuration")
        proxy_verdict(obj)


def run_candidate(env, obj):
    observation, info = env.reset(seed=obj["seed"], gameFold="test", gameName="sorting", generateGoldPath=False)
    available = [action.removeprefix("take ") for action in info["validActions"] if action.startswith("take ")]
    if Counter(available) != Counter(obj["items"]):
        raise RuntimeError(f"Source objects differ from live environment for {obj['id']}")
    trace = []
    for item in obj["proposed_order"]:
        for action in (f"take {item}", f"put {item} in box"):
            if action not in info["validActions"]:
                raise RuntimeError(f"Invalid action for {obj['id']}: {action}")
            observation, reward, done, info = env.step(action)
            trace.append({
                "id": obj["id"], "seed": obj["seed"], "step": len(trace) + 1,
                "action": action, "reward": reward, "done": done, "score": info["score"],
                "taskfailure": info.get("taskfailure", False),
                "tasksuccess": info.get("tasksuccess", False), "observation": observation,
            })
            if done:
                if info["score"] == 1.0:
                    return "CORRECT", trace
                if info.get("taskfailure", False):
                    return "INCORRECT", trace
                raise RuntimeError("Termination without success/failure; not an evaluator verdict")
    raise RuntimeError("Candidate did not terminate; not an evaluator verdict")


def build_probes(objects, verdicts):
    rows = []
    for expected, name, metric, explanation in (
        ("INCORRECT", "false_acceptance", "FP / mathematically_incorrect", "Доля неверных порядков, принятых средой."),
        ("CORRECT", "false_rejection", "FN / mathematically_correct", "Доля правильных порядков, отклонённых средой."),
    ):
        selected = [obj for obj in objects if proxy_verdict(obj) == expected]
        errors = sum(verdicts[obj["id"]] != expected for obj in selected)
        rows.append({"probe": name, "n": len(selected), "metric": metric,
                     "value": errors / len(selected),
                     "interpretation": f"{errors}/{len(selected)}. {explanation} Эталон автоматический; ручная проверка ещё требуется."})
    mixed = [obj for obj in objects if describe(obj["items"])[1] == "mixed-units"]
    errors = sum(proxy_verdict(obj, numeric_only=True) != proxy_verdict(obj) for obj in mixed)
    rows.append({"probe": "numeric_only_mixed_units", "n": len(mixed), "metric": "baseline_disagreement_with_proxy",
                 "value": errors / len(mixed),
                 "interpretation": f"{errors}/{len(mixed)}. Ошибки numeric-only на смешанных единицах по автоматическому прокси; это не сравнение с людьми."})
    return rows


def score(root, private_dir):
    from importlib.metadata import version
    if version("textworld-express") != "1.1.0":
        raise RuntimeError("Install textworld-express==1.1.0")
    from textworld_express import TextWorldExpressEnv
    private_dir = private_dir.resolve()
    if private_dir.is_relative_to(root.resolve()):
        raise ValueError("Machine outputs must be stored outside the shared repository")
    private_dir.mkdir(parents=True, exist_ok=True)
    for name in ("scorer.jsonl", "probe_results.csv", "traces.csv", "provenance.json"):
        if (private_dir / name).exists():
            raise FileExistsError("Private outputs already exist; choose a new directory")
    objects = read_jsonl(root / "audit/slice.jsonl")
    validate_slice(objects)
    env = TextWorldExpressEnv(envStepLimit=20)
    verdicts, traces = {}, []
    try:
        for obj in objects:
            verdict, trace = run_candidate(env, obj)
            verdicts[obj["id"]] = verdict
            traces.extend(trace)
    finally:
        env.close()
        # Version 1.1.0 also closes from __del__. Wait for Java to exit so that
        # its repeated cleanup cannot race with a second write to process stdin.
        env._gateway.java_process.wait(timeout=10)
    write_jsonl(private_dir / "scorer.jsonl", [{"id": obj["id"], "verdict": verdicts[obj["id"]]} for obj in objects])
    write_csv(private_dir / "probe_results.csv", PROBE_FIELDS, build_probes(objects, verdicts))
    write_csv(private_dir / "traces.csv", tuple(traces[0]), traces)
    provenance = {
        "package": "textworld-express==1.1.0", "game": "sorting", "fold": "test",
        "envStepLimit": 20, "generateGoldPath": False, "objects": len(objects),
        "slice_sha256": hashlib.sha256((root / "audit/slice.jsonl").read_bytes()).hexdigest(),
        "stop_at_first_done": True, "human_annotation_status": "pending",
    }
    (private_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Live environment run complete. Outputs are stored only in the specified private directory.")


def merge(root):
    expected = annotation_rows(read_jsonl(root / "audit/slice.jsonl"))
    merged = {}
    for person in ANNOTATORS:
        rows = read_csv(root / "annotation_packets" / f"{person}.csv")
        expected_person = [row for row in expected if row["annotator"] == person]
        if len(rows) != len(expected_person):
            raise ValueError(f"Unexpected number of rows for {person}")
        for actual, template in zip(rows, expected_person):
            for field in ("id", "annotator", "order"):
                if actual[field] != template[field]:
                    raise ValueError(f"Changed assignment field: {person}/{field}")
            if actual["instruction_version"] not in ("1.0", "1.1"):
                raise ValueError("Unknown instruction version")
            merged[(actual["id"], person)] = actual
    rows = [merged[(row["id"], row["annotator"])] for row in expected]
    # Never silently discard labels already entered in the shared table.
    for old in read_csv(root / "audit/annotations.csv"):
        new = merged[(old["id"], old["annotator"])]
        if old["label"] and (old["label"], old["elapsed_seconds"]) != (new["label"], new["elapsed_seconds"]):
            raise ValueError("Shared table has conflicting judgments; resolve before merging")
    write_csv(root / "audit/annotations.csv", ANNOTATION_FIELDS, rows)
    print("Merged individual annotation CSVs into audit/annotations.csv.")


def validate_humans(objects, rows):
    expected = {(row["id"], row["annotator"]): row for row in annotation_rows(objects)}
    if len(rows) != len(expected) or len({(row["id"], row["annotator"]) for row in rows}) != len(expected):
        raise ValueError("Missing or duplicate annotation assignments")
    groups = {obj["id"]: obj["group"] for obj in objects}
    completed = Counter()
    for row in rows:
        key = (row["id"], row["annotator"])
        if key not in expected or row["order"] != "NA":
            raise ValueError("Unknown assignment or changed presentation order")
        if row["instruction_version"] not in ("1.0", "1.1"):
            raise ValueError("Unknown instruction version")
        if not row["label"]:
            if row["elapsed_seconds"]:
                raise ValueError("Time entered without a label")
            if groups[row["id"]] == "core":
                raise ValueError("All core judgments must be complete before releasing machine verdicts")
            continue
        if row["label"] not in LABELS:
            raise ValueError("Unknown human label")
        try:
            elapsed = float(row["elapsed_seconds"])
        except ValueError:
            raise ValueError("Enter actual annotation time in seconds") from None
        if not 0 <= elapsed < float("inf"):
            raise ValueError("Invalid annotation time")
        completed[row["annotator"]] += 1
    if any(completed[person] < 24 for person in ANNOTATORS):
        raise ValueError("Each annotator needs 20 core and at least 4 second judgments")


def release(root, private_dir):
    objects = read_jsonl(root / "audit/slice.jsonl")
    validate_slice(objects)
    validate_humans(objects, read_csv(root / "audit/annotations.csv"))
    private_dir = private_dir.resolve()
    provenance = json.loads((private_dir / "provenance.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / "audit/slice.jsonl").read_bytes()).hexdigest()
    if provenance["slice_sha256"] != digest:
        raise ValueError("Machine outputs were generated for a different slice")
    machine = read_jsonl(private_dir / "scorer.jsonl")
    if len(machine) != 52 or {row["id"] for row in machine} != {obj["id"] for obj in objects}:
        raise ValueError("Incomplete machine predictions")
    if any(row["verdict"] not in LABELS for row in machine):
        raise ValueError("Unknown machine verdict")
    probes = read_csv(private_dir / "probe_results.csv")
    if len(probes) < 2:
        raise ValueError("Need at least two probes")
    # Copy only the two required submission files, not private action traces.
    write_jsonl(root / "audit/scorer.jsonl", machine)
    write_csv(root / "audit/probe_results.csv", PROBE_FIELDS, probes)
    print("Released scorer and probe results after checking completed human annotation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score", "merge", "release"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--private-dir", type=Path)
    args = parser.parse_args()
    if args.command in ("score", "release") and args.private_dir is None:
        parser.error("--private-dir is required for score/release")
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "score":
        score(args.root, args.private_dir)
    elif args.command == "merge":
        merge(args.root)
    else:
        release(args.root, args.private_dir)


if __name__ == "__main__":
    main()
