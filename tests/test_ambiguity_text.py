"""The free text ambiguity detector: features, calib-only training, no test rows."""
import numpy as np

from budget_guard import ambiguity_text as at


def test_hand_features_shape_and_values():
    X = at.hand_features(["Who is the current coach of Arsenal?", "how many seats are in the 2nd row in 1999"])
    assert X.shape == (2, len(at.HAND_NAMES))
    f = dict(zip(at.HAND_NAMES, X[0]))
    assert f["wh_who"] == 1 and f["time_sensitive"] == 1 and f["n_caps"] == 1 and f["has_year"] == 0
    g = dict(zip(at.HAND_NAMES, X[1]))
    assert g["how_many"] == 1 and g["wh_how"] == 1 and g["has_year"] == 1 and g["has_ordinal"] == 1


def test_detector_versions_fit_and_predict():
    rng = np.random.default_rng(0)
    qs = [("when did " if i % 2 else "where is ") + f"thing {i}" for i in range(200)]
    y = np.array([i % 2 == 1 for i in range(200)])
    emb = rng.normal(size=(200, 8))
    for v in ("hand", "minilm", "combined"):
        det = at.TextAmbiguityDetector(v).fit(qs, emb, y)
        p = det.predict(qs, emb)
        assert p.shape == (200,) and ((p >= 0) & (p <= 1)).all()
        assert det.out_of_fold(qs, emb, y).shape == (200,)
    # "when" questions are the ambiguous ones here, the hand version must pick that up
    assert at.TextAmbiguityDetector("hand").fit(qs, None, y).predict(qs).round().tolist() == y.tolist()


def test_cli_never_embeds_test(world, tmp_path):
    import json
    import shutil

    from budget_guard.common import iter_jsonl

    d = tmp_path / "w"
    shutil.copytree(world / "splits", d / "splits")
    df = at.run(d, tmp_path / "out", backend="hash", n_boot=100)
    assert set(df["version"]) == {"hand", "minilm", "combined"}
    cached = set(json.loads((d / "features" / "emb_hash_qids.json").read_text()))
    test_qids = {r["qid"] for r in iter_jsonl(d / "splits" / "test.jsonl")}
    assert cached and not (cached & test_qids)
