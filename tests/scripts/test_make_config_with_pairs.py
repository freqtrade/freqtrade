import json

from scripts.make_config_with_pairs import main


def test_make_config_with_pairs_replaces_whitelist(tmp_path, monkeypatch) -> None:
    base_config = {
        "exchange": {"pair_whitelist": ["OLD/INR"], "pair_blacklist": []}
    }
    pairs = ["NEW/INR", "OTHER/INR"]

    base_path = tmp_path / "base.json"
    pairs_path = tmp_path / "pairs.json"
    out_path = tmp_path / "out.json"

    base_path.write_text(json.dumps(base_config))
    pairs_path.write_text(json.dumps(pairs))

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_config_with_pairs",
            "--base",
            str(base_path),
            "--pairs",
            str(pairs_path),
            "--out",
            str(out_path),
        ],
    )

    main()

    derived = json.loads(out_path.read_text())
    assert derived["exchange"]["pair_whitelist"] == pairs
