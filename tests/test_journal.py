from smart_signal.journal import append_signal, signal_history


def test_append_and_read_signal(tmp_path, monkeypatch):
    monkeypatch.setenv("SMART_SIGNAL_CHECKPOINT", str(tmp_path / "none.pt"))
    from smart_signal import config

    monkeypatch.setattr(config, "artifacts_dir", lambda: tmp_path)
    from smart_signal import journal

    monkeypatch.setattr(journal, "artifacts_dir", lambda: tmp_path)
    rec = append_signal(
        {
            "signal": "HOLD",
            "price": 4430.29,
            "confidence": 0.55,
            "as_of": "2026-09-04T20:45:00+00:00",
            "p_buy": 0.22,
            "p_hold": 0.55,
            "p_sell": 0.23,
        }
    )
    assert rec["logged_at"]
    hist = signal_history(5)
    assert hist
    assert hist[0]["signal"] == "HOLD"
