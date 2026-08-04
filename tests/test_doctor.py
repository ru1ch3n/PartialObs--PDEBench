from __future__ import annotations

from pdeobs.doctor import checks_succeeded, format_checks, run_doctor


def test_local_doctor_returns_readable_checks() -> None:
    checks = run_doctor(cluster="local")
    rendered = format_checks(checks)
    assert "Python >= 3.10" in rendered
    assert isinstance(checks_succeeded(checks), bool)
