import pytest

from fencebot.homography import apply, solve

SRC = [(60, 1000), (1860, 1000), (1560, 250), (360, 250)]
DST = [(0, -1000), (9000, -1000), (9000, 2333), (0, 2333)]


def test_four_points_map_exactly_and_back():
    h, inv = solve(SRC, DST), solve(DST, SRC)
    for s, d in zip(SRC, DST):
        assert apply(h, s) == pytest.approx(d, abs=1e-6)
        assert apply(inv, d) == pytest.approx(s, abs=1e-6)
    p = (900.0, 640.0)
    assert apply(inv, apply(h, p)) == pytest.approx(p, abs=1e-6)


def test_perspective_is_not_affine():
    h = solve(SRC, DST)
    # 화면 가운데 세로선은 도로 가운데지만, 화면 높이의 가운데는 도로 폭의 가운데가 아니다
    x, y = apply(h, (960, 625))
    assert x == pytest.approx(4500, abs=1e-6) and y < (2333 - 1000) / 2
