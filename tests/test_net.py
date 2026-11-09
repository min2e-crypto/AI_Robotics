from fencebot.net import chain, hook_events, net_length

ORDER = [0, 1, 2, 3, 4]  # 보관, 기둥1·2·3, 당김
TOL = 30


def test_chain_skips_unhooked_posts():
    assert chain(ORDER, set()) == [0, 4]
    assert chain(ORDER, {3, 1}) == [0, 1, 3, 4]


def test_length_follows_chain():
    pts = {0: (0, 0), 1: (0, 5000), 2: (0, 5000), 3: (0, 5000), 4: (3000, 0)}
    assert net_length(ORDER, set(), pts) == 3000
    assert net_length(ORDER, {1}, pts) == 5000 + (5000**2 + 3000**2) ** 0.5


def base():
    return {0: (0, 0), 1: (500, -500), 2: (1000, -500), 3: (1500, -500), 4: (3000, 0)}


def test_post_crossing_net_line_hooks():
    prev, now = base(), base()
    prev[2], now[2] = (1000, -10), (1000, 10)  # 보관-당김 선(y=0)을 아래에서 위로
    assert hook_events(ORDER, set(), prev, now, TOL) == [(2, 0, 4, True)]


def test_post_touching_line_within_tolerance_hooks():
    prev, now = base(), base()
    prev[2], now[2] = (1000, -60), (1000, -20)
    assert hook_events(ORDER, set(), prev, now, TOL) == [(2, 0, 4, True)]


def test_passing_around_segment_end_does_not_hook():
    prev, now = base(), base()
    prev[3], now[3] = (3300, -10), (3300, 10)  # 당김 로봇(3000,0)보다 하류에서 위로
    assert hook_events(ORDER, set(), prev, now, TOL) == []


def test_sweeping_segment_over_standing_post_hooks():
    prev, now = base(), base()
    for p in (prev, now):
        p[1], p[3] = (500, -800), (1500, -800)  # 다른 기둥은 선에서 멀리
    prev[4], now[4] = (3000, -490), (3000, -510)  # 당김 로봇이 내려가 선이 기둥2를 지남
    prev[0], now[0] = (0, -490), (0, -510)
    assert hook_events(ORDER, set(), prev, now, TOL) == [(2, 0, 4, True)]


def test_wrong_segment_is_tangle():
    prev, now = base(), base()
    hooked = {3}
    now[3] = prev[3] = (1500, 1000)          # 기둥3이 이미 걸려 있음: 0-3, 3-4 구간
    prev[1], now[1] = (2200, 480), (2200, 520)  # 기둥1이 기둥3-당김 구간을 가로지름
    events = hook_events(ORDER, hooked, prev, now, TOL)
    assert events == [(1, 3, 4, False)]
