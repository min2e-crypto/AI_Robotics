"""평면 호모그래피 (영상 픽셀 ↔ 도로 좌표). 4점 대응으로 구한다. 순수 Python.

도로가 평면이면 카메라가 비스듬해도 바닥 위의 점은 정확히 옮겨진다 (ADR-015).
바닥에서 떠 있는 점(차량 지붕 등)은 카메라에서 먼 쪽으로 밀려 옮겨진다.
"""

from __future__ import annotations

Point = tuple[float, float]
Matrix = tuple[float, ...]  # 3×3을 행 순서로 편 9개 값 (마지막은 1)


def solve(src: list[Point], dst: list[Point]) -> Matrix:
    """src 4점을 dst 4점으로 옮기는 호모그래피."""
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, v])
    n = 8
    for col in range(n):  # 부분 피벗 가우스 소거
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            raise ValueError("호모그래피를 구할 수 없는 4점입니다 (세 점이 한 직선 위)")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col:
                k = rows[r][col] / rows[col][col]
                rows[r] = [a - k * b for a, b in zip(rows[r], rows[col])]
    return (*(rows[i][n] / rows[i][i] for i in range(n)), 1.0)


def apply(h: Matrix, p: Point) -> Point:
    x, y = p
    w = h[6] * x + h[7] * y + h[8]
    return (h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w
