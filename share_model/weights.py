"""모델 가중치(.pt)를 GitHub Releases로 주고받는다.

가중치는 용량이 커서 저장소에 올리지 않는다. 대신 릴리스에 첨부하고, 어느 릴리스의 어느 파일을 쓰는지만
models/weights.json에 적어 저장소에 둔다.

    python weights.py fetch                 # weights.json에 적힌 가중치를 받아 models/ 아래에 채움
    python weights.py check                 # 지금 가진 가중치가 weights.json과 같은지 확인
    python weights.py publish models-v2     # (모델 담당) 지금 가진 가중치를 새 릴리스로 올리고 weights.json을 고침
    python weights.py publish models-v2 --dry-run   # 올리지 않고 무엇이 올라갈지만 봄

fetch와 check는 아무 준비 없이 된다 (공개 저장소의 릴리스 주소에서 바로 받음).
publish만 GitHub CLI(gh)와 저장소 쓰기 권한이 필요하다. `gh auth login`을 먼저 해 둔다.
표준 라이브러리만 쓰므로 모델 실행 환경(ultralytics)이 없어도 돈다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
MANIFEST = MODELS / "weights.json"

Download = Callable[[str, str, str, Path], None]  # (저장소 'owner/name', 릴리스 태그, 첨부 파일 이름, 받을 폴더)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def asset_name(rel: str) -> str:
    """릴리스 첨부 파일 이름. 릴리스에는 폴더가 없고 best.pt가 여럿이라 폴더 이름을 앞에 붙인다."""
    return rel.replace("/", "-")


def scan(models: Path) -> dict[str, str]:
    """models 아래의 모든 .pt와 그 해시. 키는 models 기준 경로."""
    return {p.relative_to(models).as_posix(): sha256(p) for p in sorted(models.rglob("*.pt"))}


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다. 모델 담당이 `python weights.py publish <태그>`로 먼저 만들어야 합니다.")
    return json.loads(path.read_text(encoding="utf-8"))


def status(manifest: dict, models: Path) -> dict[str, str]:
    """파일마다 'ok'(같음) / 'missing'(없음) / 'changed'(내용이 다름)."""
    out = {}
    for rel, digest in manifest["files"].items():
        p = models / rel
        out[rel] = "missing" if not p.exists() else "ok" if sha256(p) == digest else "changed"
    return out


def release_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def http_download(repo: str, tag: str, asset: str, dest: Path) -> None:
    """공개 저장소의 릴리스 첨부 파일을 로그인 없이 받는다."""
    url = release_url(repo, tag, asset)
    try:
        with urllib.request.urlopen(url) as r, (dest / asset).open("wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.URLError as e:
        raise SystemExit(f"{url} 을 받지 못했습니다: {e}")


def parse_repo(remote_url: str) -> str:
    """git 원격 주소에서 'owner/name'을 꺼낸다. https와 ssh 주소 모두 받는다."""
    m = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?/?$", remote_url.strip())
    if not m:
        raise SystemExit(f"GitHub 저장소 주소가 아닙니다: {remote_url!r}. `--repo owner/name`으로 알려 주세요.")
    return m.group(1)


def origin_repo() -> str:
    done = subprocess.run(["git", "remote", "get-url", "origin"], cwd=HERE, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit("git 원격(origin)이 없습니다. `--repo owner/name`으로 알려 주세요.")
    return parse_repo(done.stdout)


def fetch(manifest: dict, models: Path, download: Download = http_download) -> list[str]:
    """weights.json과 다른 파일만 받는다. 받은 파일의 해시가 다르면 기존 파일을 건드리지 않고 멈춘다."""
    todo = [rel for rel, state in status(manifest, models).items() if state != "ok"]
    with tempfile.TemporaryDirectory() as tmp:
        for rel in todo:
            download(manifest["repo"], manifest["release"], asset_name(rel), Path(tmp))
            got = Path(tmp) / asset_name(rel)
            if sha256(got) != manifest["files"][rel]:
                raise SystemExit(f"{rel}: 받은 파일이 weights.json의 해시와 다릅니다. 릴리스 {manifest['release']}를 확인하세요.")
            (models / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(got, models / rel)
    return todo


def publish(tag: str, models: Path, manifest_path: Path, repo: str, dry_run: bool = False) -> dict:
    """models 아래의 .pt 전부를 새 릴리스로 올리고, 성공하면 weights.json을 고친다."""
    files = scan(models)
    if not files:
        raise SystemExit(f"{models} 아래에 .pt 파일이 없습니다.")
    manifest = {"repo": repo, "release": tag, "files": files}
    if dry_run:
        return manifest
    with tempfile.TemporaryDirectory() as tmp:
        assets = []
        for rel in files:
            assets.append(str(shutil.copy2(models / rel, Path(tmp) / asset_name(rel))))
        _gh("release", "create", tag, *assets, "--repo", repo, "--title", tag, "--notes", "모델 가중치. share_model/models/weights.json 참고")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _gh(*args: str) -> None:
    if not shutil.which("gh"):
        raise SystemExit("GitHub CLI(gh)가 없습니다. https://cli.github.com 에서 설치하고 `gh auth login`을 하세요.")
    done = subprocess.run(["gh", *args], cwd=HERE)
    if done.returncode:
        raise SystemExit(f"gh {' '.join(args[:2])} 실패 (위 메시지 참고)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="모델 가중치를 GitHub Releases로 주고받는다")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="weights.json에 적힌 가중치를 받는다")
    sub.add_parser("check", help="지금 가진 가중치가 weights.json과 같은지 본다")
    pub = sub.add_parser("publish", help="지금 가진 가중치를 새 릴리스로 올린다 (모델 담당)")
    pub.add_argument("tag", help="새 릴리스 태그. 예: models-v2")
    pub.add_argument("--repo", help="GitHub 저장소 owner/name. 없으면 git 원격(origin)에서 알아냄")
    pub.add_argument("--dry-run", action="store_true", help="올리지 않고 무엇이 올라갈지만 본다")
    args = parser.parse_args(argv)

    if args.command == "publish":
        manifest = publish(args.tag, MODELS, MANIFEST, args.repo or origin_repo(), args.dry_run)
        for rel, digest in manifest["files"].items():
            print(f"{asset_name(rel):32} {digest[:12]}  ← models/{rel}")
        print(f"올리지 않았습니다 (--dry-run). 올릴 곳: {manifest['repo']} 릴리스 {args.tag}" if args.dry_run
              else f"{manifest['repo']} 릴리스 {args.tag}에 올렸습니다. 고쳐진 models/weights.json을 커밋하세요.")
        return
    manifest = load_manifest(MANIFEST)
    if args.command == "fetch":
        got = fetch(manifest, MODELS)
        print(f"릴리스 {manifest['release']}: " + (f"{len(got)}개 받음 ({', '.join(got)})" if got else "이미 최신"))
        return
    states = status(manifest, MODELS)
    for rel, state in states.items():
        print(f"{state:8} models/{rel}")
    extra = sorted(set(scan(MODELS)) - set(states))
    for rel in extra:
        print(f"{'extra':8} models/{rel}  (weights.json에 없음)")
    if any(state != "ok" for state in states.values()):
        sys.exit(f"릴리스 {manifest['release']}와 다릅니다. `python weights.py fetch`로 맞추세요.")
    print(f"릴리스 {manifest['release']}와 같습니다.")


if __name__ == "__main__":
    main()
