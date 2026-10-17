"""Local copies of the robot and prop USD files the two tasks load.

The Drawer task needs the Franka and cabinet files of the Isaac 5.1 asset tree (the URLs in the
Isaac Lab task configuration do not resolve), so Drawer runs always load them from a local copy.
G1 runs can optionally do the same with the Isaac 6.0 G1 file.

    ./isaaclab.sh -p isaac/tasks/assets.py drawer <dir>
    ./isaaclab.sh -p isaac/tasks/assets.py g1 <dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import unquote, urljoin
from urllib.request import urlopen

SERVER = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/{}/Isaac/"
ASSETS = {
    "drawer": ("5.1", ["IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
                       "Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"]),
    "g1": ("6.0", ["IsaacLab/Robots/Unitree/G1/g1_minimal.usd"]),
}
# Material modules that the renderer provides, not files next to the assets.
RENDERER_MODULES = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}


def cache(task: str, root: str | Path) -> None:
    """Download the listed USD files of ``task`` and every file they reference into ``root``."""
    from pxr import UsdUtils

    version, files = ASSETS[task]
    base = SERVER.format(version)
    root = Path(root)
    pending, seen, rows = [urljoin(base, name) for name in files], set(), []
    while pending:
        url = pending.pop()
        if url in seen:
            continue
        if not url.startswith(base):
            raise ValueError(f"{url} lies outside {base}")
        relative = unquote(url[len(base):])
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with urlopen(url, timeout=60) as response:
                dest.write_bytes(response.read())
        content = dest.read_bytes()
        seen.add(url)
        rows.append(dict(path=relative, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        if dest.suffix.lower() in (".usd", ".usda", ".usdc"):
            sublayers, references, payloads = UsdUtils.ExtractExternalReferences(str(dest))
            pending += [urljoin(url, dep) for dep in (*sublayers, *references, *payloads)
                        if dep and dep not in RENDERER_MODULES]
    (root / "manifest.json").write_text(json.dumps(dict(base=base, files=rows), indent=2) + "\n")


def use_local_assets(env_cfg, task: str, root: str | Path | None) -> None:
    """Point the scene of ``task`` at a local asset copy made by :func:`cache`."""
    if root is None:
        if task == "drawer":
            raise ValueError("the Drawer task needs --assets <dir> (see isaac/tasks/assets.py)")
        return
    root = Path(root)
    if task == "drawer":
        robot, cabinet = ASSETS["drawer"][1]
        targets = ((env_cfg.scene.robot, robot), (env_cfg.scene.cabinet, cabinet))
    else:
        prefix = SERVER.format(ASSETS["g1"][0])
        source = env_cfg.scene.robot.spawn.usd_path
        if not source.startswith(prefix):
            raise ValueError(f"unexpected G1 asset {source}")
        targets = ((env_cfg.scene.robot, source[len(prefix):]),)
    for asset, relative in targets:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run isaac/tasks/assets.py {task} {root}")
        asset.spawn.usd_path = str(path)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ASSETS:
        raise SystemExit(f"usage: assets.py {{{','.join(ASSETS)}}} <dir>")
    cache(sys.argv[1], sys.argv[2])
