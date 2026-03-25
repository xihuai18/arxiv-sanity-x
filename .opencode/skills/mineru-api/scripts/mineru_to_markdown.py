#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from getpass import getpass
from pathlib import Path, PurePosixPath
from urllib import error, request

BASE_URL = "https://mineru.net/api/v4"
POLL_STATES = {"waiting-file", "pending", "running", "converting"}
ASSET_KINDS = {"image", "table", "equation", "video", "audio"}
SKILL_ROOT = Path(__file__).resolve().parents[1]


class ApiError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: int | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call MinerU API and normalize the result into clean markdown + ordered assets.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=Path, help="Local PDF path. Uses batch upload flow.")
    src.add_argument("--url", help="Public PDF or document URL. Uses single-task URL flow.")
    src.add_argument("--zip", type=Path, help="Existing MinerU result ZIP. Skips API submission.")
    src.add_argument(
        "--raw-dir",
        type=Path,
        help="Existing extracted MinerU output directory. Skips API submission.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for cleaned markdown and assets.",
    )
    parser.add_argument("--token-file", type=Path, help="Read MinerU token from a local file.")
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Polling interval in seconds. Default: 3.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Polling timeout in seconds. Default: 600.",
    )
    parser.add_argument(
        "--model-version",
        default="pipeline",
        help="pipeline, vlm, or MinerU-HTML. Default: pipeline.",
    )
    parser.add_argument("--language", help="Optional language hint, e.g. ch or en.")
    parser.add_argument("--data-id", help="Optional business-side identifier.")
    parser.add_argument("--page-ranges", help='Optional page range string, e.g. "1-5,8".')
    parser.add_argument("--ocr", action="store_true", help="Enable OCR.")
    parser.add_argument("--disable-formula", action="store_true", help="Disable formula recognition.")
    parser.add_argument("--disable-table", action="store_true", help="Disable table recognition.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass MinerU URL cache when supported.",
    )
    parser.add_argument("--cache-tolerance", type=int, help="Allowed URL cache age in seconds.")
    parser.add_argument(
        "--asset-dir",
        default="assets",
        help="Asset subdirectory name. Default: assets.",
    )
    parser.add_argument(
        "--markdown-name",
        help="Output markdown file name. Default: source markdown name.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep extracted raw MinerU files under output/raw.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_token_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    token = path.read_text(encoding="utf-8-sig").strip()
    return token or None


def parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.is_file():
        return data
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data


def resolve_token(args: argparse.Namespace) -> str:
    for item in (
        read_token_file(args.token_file) if args.token_file else None,
        os.environ.get("MINERU_API_TOKEN", "").strip() or None,
        read_token_file(Path(os.environ["MINERU_API_TOKEN_FILE"])) if os.environ.get("MINERU_API_TOKEN_FILE") else None,
    ):
        if item:
            return item

    for path in (
        SKILL_ROOT / ".env.mineru.local",
        Path.cwd() / ".env.mineru.local",
        Path.cwd() / ".env.local",
    ):
        token = parse_env_file(path).get("MINERU_API_TOKEN", "").strip()
        if token:
            return token

    token = read_token_file(Path.home() / ".config" / "mineru" / "token")
    if token:
        return token

    if sys.stdin.isatty():
        token = getpass("MinerU API token: ").strip()
        if token:
            return token

    raise SystemExit(
        "Missing MinerU token. Set MINERU_API_TOKEN, MINERU_API_TOKEN_FILE, --token-file, "
        f"or place MINERU_API_TOKEN in {SKILL_ROOT / '.env.mineru.local'} "
        "or ~/.config/mineru/token.",
    )


def request_json(method: str, url: str, *, token: str | None = None, payload: dict | None = None) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        message = f"HTTP {exc.code} from MinerU"
        try:
            data = json.loads(body)
            code = data.get("code")
            msg = data.get("msg") or message
            if exc.code == 401 or code in {"A0202", "A0211"}:
                raise ApiError(f"MinerU auth failed: {msg}", status=exc.code, code=code)
            raise ApiError(f"MinerU API error: {msg}", status=exc.code, code=code)
        except json.JSONDecodeError:
            raise ApiError(f"{message}: {body[:300]}", status=exc.code) from exc
    except error.URLError as exc:
        raise ApiError(f"Network error calling MinerU: {exc}") from exc


def request_bytes(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    req = request.Request(url, method=method, headers=headers or {}, data=data)
    with request.urlopen(req, timeout=120) as resp:
        return resp.read()


def upload_file(url: str, path: Path) -> None:
    data = path.read_bytes()
    request_bytes("PUT", url, data=data)


def build_common_payload(args: argparse.Namespace) -> dict:
    payload: dict[str, object] = {"model_version": args.model_version}
    if args.language:
        payload["language"] = args.language
    if args.ocr:
        payload["is_ocr"] = True
    if args.disable_formula:
        payload["enable_formula"] = False
    if args.disable_table:
        payload["enable_table"] = False
    if args.data_id:
        payload["data_id"] = args.data_id
    if args.page_ranges:
        payload["page_ranges"] = args.page_ranges
    if args.no_cache:
        payload["no_cache"] = True
    if args.cache_tolerance is not None:
        payload["cache_tolerance"] = args.cache_tolerance
    return payload


def submit_url(args: argparse.Namespace, token: str) -> str:
    payload = build_common_payload(args)
    payload["url"] = args.url
    data = request_json("POST", f"{BASE_URL}/extract/task", token=token, payload=payload)
    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise ApiError(f"MinerU did not return task_id: {json.dumps(data, ensure_ascii=False)}")
    return task_id


def submit_pdf(args: argparse.Namespace, token: str) -> tuple[str, str]:
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    item: dict[str, object] = {"name": args.pdf.name}
    if args.data_id:
        item["data_id"] = args.data_id
    if args.ocr:
        item["is_ocr"] = True
    if args.page_ranges:
        item["page_ranges"] = args.page_ranges

    payload = build_common_payload(args)
    payload.pop("data_id", None)
    payload.pop("page_ranges", None)
    payload.pop("is_ocr", None)
    payload["files"] = [item]

    data = request_json("POST", f"{BASE_URL}/file-urls/batch", token=token, payload=payload)
    info = data.get("data", {})
    batch_id = info.get("batch_id")
    urls = info.get("file_urls") or []
    if not batch_id or not urls:
        raise ApiError(f"MinerU did not return upload info: {json.dumps(data, ensure_ascii=False)}")

    upload_file(urls[0], args.pdf)
    return batch_id, args.pdf.name


def poll_task(task_id: str, token: str, *, interval: float, timeout: int) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = request_json("GET", f"{BASE_URL}/extract/task/{task_id}", token=token)
        info = data.get("data", {})
        state = info.get("state")
        if state == "done":
            url = info.get("full_zip_url")
            if url:
                return url
            raise ApiError("MinerU task finished but full_zip_url is missing.")
        if state == "failed":
            raise ApiError(f"MinerU task failed: {info.get('err_msg') or 'unknown error'}")
        if state not in POLL_STATES:
            raise ApiError(f"Unexpected MinerU task state: {state}")
        time.sleep(interval)
    raise ApiError(f"Timed out waiting for MinerU task {task_id}")


def poll_batch(batch_id: str, token: str, *, file_name: str | None, interval: float, timeout: int) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = request_json("GET", f"{BASE_URL}/extract-results/batch/{batch_id}", token=token)
        items = data.get("data", {}).get("extract_result") or []
        if not items:
            time.sleep(interval)
            continue
        item = next(
            (entry for entry in items if not file_name or entry.get("file_name") == file_name),
            items[0],
        )
        state = item.get("state")
        if state == "done":
            url = item.get("full_zip_url")
            if url:
                return url
            raise ApiError("MinerU batch item finished but full_zip_url is missing.")
        if state == "failed":
            raise ApiError(f"MinerU batch item failed: {item.get('err_msg') or 'unknown error'}")
        if state not in POLL_STATES:
            raise ApiError(f"Unexpected MinerU batch state: {state}")
        time.sleep(interval)
    raise ApiError(f"Timed out waiting for MinerU batch {batch_id}")


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = request.Request(url, headers={"Accept": "*/*"})
    with request.urlopen(req, timeout=120) as resp, path.open("wb") as handle:
        shutil.copyfileobj(resp, handle)


def safe_extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for item in zf.infolist():
            target = dest / item.filename
            resolved = target.resolve()
            if dest.resolve() not in resolved.parents and resolved != dest.resolve():
                raise SystemExit(f"Unsafe zip entry: {item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(item) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def unwrap(root: Path) -> Path:
    items = list(root.iterdir())
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    return root


def pick_markdown(root: Path) -> Path:
    files = sorted(root.glob("*.md"), key=lambda item: item.stat().st_size, reverse=True)
    if not files:
        raise SystemExit(f"No markdown file found under {root}")
    return files[0]


def pick_content_list(root: Path, md: Path) -> Path | None:
    exact = root / f"{md.stem}_content_list.json"
    if exact.is_file():
        return exact
    files = sorted(root.glob("*_content_list.json"))
    return files[0] if files else None


def rel(path: str) -> str:
    text = path.replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return PurePosixPath(text).as_posix()


def md_refs(text: str) -> list[str]:
    refs: list[str] = []
    patterns = [
        r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)",
        r"<(?:img|video|audio|source)\b[^>]*?\bsrc=[\"']([^\"']+)[\"'][^>]*>",
    ]
    for pattern in patterns:
        import re

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            path = rel(match.group(1))
            if path.startswith(("http://", "https://", "data:")):
                continue
            if "/" not in path:
                continue
            refs.append(path)
    return refs


def content_refs(path: Path | None) -> tuple[list[str], dict[str, str]]:
    if not path:
        return [], {}
    data = json.loads(read_text(path))
    refs: list[str] = []
    kinds: dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        raw = item.get("img_path")
        if not raw:
            continue
        key = rel(str(raw))
        refs.append(key)
        kinds.setdefault(key, str(item.get("type") or "asset"))
    return refs, kinds


def ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def label(kind: str) -> str:
    kind = kind.lower().strip()
    return kind if kind in ASSET_KINDS else "asset"


def mapping(refs: list[str], kinds: dict[str, str]) -> dict[str, str]:
    width = max(3, len(str(max(1, len(refs)))))
    out: dict[str, str] = {}
    for idx, item in enumerate(refs, start=1):
        ext = Path(item).suffix.lower() or ".bin"
        out[item] = f"{idx:0{width}d}-{label(kinds.get(item, 'asset'))}{ext}"
    return out


def rewrite_markdown(text: str, rename: dict[str, str], asset_dir: str) -> str:
    updated = text
    for old, new in rename.items():
        updated = updated.replace(old, f"{asset_dir}/{new}")

    lines = [line.rstrip() for line in updated.splitlines()]
    compact: list[str] = []
    blank = 0
    for line in lines:
        if line:
            blank = 0
            compact.append(line)
            continue
        blank += 1
        if blank <= 2:
            compact.append("")
    return "\n".join(compact).strip() + "\n"


def copy_assets(root: Path, rename: dict[str, str], out: Path) -> None:
    if not rename:
        return
    out.mkdir(parents=True, exist_ok=True)
    for old, new in rename.items():
        src = root / old
        if not src.is_file():
            raise SystemExit(f"Referenced asset not found: {src}")
        shutil.copy2(src, out / new)


def process_raw(
    root: Path, out: Path, *, asset_dir: str, markdown_name: str | None, keep_raw: bool
) -> tuple[Path, Path]:
    md = pick_markdown(root)
    content = pick_content_list(root, md)
    text = read_text(md)
    doc_refs = ordered(md_refs(text))
    json_refs, kinds = content_refs(content)

    final_refs = [item for item in ordered(json_refs) if item in set(doc_refs)]
    final_refs.extend(item for item in doc_refs if item not in final_refs)
    rename = mapping(final_refs, kinds)

    out.mkdir(parents=True, exist_ok=True)
    copy_assets(root, rename, out / asset_dir)

    target_md = out / (markdown_name or md.name)
    target_md.write_text(rewrite_markdown(text, rename, asset_dir), encoding="utf-8")

    manifest = {
        "source_markdown": md.name,
        "content_list": content.name if content else None,
        "assets": [
            {
                "source": old,
                "target": f"{asset_dir}/{new}",
                "kind": kinds.get(old, "asset"),
            }
            for old, new in rename.items()
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if keep_raw:
        raw = out / "raw"
        if raw.exists():
            shutil.rmtree(raw)
        shutil.copytree(root, raw)

    return target_md, out / asset_dir


def prepare_raw(args: argparse.Namespace) -> Path:
    if args.raw_dir:
        if not args.raw_dir.is_dir():
            raise SystemExit(f"Raw directory not found: {args.raw_dir}")
        return args.raw_dir.resolve()

    if args.zip:
        if not args.zip.is_file():
            raise SystemExit(f"ZIP not found: {args.zip}")
        temp = Path(tempfile.mkdtemp(prefix="mineru-zip-"))
        safe_extract(args.zip, temp)
        return unwrap(temp)

    token = resolve_token(args)
    if args.url:
        task_id = submit_url(args, token)
        zip_url = poll_task(task_id, token, interval=args.interval, timeout=args.timeout)
    else:
        batch_id, file_name = submit_pdf(args, token)
        zip_url = poll_batch(
            batch_id,
            token,
            file_name=file_name,
            interval=args.interval,
            timeout=args.timeout,
        )

    temp = Path(tempfile.mkdtemp(prefix="mineru-api-"))
    zip_path = temp / "result.zip"
    download_file(zip_url, zip_path)
    raw = temp / "raw"
    safe_extract(zip_path, raw)
    return unwrap(raw)


def main() -> int:
    args = parse_args()
    try:
        raw = prepare_raw(args)
        md, assets = process_raw(
            raw,
            args.output,
            asset_dir=args.asset_dir,
            markdown_name=args.markdown_name,
            keep_raw=args.keep_raw,
        )
    except ApiError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Markdown: {md}")
    print(f"Assets:   {assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
