#!/usr/bin/env python3
"""`scripts/run_practice.py`, nhưng nạp `.env` trước khi chạy.

VÌ SAO CẦN: `arena/model.py` đọc credential từ BIẾN MÔI TRƯỜNG
(`ARENA_BASE_URL` / `ARENA_API_KEY` / `ARENA_MODEL`) và cố tình KHÔNG tự
quay về đường chạy offline khi thiếu. File này chỉ làm đúng một việc: đọc
`.env` cạnh gốc lab, đẩy vào `os.environ`, rồi giao lại toàn bộ cho
`run_practice.main`. Mọi cờ của `run_practice.py` dùng lại nguyên vẹn.

    cp .env.example .env          # rồi điền ARENA_API_KEY
    python scripts/run_real.py --brief pub-08-an-toan-boc-do \
           --out runs/real-pub08.json

TỐN TIỀN THẬT: mặc định `--model real`, và bỏ trống `--brief` nghĩa là
chạy cả 9 brief công khai. Chạy MỘT brief trước, đọc `runs/*.json`, rồi
mới mở rộng.

KHÔNG nằm trong bài nộp (bài nộp chỉ thu `harness/`), và không đụng tới
`arena/` hay `scripts/run_practice.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))
if str(LAB_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB_ROOT / "scripts"))

ENV_PATH = LAB_ROOT / ".env"
PLACEHOLDER = "sk-thay-bang-key-that"


def load_dotenv(path: Path) -> int:
    """Nạp `KEY=VALUE` từ `path`. Trả về số biến đã đặt.

    Chỉ dùng thư viện chuẩn (lab không có dependency ngoài). Biến đã có
    sẵn trong môi trường được GIỮ NGUYÊN — dòng lệnh phải thắng file.
    """
    if not path.exists():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def patch_runner_config(module, temperature: float, max_tokens_param: str) -> None:
    """Đặt hai tham số CÔNG KHAI của runner mà `run_practice.py` không mở ra.

    Không sửa `arena/` — năm file đóng băng giữ nguyên MD5. Chỉ là dựng
    `RunnerConfig` với hai trường nó vốn đã có.

    1. `temperature` (mặc định 0.0 = thiết lập vòng chấm). gpt-5.6-luna
       TỪ CHỐI 0.0: `unsupported_value ... Only the default (1) value is
       supported` -> cả lượt chạy chết bằng HTTP 400.

    2. `max_tokens_param` (mặc định "auto"). Các model OpenAI đời mới chỉ
       nhận `max_completion_tokens`. Runner CÓ đường tự đổi, nhưng nó
       KHÔNG BAO GIỜ bắn được với endpoint thật, và lý do đáng ghi lại:

           RealModel._post  -> để urllib.error.HTTPError bay lên nguyên
                               trạng, KHÔNG đọc body phản hồi.
           str(HTTPError)   -> "HTTP Error 400: Bad Request"
           RealModel.complete -> bọc đúng chuỗi đó vào RealModelError.
           runner._fallback_param -> dò `_PARAM_REJECTION_HINTS`
                               ("max_tokens", "unsupported parameter", …)
                               trong chuỗi ấy -> không khớp gì -> raise.

       Câu giải thích ("Use 'max_completion_tokens' instead") nằm trong
       BODY, mà body thì đã bị vứt. Nên đặt thẳng tham số đúng từ đầu:
       `_invoke` thấy primary != "max_tokens" là gọi `_post_with_param`
       ngay, không cần lượt thử hỏng nào.
    """
    original = module.RunnerConfig

    def factory(**kwargs):
        kwargs.setdefault("temperature", temperature)
        kwargs.setdefault("max_tokens_param", max_tokens_param)
        return original(**kwargs)

    module.RunnerConfig = factory


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    def take(flag: str, default: str) -> str:
        if flag not in argv:
            return default
        index = argv.index(flag)
        value = argv[index + 1]
        del argv[index:index + 2]
        return value

    temperature = float(take("--temperature", "1.0"))
    max_tokens_param = take("--max-tokens-param", "max_completion_tokens")
    loaded = load_dotenv(ENV_PATH)

    key = os.environ.get("ARENA_API_KEY", "")
    if not key or key == PLACEHOLDER:
        where = ENV_PATH if ENV_PATH.exists() else f"{ENV_PATH} (chưa có)"
        print(
            f"Chưa có ARENA_API_KEY thật.\n"
            f"  1. cp .env.example .env\n"
            f"  2. điền key vào {where}\n"
            f"Ba biến cần có: ARENA_BASE_URL, ARENA_API_KEY, ARENA_MODEL.",
            file=sys.stderr,
        )
        return 2

    if loaded:
        print(f"[run_real] đã nạp {loaded} biến từ {ENV_PATH.name}", file=sys.stderr)
    if not any(a == "--model" or a.startswith("--model=") for a in argv):
        argv = ["--model", "real", *argv]

    import run_practice

    patch_runner_config(run_practice, temperature, max_tokens_param)
    return run_practice.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
