#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 turboderp/Qwen3.8-27B-exl3 的指定分支（每个 bpw 档位是一个 git 分支）。

用法示例：
    # 默认下载 3.50bpw（4090 24GB 推荐）
    python download_model.py

    # 指定分支与输出目录
    python download_model.py --revision 4.00bpw --out models/Qwen3.8-27B-4.00bpw

    # 国内网络可改用镜像
    python download_model.py --mirror

    # 查看所有可选分支
    python download_model.py --list
"""

import argparse
import os
import sys
import urllib.request

REPO_ID = "turboderp/Qwen3.8-27B-exl3"
OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"

# 各分支的磁盘占用（十进制 GB），用于下载前提示。
# “_V” 结尾的分支把视觉塔也量化了，比同 bpw 的版本小约 0.4–0.6 GB。
BRANCH_SIZES = {
    "SC_1.40bpw_H3": 8.4,
    "SC_1.40bpw_H3_V3": 8.0,
    "SC_1.60bpw_H3": 9.0,
    "SC_1.60bpw_H3_V3": 8.6,
    "SC_1.80bpw_H3": 9.6,
    "SC_1.80bpw_H3_V3": 9.2,
    "SC_2.00bpw_H3": 10.2,
    "SC_2.00bpw_H3_V3": 9.8,
    "SC_2.20bpw_H3": 10.8,
    "SC_2.20bpw_H3_V3": 10.4,
    "SC_3.00bpw_H4": 13.5,
    "SC_3.00bpw_H4_V4": 13.1,
    "SC_4.00bpw_H5": 16.7,
    "SC_4.00bpw_H5_V6": 16.1,
    "SC_5.00bpw_H6": 19.9,
    "SC_5.00bpw_H6_V6": 19.3,
    "SC_6.00bpw_H6": 23.0,
    "SC_6.00bpw_H6_V6": 22.6,
    "2.00bpw": 10.8,
    "2.50bpw": 12.3,
    "3.00bpw": 13.8,
    "3.50bpw": 15.4,
    "4.00bpw": 16.9,
    "5.00bpw": 19.9,
    "6.00bpw": 23.0,
}


# 一个确定存在的大体积 LFS 文件，用于探测真正的权重下载通道
# （huggingface.co 的小文件能直连，但 safetensors 走 LFS CDN，国内常常连不上）
PROBE_LFS_URL = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/model-00001-of-00018.safetensors"

PROBE_SMALL_URL = f"https://huggingface.co/{REPO_ID}/resolve/3.50bpw/config.json"


def _fetch_bytes(url: str, timeout: float, want: int = 1 << 20, minimum: int = 1 << 19) -> bool:
    """
    尝试真实拉取 want 字节，只有拿到至少 minimum 字节才算这个通道可用。

    只请求 1 个字节是不行的：某些网络下小请求能过，但一旦开始传大文件就会被阻断，
    所以这里必须真的读到足够多的数据。
    """
    try:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{want - 1}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return False
            got = 0
            while got < minimum:
                chunk = resp.read(minimum - got)
                if not chunk:
                    break
                got += len(chunk)
            return got >= minimum
    except Exception:
        return False


def probe_direct(timeout: float = 15.0) -> bool:
    """只有当元数据文件和大体积 LFS 文件都真的能下载时，才算可以直连。"""
    if not _fetch_bytes(PROBE_SMALL_URL, timeout, want=4096, minimum=512):
        return False
    return _fetch_bytes(PROBE_LFS_URL, timeout)


def probe(endpoint: str, timeout: float = 15.0) -> bool:
    if "huggingface.co" in endpoint and "hf-mirror" not in endpoint:
        return probe_direct(timeout)
    return _fetch_bytes(f"{endpoint}/{REPO_ID}/resolve/3.50bpw/config.json", timeout, want=4096, minimum=512)


def apply_endpoint(mode: str) -> str:
    """
    决定使用哪个下载端点。

    :param mode: auto / mirror / direct
    :return: 实际使用的 endpoint
    """
    if mode == "auto":
        direct_ok = probe(OFFICIAL_ENDPOINT)
        mode = "direct" if direct_ok else "mirror"
        if direct_ok:
            print("网络检测：可直连 huggingface.co")
        else:
            print("网络检测：huggingface.co 直连不通，自动改用 hf-mirror.com 镜像")

    endpoint = MIRROR_ENDPOINT if mode == "mirror" else OFFICIAL_ENDPOINT
    if mode == "mirror":
        # 让 huggingface_hub 内部的请求也走镜像
        os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
        # 镜像站不支持 Xet 传输协议，必须关掉，否则会尝试连 xet 的 CDN 而失败
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print(f"使用镜像：{MIRROR_ENDPOINT}")
    else:
        print(f"使用官方端点：{OFFICIAL_ENDPOINT}")
    return endpoint


def list_branches(endpoint: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint)
    print(f"仓库：{REPO_ID}\n")
    print(f"{'分支':<28}{'约占用':>10}   说明")
    print("-" * 78)
    size_of = {}
    for b in api.list_repo_refs(REPO_ID, repo_type="model").branches:
        size_of[b.name] = None

    notes = {
        "main": "模型卡与校准/评测数据，不含权重",
    }
    for name in sorted(size_of):
        if name == "main":
            size, note = "-", notes["main"]
        else:
            gb = BRANCH_SIZES.get(name)
            size = f"{gb:.1f} GB" if gb else "-"
            if name.startswith("SC_"):
                note = "自校准量化（同 bpw 质量更好）；_V 结尾=视觉塔也量化"
            else:
                note = "普通均匀位宽量化"
        print(f"{name:<28}{size:>10}   {note}")
    print()
    print("RTX 4090 24GB 推荐：3.50bpw（约 15.4 GB 权重，可留约 7 GB 给 KV 缓存）")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="下载 Qwen3.8-27B EXL3 量化模型的指定分支",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revision", "-r", default="3.50bpw", help="要下载的分支名（默认 3.50bpw）")
    ap.add_argument("--out", "-o", default=None, help="输出目录（默认 models/Qwen3.8-27B-<revision>）")
    ap.add_argument("--mirror", action="store_true", help="强制使用 hf-mirror.com 镜像")
    ap.add_argument("--direct", action="store_true", help="强制直连 huggingface.co（默认自动探测）")
    ap.add_argument("--token", default=None, help="HuggingFace token（一般为空，该仓库无需鉴权）")
    ap.add_argument("--list", action="store_true", help="列出所有可选分支后退出")
    ap.add_argument("--workers", "-j", type=int, default=8, help="并发下载线程数（默认 8）")
    args = ap.parse_args()

    mode = "mirror" if args.mirror else ("direct" if args.direct else "auto")
    endpoint = apply_endpoint(mode)

    if args.list:
        list_branches(endpoint)
        return 0

    from huggingface_hub import snapshot_download

    out_dir = args.out or f"models/Qwen3.8-27B-{args.revision}"
    out_dir = os.path.abspath(out_dir)

    gb = BRANCH_SIZES.get(args.revision)
    print(f"仓库   : {REPO_ID}")
    print(f"分支   : {args.revision}")
    print(f"输出到 : {out_dir}")
    if gb:
        print(f"预计磁盘占用：约 {gb:.1f} GB")
    print("（支持断点续传，中断后重新执行本命令即可）\n")

    path = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=out_dir,
        token=args.token,
        max_workers=args.workers,
        endpoint=endpoint,
    )
    print(f"\n完成：{path}")
    print("\n下一步：把 config.yaml 里的 model.path 指向该目录，然后运行 start.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
