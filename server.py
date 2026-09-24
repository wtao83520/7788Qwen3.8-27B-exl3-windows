#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3.8-27B (EXL3) 的 OpenAI 兼容推理服务。

底层用 ExLlamaV3 加载 EXL3 量化权重，对外暴露：
    GET  /v1/models
    POST /v1/chat/completions     （流式 / 非流式、图片输入、reasoning_content、tool_calls、logprobs）
    POST /v1/completions          （legacy 文本补全）
    GET  /health  /v1/health

启动：
    python server.py --config config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from jinja2 import Environment
from pydantic import BaseModel

logger = logging.getLogger("qwen38.server")

# 输出编码处理。分两种情况：
#   - 输出到控制台：Windows 控制台默认 GBK，保持原编码、只把无法编码的字符替换掉，
#     这样中文在普通控制台里仍然正常显示。
#   - 输出被重定向到文件（控制面板就是这么启动的）：强制 UTF-8。否则日志文件里是
#     GBK 字节，而读取方（面板 / 编辑器 / tail）按 UTF-8 解，中文全是乱码。
for _stream in (sys.stdout, sys.stderr):
    try:
        if getattr(_stream, "isatty", lambda: True)():
            _stream.reconfigure(errors="replace")
        else:
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PAGE_SIZE = 256  # ExLlamaV3 分页大小，max_num_tokens 必须是它的整数倍
# 每 token 的 K/V 元素数：16 层全注意力 × 4 个 KV head × 256 head_dim × 2(K/V)
# = 32768。所以每 token 的 KV 字节数 = 32768 × (k_bits + v_bits) / 2 / 8。
ELEMS_PER_TOKEN = 16 * 4 * 256 * 2
# 生成时除了 prompt 还得额外留几格，否则 exllamav3 会抛
#   AssertionError: Job requires 1025 pages (only 1024 available)
# 具体是：job 内部按 max_new_tokens + 1 + num_draft_tokens 去算页数，
# 另外多模态/前缀 token 还会再占一格（pagetable 里的 has_prefix_token）。
# 这里只算固定部分，草稿那部分在 _prepare 里按实际 num_draft_tokens 加。
RESERVED_TOKENS = 2
REASONING_EFFORTS = ("xhigh", "medium", "low")


# ===========================================================================
# 配置
# ===========================================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "api_keys": [],
        "cors_origins": ["*"],
    },
    "model": {
        "path": "models/Qwen3.8-27B-3.50bpw",
        "name": None,
        "device": "cuda:0",
        "tensor_parallel": False,
        "use_per_device": None,
        "reserve_per_device": None,
        "max_seq_len": 32768,
        "max_batch_size": 4,
        "max_chunk_size": 2048,
        "cache_quant": 8,
        "cache_k_bits": None,
        "cache_v_bits": None,
        "load_vision": True,
        "mtp_draft": True,
        "draft_model": None,
        "progressbar": True,
    },
    "defaults": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "temperature_last": False,
        "max_tokens": 32768,
        "max_tokens_limit": None,
        "seed": None,
        "stop": [],
        "include_usage": True,
    },
    "chat_template": {
        "name": None,
        "vars": {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
            "preserve_thinking": True,
        },
        "split_reasoning": True,
    },
    "limits": {
        "max_concurrent_requests": 8,
        "max_prompt_chars": None,
    },
}

# 采样参数白名单（请求级可覆盖）
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
    "temperature_last",
    "seed",
)


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 优先。"""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None) -> dict[str, Any]:
    cfg = DEFAULT_CONFIG
    if path and os.path.isfile(path):
        with open(path, encoding="utf8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    # 环境变量覆盖，便于容器化部署
    env_map = {
        "QWEN38_HOST": ("server", "host", str),
        "QWEN38_PORT": ("server", "port", int),
        "QWEN38_MODEL_PATH": ("model", "path", str),
        "QWEN38_MAX_SEQ_LEN": ("model", "max_seq_len", int),
        "QWEN38_CACHE_QUANT": ("model", "cache_quant", int),
        "QWEN38_API_KEYS": ("server", "api_keys", lambda v: [k.strip() for k in v.split(",") if k.strip()]),
    }
    for env, (section, key, cast) in env_map.items():
        if env in os.environ:
            cfg.setdefault(section, {})[env_map[env][1]] = cast(os.environ[env])
    return cfg


# ===========================================================================
# 思维链拆分：把输出按 <think>...</think> 分成 reasoning_content / content
# ===========================================================================

class ThinkSplitter:
    """
    开启思考模式时，prompt 结尾已经是 `<think>\\n`，模型先产出推理过程，
    再输出 `</think>` 然后给出正式回答。这里做流式增量切分。
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, enabled: bool, start_in_reasoning: bool):
        self.enabled = bool(enabled)
        self.mode = "reasoning" if (self.enabled and start_in_reasoning) else "content"
        self.buf = ""
        # 刚结束推理段：此时后续的空白字符（换行）属于排版，全部丢弃
        self.strip_leading = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if not self.enabled:
            return [("content", text)]

        self.buf += text
        out: list[tuple[str, str]] = []
        while True:
            if self.mode == "reasoning":
                if self.buf.startswith(self.OPEN):
                    self.buf = self.buf[len(self.OPEN):]
                    continue
                idx = self.buf.find(self.CLOSE)
                if idx >= 0:
                    if idx > 0:
                        out.append(("reasoning", self.buf[:idx]))
                    self.buf = self.buf[idx + len(self.CLOSE):]
                    self.mode = "content"
                    self.strip_leading = True
                    continue
                hold = len(self.CLOSE) - 1
                if len(self.buf) > hold:
                    out.append(("reasoning", self.buf[:-hold]))
                    self.buf = self.buf[-hold:]
                break
            else:
                if self.strip_leading:
                    # 丢掉 </think> 之后、正文之前的所有空白（可能跨多个 chunk）
                    if not self.buf.strip():
                        self.buf = ""
                        break
                    self.buf = self.buf.lstrip()
                    self.strip_leading = False
                if self.buf.startswith(self.OPEN):
                    self.buf = self.buf[len(self.OPEN):]
                    self.mode = "reasoning"
                    continue
                idx = self.buf.find(self.OPEN)
                if idx >= 0:
                    if idx > 0:
                        out.append(("content", self.buf[:idx]))
                    self.buf = self.buf[idx:]
                    continue
                hold = len(self.OPEN) - 1
                if len(self.buf) > hold:
                    out.append(("content", self.buf[:-hold]))
                    self.buf = self.buf[-hold:]
                break
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self.buf:
            return []
        if self.mode == "content" and self.strip_leading:
            # 剩余全是空白，直接丢掉
            self.buf = ""
            return []
        out = [(self.mode, self.buf)]
        self.buf = ""
        return out


# ===========================================================================
# 引擎
# ===========================================================================

class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mcfg = cfg["model"]
        self.tcfg = cfg["chat_template"]
        self.dcfg = cfg["defaults"]

        self.model_dir = os.path.abspath(self.mcfg["path"])
        self.name = self.mcfg.get("name") or os.path.basename(self.model_dir.rstrip("/\\"))

        self.model = None
        self.vision_model = None
        self.draft_model = None
        self.draft_kind = None
        # 投机解码每步会多预测这么多 token，job 的缓存预算要把这段算进去。
        # 注意不能去读 engine.generator.num_draft_tokens：engine.generator 是
        # AsyncGenerator 包装类，真正的 Generator 在它的 .generator 属性里，
        # 直接读会拿到默认值 0，于是预算算少，接近上限时就会 assert 失败。
        self.num_draft_tokens = 0
        self.cache = None
        self.draft_cache = None
        # 实际生效的 K/V 量化位数（在 _make_cache 里确定），供 /health 上报
        self.cache_k_bits: int | None = None
        self.cache_v_bits: int | None = None
        # 投机解码累计统计（只有用草稿模型时才有值），供 /health 换算出接受长度：
        #   验证轮数 ≈ 生成 token 数 - 接受草稿数（每轮除了接受的草稿，还会多出 1 个
        #   目标模型自己采样的 token）
        #   接受长度 = 生成 token 数 / 验证轮数
        self.draft_accepted = 0
        self.draft_rejected = 0
        self.draft_requests = 0
        self.tokenizer = None
        self.generator = None  # AsyncGenerator
        self.template = None
        self.max_seq_len = int(self.mcfg["max_seq_len"])

    # ---------------- 加载 ----------------

    def _load_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {"progressbar": bool(self.mcfg["progressbar"])}
        if self.mcfg["tensor_parallel"]:
            kwargs["tensor_p"] = True
            if self.mcfg.get("use_per_device"):
                kwargs["use_per_device"] = self.mcfg["use_per_device"]
            if self.mcfg.get("reserve_per_device"):
                kwargs["reserve_per_device"] = self.mcfg["reserve_per_device"]
        else:
            kwargs["device"] = self.mcfg["device"]
        return kwargs

    def _resolve_max_history(self, draft_size: int) -> int:
        """KV 缓存要给递归层（线性注意力）预留多少「历史」。

        这个值不能想当然地留 0：开启投机解码时，验证草稿 token 的那一步会把
        recurrent_history 打开，递归层就需要 [slots, max_history + 1, ...] 的状态张量。
        留 0 会得到 [slots, 1, ...]，运行时直接抛
        `RuntimeError: recurrent_state must be [num_slots, max_history + 1, ...]`。
        exllamav3 自己的 CLI 就是这么算的（见 model_init.py），这里保持一致。
        """
        fixed = self.mcfg.get("max_history")
        if fixed is not None:
            return int(fixed)
        return max(0, int(draft_size))

    def _make_cache(self, model, label: str, max_history: int = 0):
        from exllamav3 import Cache, CacheLayer_fp16, CacheLayer_quant

        bits = int(self.mcfg.get("cache_quant") or 16)
        tokens = self.max_seq_len
        if tokens % PAGE_SIZE:
            tokens = ((tokens + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        common = {
            "max_num_tokens": tokens,
            "max_batch_size": int(self.mcfg["max_batch_size"]),
            "max_history": max_history,
        }
        if bits >= 16:
            self.cache_k_bits = self.cache_v_bits = 16
            logger.info("[%s] KV 缓存 fp16 / %d tokens / max_history=%d", label, tokens, max_history)
            return Cache(model, layer_type=CacheLayer_fp16, **common)

        k_bits = int(self.mcfg.get("cache_k_bits") or bits)
        v_bits = int(self.mcfg.get("cache_v_bits") or bits)
        for b in (k_bits, v_bits):
            if not (2 <= b <= 8):
                raise SystemExit(f"cache_quant 必须是 16，或 2~8 的整数（收到 {b}）")
        self.cache_k_bits, self.cache_v_bits = k_bits, v_bits
        logger.info("[%s] KV 缓存 k=%dbit v=%dbit / %d tokens / max_history=%d",
                    label, k_bits, v_bits, tokens, max_history)
        return Cache(model, layer_type=CacheLayer_quant, k_bits=k_bits, v_bits=v_bits, **common)

    def load(self) -> None:
        """阻塞式加载全部权重，应放在工作线程里调用。"""
        from exllamav3 import Config, Model, Tokenizer

        if not os.path.isdir(self.model_dir):
            raise SystemExit(
                f"模型目录不存在：{self.model_dir}\n请先运行：python download_model.py --mirror"
            )
        if not os.path.isfile(os.path.join(self.model_dir, "config.json")):
            raise SystemExit(f"{self.model_dir} 下没有 config.json，不是有效的 EXL3 模型目录")

        t0 = time.time()
        logger.info("读取模型配置：%s", self.model_dir)
        config = Config.from_directory(self.model_dir)
        self.tokenizer = Tokenizer.from_config(config)
        load_kwargs = self._load_kwargs()

        if self.mcfg.get("load_vision", True) and "vision" in config.model_classes:
            logger.info("加载视觉塔 …")
            self.vision_model = Model.from_config(config, component="vision")
            self.vision_model.load(**load_kwargs)
        else:
            logger.info("跳过视觉塔（仅文本模式）")

        # ---- 先确定草稿模型：它决定了 KV 缓存要给递归层预留多少历史 ----
        # 注意 Model.from_config 只构造模块树、不读权重，所以提前构造是便宜的。
        self.draft_kind = None
        draft_external = None
        draft_dir = self.mcfg.get("draft_model")
        if draft_dir:
            draft_dir = os.path.abspath(draft_dir)
            if not os.path.isdir(draft_dir):
                logger.warning("draft_model 目录不存在，忽略：%s", draft_dir)
            else:
                draft_external = draft_dir

        draft_size = 0
        if draft_external:
            try:
                self.draft_model = Model.from_config(Config.from_directory(draft_external))
            except Exception as exc:
                logger.warning("外部草稿模型不可用，尝试回退：%s", exc)
                self.draft_model = None
        if self.draft_model is None and self.mcfg.get("mtp_draft", True) and "mtp" in config.model_classes:
            try:
                self.draft_model = Model.from_config(config, component="mtp")
            except Exception as exc:
                logger.warning("MTP 草稿层不可用，回退到普通解码：%s", exc)
                self.draft_model = None

        if self.draft_model is not None:
            draft_size = int(self.draft_model.caps.get("default_draft_size", 4))
        max_history = self._resolve_max_history(draft_size)
        if max_history:
            logger.info("递归层历史预留：max_history=%d（投机解码草稿 %d token）", max_history, draft_size)

        self.model = Model.from_config(config)
        self.cache = self._make_cache(self.model, "text", max_history)
        where = "张量并行" if self.mcfg["tensor_parallel"] else str(self.mcfg["device"])
        logger.info("加载主模型（%s）…", where)
        self.model.load(**load_kwargs)

        if self.draft_model is not None:
            label = "外部草稿模型" if draft_external else "MTP 草稿层"
            try:
                logger.info("加载%s（投机解码）…", label)
                # 草稿缓存必须和主缓存同样大小，也要同样的历史预留
                self.draft_cache = self._make_cache(self.draft_model, label, max_history)
                self.draft_model.load(**load_kwargs)
                if draft_external:
                    self.draft_kind = "dflash" if self.draft_model.caps.get("dflash_draft") else "draft"
                else:
                    self.draft_kind = "mtp"
                # num_draft_tokens 必须和 exllamav3 自己算出来的值一致：Generator 在
                # 没有显式传值时用 draft_model.caps["default_draft_size"]（MTP 是 4，
                # DFlash2 是 block_size - 1 = 7），job 再按
                #   max_new_tokens + 1 + num_draft_tokens
                # 去算需要多少缓存页。
                # 这里曾经对非 MTP 的草稿写死 0（以为「别多占缓存预算」），结果是
                # _prepare 少留 6 格，请求把上下文顶满时会抛
                #   AssertionError: Job requires N pages (only M available)
                # DFlash2 的块 = 7 草稿 + 1 锚点，正好等于 default_draft_size + 1。
                self.num_draft_tokens = draft_size
            except Exception as exc:  # 投机解码只是加速项，失败就回退
                logger.warning("%s加载失败，关闭投机解码：%s", label, exc)
                self.draft_model = None
                self.draft_cache = None
                self.draft_kind = None
                self.num_draft_tokens = 0

        self._load_template()
        logger.info("模型加载完成，用时 %.1fs", time.time() - t0)

    def _load_template(self) -> None:
        """读取模型自带的 HF Jinja 对话模板。"""
        text = None
        name = self.tcfg.get("name")
        for candidate in filter(None, [name, os.path.join(self.model_dir, name or ""), "chat_template.jinja", "chat_template.j2"]):
            path = candidate if os.path.isabs(candidate) else os.path.join(self.model_dir, candidate)
            if os.path.isfile(path):
                with open(path, encoding="utf8") as f:
                    text = f.read()
                break
        if text is None:
            tc = os.path.join(self.model_dir, "tokenizer_config.json")
            if os.path.isfile(tc):
                with open(tc, encoding="utf8") as f:
                    text = json.load(f).get("chat_template")
        if not text:
            logger.warning("未找到 chat_template，回退到内置 ChatML 模板")
            self.template = None
            return

        env = Environment(autoescape=False)

        def raise_exception(message: str):
            raise ValueError(message)

        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt="%Y-%m-%d": time.strftime(fmt)
        self.template = env.from_string(text)
        logger.info("已加载对话模板（%d 字符）", len(text))

    def _render_chatml_fallback(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(f"<|im_start|>{msg.get('role', 'user')}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n<think>\n")
        return "".join(parts)

    def render(self, messages: list[dict], template_vars: dict) -> tuple[str, bool]:
        """
        渲染对话模板。

        :return: (prompt, 输出是否从推理段开始)
        """
        thinking = bool(template_vars.get("enable_thinking", True))
        if self.template is None:
            return self._render_chatml_fallback(messages), thinking
        try:
            prompt = self.template.render(messages=messages, add_generation_prompt=True, **template_vars)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"对话模板渲染失败：{exc}") from exc
        started_in_reasoning = thinking and prompt.rstrip().endswith("<think>")
        return prompt, started_in_reasoning

    # ---------------- 采样器 ----------------

    def build_sampler(self, p: dict):
        from exllamav3 import ComboSampler

        logit_bias = None
        if p.get("logit_bias"):
            vocab = getattr(self.model.config, "vocab_size", None) or (1 << 30)
            logit_bias = {}
            for key, value in p["logit_bias"].items():
                try:
                    token_id = int(key)
                except (TypeError, ValueError):
                    continue  # OpenAI 的 logit_bias 键是 token id，不是数字的忽略
                if 0 <= token_id < vocab:
                    logit_bias[token_id] = float(value)
            if not logit_bias:
                logit_bias = None

        return ComboSampler(
            rep_p=float(p["repetition_penalty"]),
            freq_p=float(p["frequency_penalty"]),
            pres_p=float(p["presence_penalty"]),
            temperature=float(p["temperature"]),
            min_p=float(p["min_p"]),
            top_k=int(p["top_k"]),
            top_p=float(p["top_p"]),
            temp_last=bool(p["temperature_last"]),
            logit_bias=logit_bias,
            dry_multiplier=float(p.get("dry_multiplier") or 0.0),
        )

    # ---------------- 图片 ----------------

    def _decode_image(self, url: str):
        from PIL import Image

        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if "base64" not in header:
                raise HTTPException(status_code=400, detail="仅支持 base64 的 data URI 图片")
            try:
                raw = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail="图片 base64 解码失败") from exc
            return Image.open(io.BytesIO(raw)).convert("RGB")

        if url.startswith(("http://", "https://")):
            import requests

            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        if os.path.isfile(url):
            return Image.open(url).convert("RGB")

        raise HTTPException(status_code=400, detail=f"不支持的图片地址：{url[:64]}")

    def prepare_images(self, messages: list[dict]) -> tuple[list[dict], list]:
        """
        把 OpenAI 的图片内容块替换成 MMEmbedding 的文本别名；
        渲染后再由 tokenizer 把别名展开成真正的视觉 token。
        """
        def image_parts(msg) -> bool:
            content = msg.get("content")
            return isinstance(content, list) and any(
                isinstance(p, dict) and (p.get("type") in ("image_url", "image") or "image_url" in p)
                for p in content
            )

        if self.vision_model is None:
            if any(image_parts(m) for m in messages):
                raise HTTPException(
                    status_code=400,
                    detail="模型未加载视觉塔（model.load_vision = false），无法处理图片输入",
                )
            return messages, []

        out_messages: list[dict] = []
        embeddings: list = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out_messages.append(msg)
                continue
            new_parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind in ("image_url", "image") or "image_url" in part:
                    # 用和 build_chat_messages 相同的解析，两边行为保持一致
                    url = _extract_image_url(part)
                    mme = self.vision_model.get_image_embeddings(
                        tokenizer=self.tokenizer, image=self._decode_image(url)
                    )
                    embeddings.append(mme)
                    new_parts.append({"type": "text", "text": mme.text_alias})
                else:
                    new_parts.append({"type": "text", "text": part.get("text", "")})
            out_messages.append({**msg, "content": new_parts})
        return out_messages, embeddings

    async def close(self) -> None:
        if self.generator is not None:
            try:
                await self.generator.close()
            except Exception:
                logger.exception("关闭生成器时出错")
            self.generator = None


# ===========================================================================
# 请求模型
# ===========================================================================

class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: str | None = None
    tool_calls: Any = None
    tool_call_id: str | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    # 采样参数
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    temperature_last: bool | None = None
    logit_bias: dict[str, float] | None = None
    dry_multiplier: float | None = None
    # 输出控制
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = 1
    stream: bool = False
    stream_options: dict | None = None
    logprobs: bool | None = False
    top_logprobs: int | None = 0
    # Qwen3.8 特有
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    preserve_thinking: bool | None = None
    # 工具调用
    tools: list[dict] | None = None
    tool_choice: Any = None

    model_config = {"extra": "allow"}


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    temperature_last: bool | None = None
    logit_bias: dict[str, float] | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    logprobs: bool | None = False
    top_logprobs: int | None = 0

    model_config = {"extra": "allow"}


# ===========================================================================
# 全局状态与 FastAPI
# ===========================================================================

CFG: dict[str, Any] = load_config(os.environ.get("QWEN38_CONFIG", "config.yaml"))
ENGINE: Engine | None = None
SEMAPHORE: asyncio.Semaphore | None = None
LOAD_ERROR: str | None = None
# uvicorn 的 Server 实例，供 /admin/shutdown 优雅退出用（见 main()）
SERVER: Any = None
# 服务进程启动时刻，用于算运行时长
STARTED_AT: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE, SEMAPHORE, LOAD_ERROR
    SEMAPHORE = asyncio.Semaphore(max(1, int(CFG["limits"]["max_concurrent_requests"])))
    engine = Engine(CFG)
    try:
        await asyncio.to_thread(engine.load)
    except SystemExit as exc:
        LOAD_ERROR = str(exc)
        logger.error("模型加载失败：%s", exc)
        ENGINE = None
        yield
        return
    except Exception as exc:  # 环境不完整 / 权重损坏 / 显存不足等
        hint = ""
        if isinstance(exc, ImportError):
            hint = "（运行环境不完整：请先执行 setup.ps1 或 _bootstrap.ps1 安装 torch 与 exllamav3）"
        elif "out of memory" in str(exc).lower():
            hint = "（显存不足：请调小 model.max_seq_len / cache_quant，或设置 load_vision: false）"
        LOAD_ERROR = f"{type(exc).__name__}: {exc}{hint}"
        logger.error("模型加载失败：%s", LOAD_ERROR)
        logger.debug("详细堆栈", exc_info=True)
        ENGINE = None
        yield
        return

    from exllamav3 import AsyncGenerator

    engine.generator = AsyncGenerator(
        model=engine.model,
        cache=engine.cache,
        tokenizer=engine.tokenizer,
        max_batch_size=int(CFG["model"]["max_batch_size"]),
        max_chunk_size=int(CFG["model"].get("max_chunk_size") or 2048),
        draft_model=engine.draft_model,
        draft_cache=engine.draft_cache,
    )
    ENGINE = engine

    # 一致性校验：服务端算「还能生成多少 token」用的是 engine.num_draft_tokens，
    # 而 job 真正用的页数预算是 AsyncGenerator 里那个 Generator 自己的
    # num_draft_tokens。两者不一致时，请求把上下文顶满会直接抛
    #   AssertionError: Job requires N pages (only M available)
    # 所以在启动时对一次账，不一致就明确告警，不要等到用户请求才炸。
    inner_generator = getattr(engine.generator, "generator", None)
    real_draft = int(getattr(inner_generator, "num_draft_tokens", -1) or 0)
    if engine.draft_kind and real_draft >= 0 and real_draft != engine.num_draft_tokens:
        logger.warning(
            "草稿 token 数不一致：服务端按 %d 预留，exllamav3 实际按 %d 算页数。"
            "「prompt + max_tokens」贴近上下文上限的请求会报页数不足。",
            engine.num_draft_tokens, real_draft,
        )

    logger.info(
        "服务就绪：%s（%s，%s）",
        engine.name,
        f"投机解码={engine.draft_kind}" if engine.draft_kind else "无投机解码",
        "含视觉塔" if engine.vision_model else "纯文本",
    )
    try:
        yield
    finally:
        await engine.close()


app = FastAPI(title="Qwen3.8-27B EXL3 OpenAI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG["server"].get("cors_origins") or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _auth(authorization: str | None = Header(default=None)) -> None:
    keys = CFG["server"].get("api_keys") or []
    if not keys:
        return
    token = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else ""
    if token not in keys:
        raise HTTPException(status_code=401, detail="API key 无效")


def get_engine() -> Engine:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail=LOAD_ERROR or "模型尚未加载完成")
    return ENGINE


SEEDED = object()


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


def _new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _finish_reason(eos_reason: str | None) -> str:
    return "length" if eos_reason == "max_new_tokens" else "stop"


def _usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# ===========================================================================
# 参数解析
# ===========================================================================

def resolve_params(engine: Engine, req) -> tuple[dict, list[str]]:
    """把请求里的采样参数与配置默认值合并，返回 (params, stop_conditions)。"""
    d = engine.dcfg
    out: dict[str, Any] = {key: d.get(key) for key in SAMPLING_KEYS}
    out["logit_bias"] = None
    out["dry_multiplier"] = 0.0
    out["max_tokens"] = d.get("max_tokens") or 32768

    for key in SAMPLING_KEYS:
        value = getattr(req, key, None)
        if value is not None:
            out[key] = value
    if getattr(req, "logit_bias", None):
        out["logit_bias"] = req.logit_bias
    if getattr(req, "dry_multiplier", None) is not None:
        out["dry_multiplier"] = req.dry_multiplier

    out["temperature"] = float(out["temperature"] if out["temperature"] is not None else 1.0)
    out["top_p"] = float(out["top_p"] if out["top_p"] is not None else 1.0)
    out["top_k"] = int(out["top_k"] or 0)
    out["min_p"] = float(out["min_p"] or 0.0)
    out["repetition_penalty"] = float(out["repetition_penalty"] or 1.0)
    out["presence_penalty"] = float(out["presence_penalty"] or 0.0)
    out["frequency_penalty"] = float(out["frequency_penalty"] or 0.0)
    out["temperature_last"] = bool(out["temperature_last"] or False)
    if out["temperature"] < 0:
        raise HTTPException(status_code=400, detail="temperature 不能为负")

    # 生成长度上限的来源很容易搞错，所以记下来供日志显示：
    #   请求里传了 max_tokens / max_completion_tokens → 以请求为准（配置默认值不起作用）
    #   请求里没传 → 用 config.yaml 的 defaults.max_tokens
    #   两种情况都可能再被 defaults.max_tokens_limit 截断（null = 不限制）
    mt = getattr(req, "max_completion_tokens", None) or getattr(req, "max_tokens", None)
    if mt is not None:
        out["max_tokens"] = int(mt)
        source = "请求指定"
    else:
        source = "配置默认"
    limit = d.get("max_tokens_limit")
    if limit and out["max_tokens"] > int(limit):
        out["max_tokens"] = int(limit)
        source += f"，被 max_tokens_limit={int(limit)} 截断"
    if out["max_tokens"] < 1:
        raise HTTPException(status_code=400, detail="max_tokens 必须 >= 1")
    out["max_tokens_source"] = source

    stop = getattr(req, "stop", None)
    if stop:
        stops = [stop] if isinstance(stop, str) else [s for s in stop if s]
    else:
        stops = list(d.get("stop") or [])
    return out, stops


def _extract_image_url(part: dict) -> str:
    """从内容块里取出图片 URL 字符串。

    OpenAI 标准格式是嵌套的：{"type": "image_url", "image_url": {"url": "..."}}，
    但也有客户端直接把字符串放在 image_url / image 里。

    这里必须统一成**字符串**：之前只是原样取出再重新包一层，遇到标准格式就会变成
    {"url": {"url": "..."}}，下游解一层拿到 dict，报
    `AttributeError: 'dict' object has no attribute 'startswith'`（HTTP 500）。
    """
    raw = part.get("image_url")
    if raw is None:
        raw = part.get("image")
    # 容忍多层嵌套（不同客户端的包装习惯不一样）
    for _ in range(4):
        if not isinstance(raw, dict):
            break
        raw = raw.get("url") or raw.get("image_url")
    if not raw:
        raise HTTPException(status_code=400, detail="image_url 内容块缺少 url")
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="image_url.url 必须是字符串")
    return raw


def build_chat_messages(req: ChatCompletionRequest) -> list[dict]:
    """把 OpenAI 消息转成模板可用的 dict 列表。"""
    messages = []
    for msg in req.messages:
        item: dict[str, Any] = {"role": msg.role}
        content = msg.content
        if isinstance(content, list):
            cleaned = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                kind = part.get("type")
                if kind in ("text", "input_text") or "text" in part:
                    cleaned.append({"type": "text", "text": part.get("text", "")})
                elif kind in ("image_url", "image") or "image_url" in part:
                    # 一律存成规范形式 {"image_url": {"url": "<字符串>"}}
                    cleaned.append(
                        {"type": "image_url", "image_url": {"url": _extract_image_url(part)}}
                    )
                elif kind in ("video", "video_url"):
                    raw = part.get("video_url") or part.get("video")
                    if isinstance(raw, dict):
                        raw = raw.get("url")
                    cleaned.append({"type": "video", "video": raw})
            item["content"] = cleaned
        else:
            item["content"] = content if content is not None else ""
        if msg.name:
            item["name"] = msg.name
        if msg.tool_call_id:
            item["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                tc = tc if isinstance(tc, dict) else tc.model_dump(exclude_none=True)
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"value": args}
                calls.append({"type": "function", "function": {"name": fn.get("name"), "arguments": args or {}}})
            item["tool_calls"] = calls
        messages.append(item)
    return messages


def apply_limits(messages: list[dict]) -> None:
    limit = CFG["limits"].get("max_prompt_chars")
    if not limit:
        return
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text") or "")
                    if part.get("type") == "image_url":
                        total += 1024  # 图片按固定预算计入
    if total > int(limit):
        raise HTTPException(status_code=413, detail=f"prompt 过长（{total} > {limit} 字符）")


TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
# 宽松版：不需要 </function> / </tool_call> 收尾。用于被 max_tokens 截断的情况 ——
# 这时严格版一条都匹配不上，调用会整个丢掉（而且已经生成了几十个 token）。
TOOL_CALL_LOOSE_RE = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>(.*?)(?=</function>|</tool_call>|\Z)", re.DOTALL
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def _tool_param_types(tools: Any) -> dict[str, dict[str, str]]:
    """从请求的 tools 里抽出 {函数名: {参数名: 声明的类型}}。

    为什么需要它：模型吐出的参数是**纯文本**，类型信息只存在于 tool schema 里。
    没它就只能猜，而猜错过 —— 模型写 `<parameter=isRegexp>True</parameter>`
    （首字母大写），而 json.loads 只认小写 true，于是 isRegexp 变成字符串 "True"，
    布尔参数传了字符串，按 schema 严格校验的客户端会直接拒掉整个工具调用。
    """
    out: dict[str, dict[str, str]] = {}
    for tool in tools or []:
        fn = (tool or {}).get("function") or {}
        name = fn.get("name")
        props = (fn.get("parameters") or {}).get("properties") or {}
        if not name or not isinstance(props, dict):
            continue
        types: dict[str, str] = {}
        for pname, pspec in props.items():
            if isinstance(pspec, dict) and isinstance(pspec.get("type"), str):
                types[pname] = pspec["type"]
        if types:
            out[name] = types
    return out


_BOOL_TRUE = {"true", "yes", "1"}
_BOOL_FALSE = {"false", "no", "0"}


def _coerce_param(raw: str, declared: str | None) -> Any:
    """把模型给出的参数文本转成合适的 JSON 类型。

    有 tool schema 就按声明的类型转（最可靠）；没有就退回启发式。
    """
    text = raw.strip()
    if declared == "string":
        # schema 说是字符串就老老实实当字符串：否则 "50" 会被 json.loads
        # 变成数字，按 schema 校验的客户端会报参数类型不对。
        return text
    if declared == "boolean":
        low = text.lower()
        if low in _BOOL_TRUE:
            return True
        if low in _BOOL_FALSE:
            return False
        return text
    if declared in ("integer", "number"):
        try:
            return int(text) if declared == "integer" else float(text)
        except ValueError:
            pass
    if declared in ("object", "array"):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
    # 无 schema（或上面没转成）：先让 JSON 解析器做标准转换（true/false/数字/null），
    # 再兜一层大小写不敏感的布尔字面量 —— 模型很爱写成 True/False。
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low == "null":
        return None
    return text


def _build_call(name: str, body: str,
                specs: dict[str, dict[str, str]] | None = None) -> dict:
    """从 <function=NAME> 的**内部文本**构造一个 OpenAI 工具调用。"""
    types = (specs or {}).get(name) or {}
    arguments: dict[str, Any] = {}
    for pm in PARAM_RE.finditer(body):
        pname, pvalue = pm.group(1).strip(), pm.group(2)
        arguments[pname] = _coerce_param(pvalue, types.get(pname))
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def parse_tool_calls(text: str, specs: dict[str, dict[str, str]] | None = None,
                     ) -> tuple[str, list[dict]]:
    """解析 Qwen 的 <tool_call> 格式，返回 (去掉工具调用后的文本, tool_calls)。"""
    calls = [_build_call(m.group(1).strip(), m.group(2), specs)
             for m in TOOL_CALL_RE.finditer(text)]
    if not calls:
        return text, []
    return TOOL_CALL_RE.sub("", text).strip(), calls


def parse_tool_calls_loose(text: str, specs: dict[str, dict[str, str]] | None = None) -> list[dict]:
    """宽松解析：容忍缺少闭合标签（生成被 max_tokens 截断时）。"""
    return [_build_call(m.group(1).strip(), m.group(2), specs)
            for m in TOOL_CALL_LOOSE_RE.finditer(text)]


def _partial_suffix_len(buf: str, tag: str) -> int:
    """buf 结尾有多少个字符「可能是一个还没收完的 tag 开头」。

    例：buf="看这个 <tool" tag="<tool_call>" → 5（"<tool" 是 tag 的前缀，
    后面还可能续成 "<tool_call>"，所以这 5 个字符不能当正文发出去）。
    """
    for n in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:n]):
            return n
    return 0


class ToolCallStreamer:
    """在**流式**输出里识别 <tool_call> 块，转成 OpenAI 的 tool_calls 增量。

    为什么必须有这个类：非流式那条路会调 parse_tool_calls() 把工具调用从正文里
    摘出来，而流式那条路原本只做「推理 / 正文」切分，从来没解析过工具调用 ——
    于是 <tool_call>...</tool_call> 原样当成 content 发给了客户端。
    ★ VS Code Copilot（以及绝大多数客户端）**一律用流式**，所以它们在
      “非流式能用工具、流式不能用”这就是根因。

    做法：边收边扫。看到完整的 <tool_call>…</tool_call> 就立刻发 tool_calls 增量
    （不等 eos，降延迟）；正文照常发出去，只把「可能是 <tool_call> 开头」的
    尾巴留住（最多 len("<tool_call>")-1 = 10 字符），避免把 `<tool` 这种半个
    标签当正文发出去。
    """

    HOLD = len(TOOL_CALL_OPEN) - 1

    def __init__(self, enabled: bool, specs: dict[str, dict[str, str]] | None = None):
        self.enabled = bool(enabled)
        self.specs = specs or {}
        self.buf = ""        # 还没决定去向的文本
        self.in_call = False  # 已经看到 <tool_call>，在等 </tool_call>
        self.index = 0        # OpenAI 要求 tool_calls 的 index 从 0 连续编号

    def feed(self, text: str) -> list[tuple[str, Any]]:
        """喂入一段正文，返回 [(kind, value)]，kind ∈ {"content", "tool_call"}。"""
        if not self.enabled:
            return [("content", text)] if text else []
        if text:
            self.buf += text

        out: list[tuple[str, Any]] = []
        while True:
            if self.in_call:
                end = self.buf.find(TOOL_CALL_CLOSE)
                if end < 0:
                    break                      # 还没收完，等下一段
                end += len(TOOL_CALL_CLOSE)
                block = TOOL_CALL_OPEN + self.buf[:end]
                self.buf = self.buf[end:]
                self.in_call = False
                for call in parse_tool_calls(block, self.specs)[1]:
                    call["index"] = self.index
                    self.index += 1
                    out.append(("tool_call", call))
                continue

            start = self.buf.find(TOOL_CALL_OPEN)
            if start < 0:
                keep = _partial_suffix_len(self.buf, TOOL_CALL_OPEN)
                if len(self.buf) > keep:
                    out.append(("content", self.buf[: len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break

            if start > 0:
                out.append(("content", self.buf[:start]))
            # 丢掉开标签本身，后面的内容由 in_call 分支处理
            self.buf = self.buf[start + len(TOOL_CALL_OPEN):]
            self.in_call = True
        return out

    def flush(self) -> list[tuple[str, Any]]:
        """生成结束时调用。处理没等到闭合标签的残留。"""
        out: list[tuple[str, Any]] = []
        if self.in_call and self.buf.strip():
            # <tool_call> 开了但没等到 </tool_call>（多半被 max_tokens 截断）。
            # 尽量救回来：参数已经收全的那部分还有用，直接丢掉太浪费。
            calls = parse_tool_calls_loose(TOOL_CALL_OPEN + self.buf, self.specs)
            if calls:
                for call in calls:
                    call["index"] = self.index
                    self.index += 1
                    out.append(("tool_call", call))
            else:
                # 连函数名都没收完，没东西可救 —— 但也不能默默丢掉，
                # 原样发出去至少用户/日志里能看到模型到底吐了什么。
                out.append(("content", TOOL_CALL_OPEN + self.buf))
        elif self.buf:
            out.append(("content", self.buf))
        self.buf = ""
        self.in_call = False
        return out


# ===========================================================================
# 核心推理
# ===========================================================================

async def run_generation(
    engine: Engine,
    input_ids,
    embeddings,
    params: dict,
    stops: list[str],
    return_probs: bool,
    top_logprobs: int,
) -> AsyncIterator[dict]:
    """跑一个 Job，逐个产出 ExLlamaV3 的结果字典。"""
    from exllamav3 import AsyncJob

    # 停止条件必须显式把模型的 EOS 一起传进去：exllamav3 的 Job.stop_tokens
    # 默认是空集合，它不会自己去查 config.eos_token_id_list。不传的后果就是模型
    # 一直生成到 max_tokens，输出里滚出大段 <|im_start|><|im_end|> 垃圾。
    # 注意 eos_token_id_list 是构造 Tokenizer 时才被补全的（会合并
    # tokenizer_config.json 和 generation_config.json 里的值），所以要从 config 上取。
    stop_conditions: list[Any] = list(stops or [])
    eos_ids = list(getattr(engine.model.config, "eos_token_id_list", None) or [])
    if not eos_ids and engine.tokenizer.eos_token_id is not None:
        eos_ids = [engine.tokenizer.eos_token_id]
    for tid in eos_ids:
        if tid not in stop_conditions:
            stop_conditions.append(int(tid))

    job = AsyncJob(
        engine.generator,
        input_ids=input_ids,
        max_new_tokens=int(params["max_tokens"]),
        sampler=engine.build_sampler(params),
        seed=params.get("seed"),
        stop_conditions=stop_conditions or None,
        embeddings=embeddings or None,
        return_probs=bool(return_probs),
        return_top_tokens=int(top_logprobs or 0),
        decode_special_tokens=False,
    )
    done = False
    try:
        async for result in job:
            # EOS 帧会带上投机解码的统计（用草稿模型时才有）。累计起来供 /health 换算
            # 接受长度：那个指标比 tok/s 更能说明草稿模型好不好（tok/s 会被
            # 草稿本身的额外计算开销混进去）。
            if result.get("eos") and "accepted_draft_tokens" in result:
                engine.draft_accepted += int(result.get("accepted_draft_tokens") or 0)
                engine.draft_rejected += int(result.get("rejected_draft_tokens") or 0)
                engine.draft_requests += 1
            yield result
            if result.get("eos"):
                done = True
    finally:
        if not done:
            try:
                await job.cancel()
            except Exception:
                logger.debug("取消任务时出错", exc_info=True)


def _decode_token(engine: Engine, token_id: int) -> str:
    """把单个 token 解码成可读文本（专供 logprobs 展示用）。

    这里故意开 decode_special_tokens：logprobs 是给人看的诊断信息，
    带 special 才能把结尾的 <|im_end|> 这种 token 显示成字面量。
    关掉的话它会变成空字符串，看着就像凭空多了一个 "" 条目。

    exllamav3 的 decode 签名是 str | list[str]：形状 (1, n) 时返回列表，
    直接当字符串用会抛 "'list' object has no attribute 'encode'"。
    """
    import torch

    for special in (True, False):
        try:
            out = engine.tokenizer.decode(torch.tensor([[token_id]]), decode_special_tokens=special)
        except Exception:
            continue
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        if out:
            return out
    return ""


def _merge_prob_frames(frames: list[dict]) -> dict | None:
    """把多帧里的概率字段拼成一份完整的 logprobs 数据源。

    exllamav3 是逐帧吐 token_ids / token_probs / top_k_* 的，EOS 那一帧还会把它们
    塞进 held 子字典（见 job.py 的 emit 分支）。只取最后一帧的话，logprobs 里就只会
    剩最后一个 token，所以这里把所有帧都收集起来再沿序列维拼接。
    """
    import torch

    buckets: dict[str, list[Any]] = {}
    for raw in frames:
        if not isinstance(raw, dict):
            continue
        held = raw.get("held") if isinstance(raw.get("held"), dict) else {}
        for key in ("token_ids", "token_probs", "top_k_tokens", "top_k_probs"):
            val = raw.get(key) if raw.get(key) is not None else held.get(key)
            if val is not None:
                buckets.setdefault(key, []).append(val)

    if not buckets.get("token_ids") or not buckets.get("token_probs"):
        return None

    merged: dict[str, Any] = {}
    for key, parts in buckets.items():
        if len(parts) == 1:
            merged[key] = parts[0]
            continue
        try:
            merged[key] = torch.cat(parts, dim=1)
        except Exception:
            # 形状对不上（例如某些帧缺 top_k）时退回到最后一帧，至少不至于整个失败
            merged[key] = parts[-1]
    return merged


def build_logprobs(engine: Engine, src: dict, with_top: bool) -> dict | None:
    """把 ExLlamaV3 的张量转成 OpenAI 的 logprobs 结构。

    src 由 _merge_prob_frames 拼好（顶层字段名与 exllamav3 一致）。
    """
    import math

    token_ids = src.get("token_ids")
    token_probs = src.get("token_probs")
    if token_ids is None or token_probs is None:
        return None
    ids = token_ids[0].tolist()
    probs = token_probs[0].tolist()
    top_k_tokens = src.get("top_k_tokens") if with_top else None
    top_k_probs = src.get("top_k_probs") if with_top else None

    content = []
    for i, (tid, prob) in enumerate(zip(ids, probs)):
        token = _decode_token(engine, tid)
        entry: dict[str, Any] = {
            "token": token,
            "logprob": math.log(max(float(prob), 1e-12)),
            "bytes": list(token.encode("utf-8")),
            "top_logprobs": [],
        }
        if top_k_tokens is not None and top_k_probs is not None and i < top_k_tokens.shape[1]:
            for tid2, prob2 in zip(top_k_tokens[0, i].tolist(), top_k_probs[0, i].tolist()):
                tok2 = _decode_token(engine, tid2)
                entry["top_logprobs"].append(
                    {
                        "token": tok2,
                        "logprob": math.log(max(float(prob2), 1e-12)),
                        "bytes": list(tok2.encode("utf-8")),
                    }
                )
        content.append(entry)
    return {"content": content}


def _prepare(engine: Engine, req, is_chat: bool):
    """公共准备步骤：解析参数 → 构造 prompt / input_ids。"""
    params, stops = resolve_params(engine, req)

    if is_chat:
        messages = build_chat_messages(req)
        apply_limits(messages)
        messages, embeddings = engine.prepare_images(messages)

        template_vars = dict(engine.tcfg.get("vars") or {})
        for key in ("enable_thinking", "reasoning_effort", "preserve_thinking"):
            value = getattr(req, key, None)
            if value is not None:
                template_vars[key] = value
        effort = template_vars.get("reasoning_effort")
        if effort is not None and effort not in REASONING_EFFORTS:
            raise HTTPException(
                status_code=400,
                detail=f"reasoning_effort 只能是 {', '.join(REASONING_EFFORTS)} 之一（收到 {effort!r}）",
            )
        if req.tools:
            template_vars["tools"] = req.tools

        prompt, start_in_reasoning = engine.render(messages, template_vars)
        input_ids = engine.tokenizer.encode(
            prompt, add_bos=False, encode_special_tokens=True, embeddings=embeddings or None
        )
    else:
        prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        if not prompts:
            raise HTTPException(status_code=400, detail="prompt 不能为空")
        input_ids = engine.tokenizer.encode(prompts[0], add_bos=False, encode_special_tokens=True)
        embeddings = []
        start_in_reasoning = False

    prompt_tokens = int(input_ids.shape[-1])
    # 必须给投机解码的草稿窗口也留位置：job 内部用的预算是
    #   max_new_tokens + 1 + num_draft_tokens
    # 不留的话 prompt + max_tokens 接近 max_seq_len 时会直接 assert 失败。
    draft_tokens = int(getattr(engine, "num_draft_tokens", 0) or 0)
    reserved = RESERVED_TOKENS + draft_tokens
    if prompt_tokens + params["max_tokens"] + reserved > engine.max_seq_len:
        allowed = engine.max_seq_len - prompt_tokens - reserved
        if allowed < 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"prompt 长度 {prompt_tokens} 已超出上下文上限 {engine.max_seq_len}"
                    f"（还要预留 {reserved} 个 token 给生成与草稿窗口）"
                ),
            )
        logger.warning(
            "max_tokens 收窄为 %d（prompt %d + 生成 %d + 预留 %d 需 <= %d）",
            allowed, prompt_tokens, params["max_tokens"], reserved, engine.max_seq_len,
        )
        params["max_tokens"] = allowed
        params["max_tokens_source"] += "，再被上下文上限收窄"

    # 每个请求一行，把「总上下文预算到底怎么花掉的」写清楚。
    # 问「为什么只生成了这么点」时，先看这行：多半是客户端自己传了小的 max_tokens，
    # 而不是服务端默认值或上下文不够。
    logger.info(
        "本请求：prompt %d + 生成上限 %d（%s）+ 预留 %d = 占满 %d / 上下文 %d",
        prompt_tokens,
        params["max_tokens"],
        params.get("max_tokens_source", "?"),
        reserved,
        prompt_tokens + params["max_tokens"] + reserved,
        engine.max_seq_len,
    )

    return params, stops, input_ids, embeddings, prompt_tokens, start_in_reasoning


# ===========================================================================
# 路由：chat completions
# ===========================================================================

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: None = Depends(_auth)):
    engine = get_engine()
    if req.n and int(req.n) > 1:
        raise HTTPException(status_code=400, detail="暂不支持 n > 1")

    params, stops, input_ids, embeddings, prompt_tokens, start_in_reasoning = await asyncio.to_thread(
        _prepare, engine, req, True
    )

    want_logprobs = bool(req.logprobs) or int(req.top_logprobs or 0) > 0
    top_logprobs = int(req.top_logprobs or 0)
    split_reasoning = bool(engine.tcfg.get("split_reasoning", True))
    include_usage = bool(
        (req.stream_options or {}).get("include_usage", engine.dcfg.get("include_usage", True))
    )
    model_name = req.model or engine.name
    created = int(time.time())
    cid = _new_id()
    # 工具参数的声明类型。用它来正确转换参数值（尤其是布尔/数字），
    # 否则模型写 True（首字母大写）会被当成字符串 "True" 传出去。
    tool_specs = _tool_param_types(req.tools)

    async def events() -> AsyncIterator[dict]:
        async with SEMAPHORE:
            async for raw in run_generation(
                engine, input_ids, embeddings, params, stops, want_logprobs, top_logprobs
            ):
                yield raw

    # ---------------- 流式 ----------------
    if req.stream:

        async def sse_gen() -> AsyncIterator[str]:
            splitter = ThinkSplitter(split_reasoning, start_in_reasoning)
            # 流式也必须解析工具调用，否则 <tool_call> 会原样当成正文发给客户端。
            # ★ VS Code Copilot 一律走流式，所以这一行就是它能不能用工具的关键。
            tc_stream = ToolCallStreamer(bool(req.tools), tool_specs)
            completion_tokens = 0
            finish = "stop"
            saw_tool_call = False

            def chunk_sse(delta: dict, fin: str | None = None) -> str:
                return _sse(
                    json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": fin}],
                        },
                        ensure_ascii=False,
                    )
                )

            def emit_call(call: dict) -> Iterator[str]:
                """把解析出的调用变成 OpenAI 流式 tool_calls 增量。

                分两帧发（先 id/name，再 arguments），与 vLLM / llama.cpp 一致：
                不少客户端看到第一帧就把 index 槽位建好并记下函数名，把 name 和
                arguments 塞在同一帧里反而可能漏掉名字。
                """
                yield chunk_sse({
                    "tool_calls": [{
                        "index": call["index"],
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": ""},
                    }]
                })
                yield chunk_sse({
                    "tool_calls": [{
                        "index": call["index"],
                        "function": {"arguments": call["function"]["arguments"]},
                    }]
                })

            yield chunk_sse({"role": "assistant", "content": ""})
            try:
                async for raw in events():
                    pieces: list[tuple[str, str]] = []
                    if raw.get("text"):
                        pieces += splitter.feed(raw["text"])
                    is_eos = bool(raw.get("eos"))
                    if is_eos:
                        pieces += splitter.flush()
                        completion_tokens = int(raw.get("new_tokens") or 0)
                        finish = _finish_reason(raw.get("eos_reason"))

                    for kind, chunk in pieces:
                        if not chunk:
                            continue
                        if kind == "reasoning":
                            # 推理段不参与工具解析：工具调用按模板约定出现在正文里，
                            # 而且 reasoning_content 是已经流出去的，没法再撤回。
                            yield chunk_sse({"reasoning_content": chunk})
                            continue
                        for ev, value in tc_stream.feed(chunk):
                            if ev == "content":
                                yield chunk_sse({"content": value})
                            else:
                                saw_tool_call = True
                                for frame in emit_call(value):
                                    yield frame

                    if is_eos:
                        # 切分器放完之后，工具解析器里可能还有残留
                        # （比如 <tool_call> 开了但没等到 </tool_call>）
                        for ev, value in tc_stream.flush():
                            if ev == "content":
                                yield chunk_sse({"content": value})
                            else:
                                saw_tool_call = True
                                for frame in emit_call(value):
                                    yield frame
                        if saw_tool_call:
                            # 必须有这个 finish_reason，客户端才会去执行工具；
                            # 报 "stop" 的话客户端会当成普通回答结束。
                            finish = "tool_calls"
                        yield chunk_sse({}, finish)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("流式生成失败")
                yield _sse(
                    json.dumps(
                        {"id": cid, "object": "chat.completion.chunk", "error": {"message": str(exc)}},
                        ensure_ascii=False,
                    )
                )
            if include_usage:
                yield _sse(
                    json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": _usage(prompt_tokens, completion_tokens),
                        },
                        ensure_ascii=False,
                    )
                )
            yield _sse("[DONE]")

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # ---------------- 非流式 ----------------
    splitter = ThinkSplitter(split_reasoning, start_in_reasoning)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    completion_tokens = 0
    finish = "stop"
    prob_frames: list[dict] = []
    async for raw in events():
        if want_logprobs:
            prob_frames.append(raw)
        if raw.get("text"):
            for kind, chunk in splitter.feed(raw["text"]):
                (reasoning_parts if kind == "reasoning" else content_parts).append(chunk)
        if raw.get("eos"):
            for kind, chunk in splitter.flush():
                (reasoning_parts if kind == "reasoning" else content_parts).append(chunk)
            completion_tokens = int(raw.get("new_tokens") or 0)
            finish = _finish_reason(raw.get("eos_reason"))

    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)

    tool_calls: list[dict] = []
    if req.tools:
        content, tool_calls = parse_tool_calls(content, tool_specs)
        if not tool_calls and TOOL_CALL_OPEN in content:
            # 严格解析一条都没匹配上，但又确实看到了 <tool_call>：
            # 典型是被 max_tokens 截断，少了 </function>/</tool_call>。
            # 用宽松解析尽量救回来，比把整段 XML 当正文返回好得多。
            idx = content.find(TOOL_CALL_OPEN)
            rescued = parse_tool_calls_loose(content[idx:], tool_specs)
            if rescued:
                tool_calls = rescued
                content = content[:idx].strip()
        if tool_calls:
            finish = "tool_calls"

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if split_reasoning and reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls

    body: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": _usage(prompt_tokens, completion_tokens),
    }
    if want_logprobs:
        src = _merge_prob_frames(prob_frames)
        lp = build_logprobs(engine, src, top_logprobs > 0) if src else None
        if lp:
            body["choices"][0]["logprobs"] = lp
    return JSONResponse(body)


# ===========================================================================
# 路由：legacy completions
# ===========================================================================

@app.post("/v1/completions")
async def completions(req: CompletionRequest, _: None = Depends(_auth)):
    engine = get_engine()
    params, stops, input_ids, _, prompt_tokens, _ = await asyncio.to_thread(_prepare, engine, req, False)

    want_logprobs = bool(req.logprobs) or int(req.top_logprobs or 0) > 0
    top_logprobs = int(req.top_logprobs or 0)
    model_name = req.model or engine.name
    created = int(time.time())
    cid = "cmpl-" + uuid.uuid4().hex

    async def events() -> AsyncIterator[dict]:
        async with SEMAPHORE:
            async for raw in run_generation(engine, input_ids, None, params, stops, want_logprobs, top_logprobs):
                yield raw

    if req.stream:

        async def sse_gen() -> AsyncIterator[str]:
            try:
                async for raw in events():
                    eos = bool(raw.get("eos"))
                    text = raw.get("text") or ""
                    if not text and not eos:
                        continue
                    yield _sse(
                        json.dumps(
                            {
                                "id": cid,
                                "object": "text_completion",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": text,
                                        "logprobs": None,
                                        "finish_reason": _finish_reason(raw.get("eos_reason")) if eos else None,
                                    }
                                ],
                            },
                            ensure_ascii=False,
                        )
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("流式补全失败")
                yield _sse(json.dumps({"error": {"message": str(exc)}}, ensure_ascii=False))
            yield _sse("[DONE]")

        return StreamingResponse(
            sse_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text_parts: list[str] = []
    completion_tokens = 0
    finish = "stop"
    prob_frames: list[dict] = []
    async for raw in events():
        if want_logprobs:
            prob_frames.append(raw)
        if raw.get("text"):
            text_parts.append(raw["text"])
        if raw.get("eos"):
            completion_tokens = int(raw.get("new_tokens") or 0)
            finish = _finish_reason(raw.get("eos_reason"))

    body: dict[str, Any] = {
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "text": "".join(text_parts), "logprobs": None, "finish_reason": finish}],
        "usage": _usage(prompt_tokens, completion_tokens),
    }
    if want_logprobs:
        src = _merge_prob_frames(prob_frames)
        lp = build_logprobs(engine, src, top_logprobs > 0) if src else None
        if lp:
            body["choices"][0]["logprobs"] = lp
    return JSONResponse(body)


# ===========================================================================
# 路由：models / health
# ===========================================================================

@app.get("/v1/models")
async def list_models(_: None = Depends(_auth)):
    name = ENGINE.name if ENGINE else os.path.basename(str(CFG["model"]["path"]).rstrip("/\\"))
    return {"object": "list", "data": [{"id": name, "object": "model", "created": int(time.time()), "owned_by": "local"}]}


@app.get("/health")
@app.get("/v1/health")
async def health():
    if ENGINE is None:
        return JSONResponse(
            {"status": "error" if LOAD_ERROR else "loading", "detail": LOAD_ERROR}, status_code=503
        )
    info: dict[str, Any] = {
        "status": "ok",
        "model": ENGINE.name,
        "max_seq_len": ENGINE.max_seq_len,
        "cache_quant": CFG["model"].get("cache_quant"),
        # 单独暴露生效的 K/V 位数：cache_quant 只是缺省值，真正生效的是
        # cache_k_bits / cache_v_bits（非对称量化时会和 cache_quant 不一样）
        "cache_k_bits": ENGINE.cache_k_bits,
        "cache_v_bits": ENGINE.cache_v_bits,
        # 请求没传 max_tokens 时用的默认值，方便不开日志就能确认生成长度上限
        "max_tokens": CFG["defaults"].get("max_tokens"),
        "max_tokens_limit": CFG["defaults"].get("max_tokens_limit"),
        "tensor_parallel": CFG["model"]["tensor_parallel"],
        "vision": ENGINE.vision_model is not None,
        "mtp_draft": ENGINE.draft_model is not None,
        "draft_kind": ENGINE.draft_kind,
        "draft_tokens_per_window": ENGINE.num_draft_tokens,
        "draft_accepted_tokens": ENGINE.draft_accepted,
        "draft_rejected_tokens": ENGINE.draft_rejected,
        "draft_requests": ENGINE.draft_requests,
        "pid": os.getpid(),
        "uptime_s": round(time.time() - STARTED_AT, 1),
    }
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(idx),
                "vram_free_gb": round(free / 1024**3, 2),
                "vram_total_gb": round(total / 1024**3, 2),
            }
    except Exception:
        pass
    return info


# ===========================================================================
# 路由：管理（供 control_panel.py 调用）
# ===========================================================================

def _is_loopback(request: Request) -> bool:
    client = request.client
    host = (client.host if client else "") or ""
    # 兼容 IPv4 / IPv6 回环写法
    return host in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1")


@app.post("/admin/shutdown")
async def admin_shutdown(request: Request):
    """优雅退出。

    只允许回环地址调用：本服务默认监听 0.0.0.0，如果放开的话同局域网任何机器
    都能把服务关掉。控制面板和推理服务在同一台机器上，走 127.0.0.1 即可。
    """
    if not CFG["server"].get("allow_shutdown", True):
        raise HTTPException(status_code=403, detail="配置里已禁用远程关机（server.allow_shutdown: false）")
    if not _is_loopback(request):
        raise HTTPException(status_code=403, detail="只允许从本机（回环地址）调用")
    if SERVER is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    logger.info("收到关机请求，正在优雅退出（会先释放显存）…")
    SERVER.should_exit = True
    return {"status": "shutting_down", "pid": os.getpid()}


# ===========================================================================
# 入口
# ===========================================================================

def main() -> int:
    global CFG, SERVER, STARTED_AT
    ap = argparse.ArgumentParser(description="Qwen3.8-27B EXL3 OpenAI 兼容服务")
    ap.add_argument("--config", "-c", default=os.environ.get("QWEN38_CONFIG", "config.yaml"), help="配置文件路径")
    ap.add_argument("--host", default=None, help="覆盖监听地址")
    ap.add_argument("--port", type=int, default=None, help="覆盖监听端口")
    ap.add_argument("--model-path", default=None, help="覆盖模型目录")
    ap.add_argument("--max-seq-len", type=int, default=None, help="覆盖上下文长度")
    ap.add_argument("--no-vision", action="store_true", help="不加载视觉塔（省显存）")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    CFG = load_config(args.config)
    if args.host:
        CFG["server"]["host"] = args.host
    if args.port:
        CFG["server"]["port"] = args.port
    if args.model_path:
        CFG["model"]["path"] = args.model_path
    if args.max_seq_len:
        CFG["model"]["max_seq_len"] = args.max_seq_len
    if args.no_vision:
        CFG["model"]["load_vision"] = False

    # CORS 中间件在 import 时就按当时的配置建好了；如果 --config 指定了别的文件，
    # 这里同步一次，避免改了 cors_origins 不生效
    origins = CFG["server"].get("cors_origins") or ["*"]
    for middleware in app.user_middleware:
        if middleware.cls is CORSMiddleware:
            middleware.kwargs["allow_origins"] = origins
            break

    # 把生效的关键参数打出来，方便确认到底跑的是哪套配置
    mc, dc, lc = CFG["model"], CFG["defaults"], CFG["limits"]
    bits = int(mc.get("cache_quant") or 16)
    # 真正生效的是 cache_k_bits / cache_v_bits（非对称量化时和 cache_quant 不一样），
    # 所以显存估算必须按 K/V 各自的位数加权算，直接拿 cache_quant 会算错
    # （k8v4 会被算成 4.00 GiB，实际是 6.00 GiB）。
    k_bits = bits if bits >= 16 else int(mc.get("cache_k_bits") or bits)
    v_bits = bits if bits >= 16 else int(mc.get("cache_v_bits") or bits)
    if bits >= 16:
        kv_gib = mc["max_seq_len"] * ELEMS_PER_TOKEN * 2 / 1024**3
        kv_label = "fp16"
    else:
        kv_gib = mc["max_seq_len"] * ELEMS_PER_TOKEN * ((k_bits + v_bits) / 2) / 8 / 1024**3
        kv_label = f"K{k_bits}bit/V{v_bits}bit" if k_bits != v_bits else f"{k_bits}bit"
    logger.info("生效配置：")
    logger.info("  模型目录     : %s", os.path.abspath(mc["path"]))
    logger.info("  上下文预算   : %d tokens（并发 %d 路共享）", mc["max_seq_len"], mc["max_batch_size"])
    logger.info("  KV 量化      : %s → 约 %.2f GiB", kv_label, kv_gib)
    logger.info("  视觉塔 / MTP : %s / %s", mc.get("load_vision"), mc.get("mtp_draft"))
    logger.info("  递归层历史   : %s", mc.get("max_history") if mc.get("max_history") is not None else "自动（跟随草稿长度）")
    logger.info("  默认采样     : temp=%s top_p=%s top_k=%s min_p=%s rep=%s",
                dc.get("temperature"), dc.get("top_p"), dc.get("top_k"), dc.get("min_p"), dc.get("repetition_penalty"))
    logger.info("  输出上限     : max_tokens=%s, 硬上限=%s", dc.get("max_tokens"), dc.get("max_tokens_limit") or "无")
    logger.info("  并发请求上限 : %s", lc.get("max_concurrent_requests"))
    if mc["max_seq_len"] > 131072 and (k_bits + v_bits) / 2 > 4:
        logger.warning(
            "上下文 %d + KV %s（均 %.1f bit）在 24GB 卡上余量很小，"
            "容易因显存不足掉进页文件而急剧变慢，建议降到 K4/V4 或调小 max_seq_len",
            mc["max_seq_len"], kv_label, (k_bits + v_bits) / 2,
        )

    import uvicorn

    STARTED_AT = time.time()
    logger.info("监听 http://%s:%s", CFG["server"]["host"], CFG["server"]["port"])
    # 用显式的 Server 实例而不是 uvicorn.run()：这样 /admin/shutdown 才能拿到
    # 它并置 should_exit，走完整的 lifespan 关闭流程（释放显存），
    # 而不是被控制面板硬杀进程。
    config = uvicorn.Config(
        app,
        host=CFG["server"]["host"],
        port=int(CFG["server"]["port"]),
        log_level=args.log_level.lower(),
        timeout_keep_alive=75,
    )
    SERVER = uvicorn.Server(config)
    SERVER.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
