"""Reusable local HuggingFace pipeline wrapper."""

from __future__ import annotations

import ast
import copy
import json
import os
from contextlib import contextmanager
from json import JSONDecodeError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

@contextmanager
def _allow_huggingface_download():
    """Enable Hub access only while a missing model snapshot is downloaded."""
    offline_variables = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {
        name: os.environ.get(name)
        for name in offline_variables
    }
    for name in offline_variables:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _build_generation_config(
    base_config: Any,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> Any:
    """Return one self-contained generation config for a pipeline call.

    """
    generation_config = copy.deepcopy(base_config)
    generation_config.max_length = None
    generation_config.max_new_tokens = int(max_new_tokens)
    generation_config.do_sample = bool(do_sample and temperature > 0.0)
    generation_config.temperature = (
        float(temperature) if generation_config.do_sample else None
    )
    return generation_config


@contextmanager
def _allow_nondeterministic_torch_operations():
    """Temporarily allow Qwen CUDA kernels unsupported by deterministic mode."""
    import torch

    previous_debug_mode = torch.get_deterministic_debug_mode()
    if previous_debug_mode == 0:
        yield
        return

    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.set_deterministic_debug_mode(previous_debug_mode)


def _extract_json_block(text: str) -> Dict[str, Any]:
    """Extract first structured object from text with resilient parsing."""
    text = text.strip()
    if not text:
        raise ValueError("Empty model output")

    decoder = json.JSONDecoder()

    # Fast path: whole output is valid JSON object.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Robust path: scan from each "{" and attempt raw JSON decode.
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except JSONDecodeError:
            continue

    # Last resort: parse Python-literal style dict payload from first '{' to last '}'.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Also support outputs where model returns a Python-like list of dicts.
    list_start = text.find("[")
    list_end = text.rfind("]")
    if list_start != -1 and list_end != -1 and list_end > list_start:
        list_candidate = text[list_start : list_end + 1]
        try:
            parsed_list = ast.literal_eval(list_candidate)
            if isinstance(parsed_list, list):
                for item in parsed_list:
                    if isinstance(item, dict):
                        return item
        except Exception:
            pass

    raise ValueError("No parseable JSON object found in model output")


@dataclass
class HFLocalLLM:
    """Local text generation helper with deterministic settings."""

    model_name: str
    max_new_tokens: int = 256
    max_input_tokens: int = 40000
    temperature: float = 0.0
    do_sample: bool = False
    device: int = -1
    verbose: bool = False
    preview_chars: int = 1400

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        _ = limit
        return text.strip()

    @staticmethod
    def _resolve_local_model_path(model_name: str) -> str:
        """Return a cached snapshot, downloading it when it is absent."""
        cache_dir = HFLocalLLM._huggingface_cache_dir()
        model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.is_dir():
            snapshots = sorted(
                path for path in snapshots_dir.iterdir() if path.is_dir()
            )
            if snapshots:
                return str(snapshots[-1])
        return HFLocalLLM._download_model_snapshot(model_name, cache_dir)

    @staticmethod
    def _huggingface_cache_dir() -> Path:
        cache_override = os.environ.get("HF_HUB_CACHE")
        if cache_override:
            return Path(cache_override).expanduser()
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home).expanduser() / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    @staticmethod
    def _download_model_snapshot(model_name: str, cache_dir: Path) -> str:
        """Download one Hub snapshot into the local cache and return its path."""
        print(
            f"[LLM] Model '{model_name}' is not cached; downloading it to "
            f"{cache_dir}."
        )
        try:
            with _allow_huggingface_download():
                from huggingface_hub import snapshot_download

                snapshot_path = snapshot_download(
                    repo_id=model_name,
                    cache_dir=str(cache_dir),
                )
        except Exception as exc:
            raise FileNotFoundError(
                f"Model '{model_name}' is unavailable locally and automatic "
                "download failed. Check network access and Hugging Face "
                "credentials for gated repositories."
            ) from exc
        return str(Path(snapshot_path))

    def __post_init__(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )

        self._call_counter = 0
        self._generation_attempt_counter = 0
        self._input_token_counter = 0
        self._output_token_counter = 0
        local_path = self._resolve_local_model_path(self.model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            local_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        device = self.device

        # Automatic placement can distribute the model across available GPUs and CPU.
        if isinstance(device, str) and device.lower() in {
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        }:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16,
                device_map=device.lower(),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=True,
            )

        # CPU placement uses full precision.
        elif str(device).lower() in {"cpu", "-1"}:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=True,
            )

        # An integer selects one GPU and enables 4-bit quantization.
        elif isinstance(device, int):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                quantization_config=bnb_config,
                device_map={"": device},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                local_files_only=True,
            )

        else:
            raise ValueError(
                f"Invalid device: {device!r}. Use an integer such as 0, 'auto', or 'cpu'."
            )

        self.context_window = min(
            getattr(
                model.config, "max_position_embeddings", tokenizer.model_max_length
            ),
            tokenizer.model_max_length,
        )

        self._pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        self._base_generation_config = copy.deepcopy(self._pipe.generation_config)

    def _render_chat_input(self, system_prompt: str, user_prompt: str) -> str:
        """Render chat messages exactly as ``generate_json`` will send them."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self._pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self._pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return "/no_think\n" + rendered


    @property
    def input_token_budget(self) -> int:
        """Maximum input size that leaves room for generation and safety tokens."""
        safe_context_budget = self.context_window - self.max_new_tokens - 256
        if self.max_input_tokens is None:
            return int(safe_context_budget)
        return int(min(self.max_input_tokens, safe_context_budget))

    def count_chat_tokens(self, system_prompt: str, user_prompt: str) -> int:
        """Count a chat prompt with the same template used for generation."""
        pipe_input = self._render_chat_input(system_prompt, user_prompt)
        return len(self._pipe.tokenizer(pipe_input)["input_ids"])

    def usage_metrics(self) -> Dict[str, int]:
        """Return cumulative resource counters for the current experiment."""
        return {
            "llm_calls": int(self._call_counter),
            "generation_attempts": int(self._generation_attempt_counter),
            "input_tokens": int(self._input_token_counter),
            "output_tokens": int(self._output_token_counter),
            "total_tokens": int(self._input_token_counter + self._output_token_counter),
        }

    def generate_json(
        self,
        prompt: str = "",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> Dict[str, Any]:
        """Generate and parse JSON from local model output.

        When *system_prompt* and *user_prompt* are both provided they are sent
        as a chat-formatted messages list so the model's chat template is
        applied automatically.  Otherwise *prompt* is used as a raw string.
        """
        self._call_counter += 1

        if system_prompt is not None and user_prompt is not None:
            pipe_input = self._render_chat_input(system_prompt, user_prompt)

        else:
            pipe_input = "/no_think\n" + prompt

        
        n_input_tokens = len(self._pipe.tokenizer(pipe_input)["input_ids"])
        print("input tokens:", n_input_tokens)
        print("max_new_tokens:", self.max_new_tokens)
        effective_max_input_tokens = self.input_token_budget

        if n_input_tokens > effective_max_input_tokens:
            print(
                f"  > [LLM] Prompt too long: {n_input_tokens} tokens. "
                f"Truncating to {effective_max_input_tokens} tokens."
            )

            encoded = self._pipe.tokenizer(
                pipe_input,
                truncation=True,
                max_length=effective_max_input_tokens,
                return_tensors=None,
            )

            pipe_input = self._pipe.tokenizer.decode(
                encoded["input_ids"],
                skip_special_tokens=False,
            )

            n_input_tokens = len(self._pipe.tokenizer(pipe_input)["input_ids"])
            print("input tokens after truncation:", n_input_tokens)

        self._input_token_counter += int(n_input_tokens)


        import time

        start_t = time.time()

        parsed = {}
        text = ""
        max_attempts = 3
        current_temp = self.temperature

        for attempt in range(max_attempts):
            self._generation_attempt_counter += 1
            generation_config = _build_generation_config(
                self._base_generation_config,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=current_temp,
            )
            with _allow_nondeterministic_torch_operations():
                outputs = self._pipe(
                    pipe_input,
                    generation_config=generation_config,
                    return_full_text=False,
                    clean_up_tokenization_spaces=False,
                )

            raw_output = outputs[0]["generated_text"]
            if isinstance(raw_output, list):
                text = raw_output[-1].get("content", "") if raw_output else ""
            else:
                text = str(raw_output)
            self._output_token_counter += len(
                self._pipe.tokenizer(str(text), add_special_tokens=False)["input_ids"]
            )

            try:
                parsed = _extract_json_block(text)
                if parsed:  # If we got a valid non-empty dict, we're good
                    break
            except Exception as exc:
                if self.verbose:
                    print(
                        f"[LLM] parse warning (attempt {attempt+1}): {type(exc).__name__}: {exc}"
                    )

            # If we reach here, parsing failed or was empty. Retry with lower temperature.
            current_temp = 0.1
            if not self.verbose and attempt < max_attempts - 1:
                print(
                    f"  > [LLM] JSON parsing failed; retrying with temperature={current_temp}..."
                )

        if not self.verbose:
            print(f"  > [LLM] Response received in {time.time() - start_t:.1f}s.")

        if self.verbose:
            print("[LLM] raw output begin")
            print(self._clip(str(text), self.preview_chars))
            print("[LLM] raw output end")

        if self.verbose:
            print("[LLM] parsed json:", json.dumps(parsed, ensure_ascii=True))
        return parsed
