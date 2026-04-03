import atexit
from dataclasses import fields

import torch.multiprocessing as mp
from transformers import AutoTokenizer

from vllm.config import Config
from vllm.engine.model_runner import ModelRunner
from vllm.engine.scheduler import Scheduler
from vllm.engine.sequence import Sequence
from vllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model=model, **config_kwargs)

        self.ps = []
        self.events = []
        self._closed = False

        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True

        if hasattr(self, "model_runner"):
            try:
                self.model_runner.call("exit")
            except Exception:
                pass
            del self.model_runner

        for p in self.ps:
            try:
                p.join()
            except Exception:
                pass

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

        return seq.seq_id

    def is_finished(self):
        return self.scheduler.is_finished()

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)

        step_tokens = []
        for seq, token_id in zip(seqs, token_ids):
            step_tokens.append(
                {
                    "seq_id": seq.seq_id,
                    "token_id": token_id,
                    "completion_token_ids": seq.completion_token_ids[:],
                    "is_finished": seq.is_finished,
                }
            )

        finished = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        return {
            "step_tokens": step_tokens,
            "finished": finished,
            "is_prefill": is_prefill,
        }

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        prompt_seq_ids = []
        outputs = {}

        for prompt, sp in zip(prompts, sampling_params):
            seq_id = self.add_request(prompt, sp)
            prompt_seq_ids.append(seq_id)

        while not self.is_finished():
            out = self.step()
            for seq_id, token_ids in out["finished"]:
                outputs[seq_id] = {
                    "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                    "token_ids": token_ids,
                }

        return [outputs[seq_id] for seq_id in prompt_seq_ids]

    def generate_stream(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ):
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        prompt_seq_ids = []
        finished_set = set()

        for prompt, sp in zip(prompts, sampling_params):
            seq_id = self.add_request(prompt, sp)
            prompt_seq_ids.append(seq_id)

        while not self.is_finished():
            out = self.step()

            for item in out["step_tokens"]:
                seq_id = item["seq_id"]
                token_id = item["token_id"]
                completion_token_ids = item["completion_token_ids"]

                yield {
                    "type": "token",
                    "seq_id": seq_id,
                    "token_id": token_id,
                    "delta_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                    "full_text": self.tokenizer.decode(completion_token_ids, skip_special_tokens=True),
                    "token_ids": completion_token_ids,
                    "is_prefill": out["is_prefill"],
                }

            for seq_id, token_ids in out["finished"]:
                if seq_id not in finished_set:
                    finished_set.add(seq_id)
                    yield {
                        "type": "finished",
                        "seq_id": seq_id,
                        "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                        "token_ids": token_ids,
                    }
