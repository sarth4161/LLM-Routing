# llm.py - step 6: call a model tier via LiteLLM (or fake it in MOCK mode),
# compute cost from prices.yaml, retry once, then fall back to the next bigger tier.
# Also holds the prompt templates, "Final answer:" parsing and deterministic grading.
import hashlib
import random
import re
import time

import yaml

from backend import config

PROMPTS = {
    "mcq": "Answer this multiple-choice question. Think briefly, then end with a line "
           "exactly like 'Final answer: X' where X is one letter A, B, C or D.\n\n{query}",
    "math": "Solve this math problem step by step, then end with a line exactly like "
            "'Final answer: N' where N is a single number (no units).\n\n{query}",
    "text": "Answer the question concisely, then end with a line exactly like "
            "'Final answer: <your short answer>'.\n\n{query}",
}

_prices = None


def load_prices():
    global _prices
    if _prices is None:
        with open(config.PRICES_YAML) as f:
            _prices = yaml.safe_load(f)
    return _prices


def cost_of(tier, input_tokens, output_tokens):
    p = load_prices()[tier]
    return (input_tokens * p["input_per_1m"] + output_tokens * p["output_per_1m"]) / 1e6


def extract_final_answer(text, task_type):
    """Return the normalized final answer, or None if missing/invalid."""
    found = re.findall(r"final answer\s*:\s*(.+)", text or "", flags=re.I)
    if not found:
        return None
    ans = found[-1].strip().strip("*`. ")
    if task_type == "mcq":
        m = re.match(r"^\(?([A-Da-d])\b", ans)
        return m.group(1).upper() if m else None
    if task_type == "math":
        m = re.search(r"-?\d[\d,]*\.?\d*", ans.replace("$", ""))
        if not m:
            return None
        num = float(m.group().replace(",", "").rstrip("."))
        return str(int(num)) if num == int(num) else str(round(num, 4))
    return ans.lower() or None


def is_correct(answer, reference, task_type):
    """Deterministic grading: letter match, number match (small tolerance), or text match."""
    if answer is None or reference is None:
        return False
    ref = str(reference)
    if task_type == "mcq":
        return answer == extract_final_answer("Final answer: " + ref, "mcq")
    if task_type == "math":
        ref_num = extract_final_answer("Final answer: " + ref, "math")
        try:
            return abs(float(answer) - float(ref_num)) < 1e-4
        except (TypeError, ValueError):
            return False
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
    return norm(answer) == norm(ref)


def _mock_text(tier, query, task_type, reference, temperature):
    """Fake model output: correct with the tier's accuracy, using the reference answer.
    temperature 0 is deterministic per (tier, query) so reruns and setup.py agree."""
    seed = None if temperature > 0 else int(hashlib.md5(f"{tier}|{query}".encode()).hexdigest(), 16)
    rng = random.Random(seed)
    correct = reference is not None and rng.random() < config.MOCK_ACCURACY[tier]
    if task_type == "mcq":
        ref = extract_final_answer(f"Final answer: {reference}", "mcq") or "A"
        ans = ref if correct else rng.choice([c for c in "ABCD" if c != ref])
    elif task_type == "math":
        ref = extract_final_answer(f"Final answer: {reference}", "math") or "0"
        ans = ref if correct else str(int(float(ref)) + rng.choice([-10, -2, -1, 1, 3, 12]))
    else:
        ans = str(reference) if correct else "mock wrong answer"
    hedge = "" if correct or rng.random() < 0.6 else "I think maybe this is right. "
    return f"[mock {tier}] {hedge}Reasoning...\nFinal answer: {ans}", rng


def call_model(tier, query, task_type="text", reference=None, temperature=0.0):
    """One call to one tier. Returns a dict; never raises (ok=False on error)."""
    prompt = PROMPTS.get(task_type, PROMPTS["text"]).format(query=query)
    model = load_prices()[tier]["model"]
    start = time.perf_counter()
    try:
        if config.MOCK_LLM:
            text, rng = _mock_text(tier, query, task_type, reference, temperature)
            in_tok = len(prompt) // 4 + 10
            out_tok = rng.randint(30, 120) * (1 + config.TIERS.index(tier))
            ms = rng.uniform(300, 700) * (1 + 1.5 * config.TIERS.index(tier))  # fake latency
        else:
            import litellm
            resp = litellm.completion(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=config.MAX_OUTPUT_TOKENS,
                timeout=config.REQUEST_TIMEOUT_S)
            text = resp.choices[0].message.content or ""
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
            ms = (time.perf_counter() - start) * 1000
        return {"tier": tier, "model": model, "text": text,
                "answer": extract_final_answer(text, task_type),
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cost": cost_of(tier, in_tok, out_tok), "ms": round(ms, 1), "ok": True}
    except Exception as e:
        return {"tier": tier, "model": model, "text": "", "answer": None, "error": str(e),
                "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                "ms": round((time.perf_counter() - start) * 1000, 1), "ok": False}


def call_with_fallback(tier, query, task_type="text", reference=None, temperature=0.0):
    """Try `tier`, retry once, then move up a tier. Returns (final_result, all_calls)."""
    calls = []
    for t in config.TIERS[config.TIERS.index(tier):]:
        for _ in range(2):
            res = call_model(t, query, task_type, reference, temperature)
            calls.append(res)
            if res["ok"]:
                return res, calls
    return calls[-1], calls


if __name__ == "__main__":
    # Quick smoke test: python -m backend.llm
    from backend import db
    db.init_db(load_prices())
    print("MOCK_LLM =", config.MOCK_LLM)
    q = "What is the capital of France?\nA) Berlin\nB) Paris\nC) Rome\nD) Madrid"
    for tier in config.TIERS:
        r, calls = call_with_fallback(tier, q, "mcq", reference="B")
        print(f"{tier:6} ok={r['ok']} answer={r['answer']} correct={is_correct(r['answer'], 'B', 'mcq')} "
              f"tokens={r['input_tokens']}+{r['output_tokens']} cost={r['cost']:.6f} ms={r['ms']} "
              f"calls={len(calls)} {r.get('error', '')}")
