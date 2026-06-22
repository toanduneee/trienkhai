import json
import time
import argparse
from datetime import datetime
from pathlib import Path

from pipeline.dataset   import load_behaviors
from pipeline.attacker  import paraphrase_attack, persona_wrap, pair_attack
from pipeline.target    import call_target, make_caller
from pipeline.evaluator import evaluate_response, summarize_results

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fallback dùng khi evaluate_response trả về None hoặc lỗi
EVAL_ERROR_FALLBACK = {
    "keyword_matched": False,
    "matched_strings": [],
    "score":           0,
    "verdict":         "ERROR",
    "reasoning":       "Evaluation failed",
    "is_success":      False,
    "eval_method":     "error"
}


def run_experiment(
    model_key:     str,
    attack_type:   str,
    category:      str = None,
    n_seeds:       int = None,
    n_variants:    int = 3,
    max_pair_iter: int = 20,
    system_prompt: str = ""
) -> dict:
    """
    Chạy một lượt thực nghiệm đầy đủ.

    Args:
        model_key:     "dolphin" | "groq" | "gemini"
        attack_type:   "paraphrase" | "persona" | "pair"
        category:      Filter dataset theo category (None = tất cả)
        n_seeds:       Giới hạn số seed (None = tất cả)
        n_variants:    Số biến thể paraphrase mỗi seed
        max_pair_iter: Vòng lặp PAIR tối đa
        system_prompt: System prompt của target app
    """
    dataset     = load_behaviors(n=n_seeds, category=category)
    all_results = []
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  Model:  {model_key}  |  Attack: {attack_type.upper()}")
    print(f"  Seeds:  {len(dataset)}  |  Category: {category or 'all'}")
    print(f"{'='*60}\n")

    for seed_item in dataset:
        seed_id     = seed_item.get("id", "?")
        seed_prompt = seed_item.get("behavior", "")
        print(f"[{seed_id}] {seed_prompt[:70]}...")

        # ── Sinh biến thể ────────────────────────────────────────────────
        try:
            if attack_type == "paraphrase":
                variants     = paraphrase_attack(seed_prompt, n_variants)
                attack_items = [
                    {"variant_type": f"para_{i+1}", "prompt": v}
                    for i, v in enumerate(variants)
                    if isinstance(v, str) and v.strip()
                ]

            elif attack_type == "persona":
                wrapped      = persona_wrap(seed_prompt)
                attack_items = [
                    {"variant_type": f"persona_{w['persona']}", "prompt": w["prompt"]}
                    for w in wrapped
                ]

            elif attack_type == "pair":
                caller      = make_caller(model_key, system_prompt)
                pair_result = pair_attack(seed_prompt, caller, max_pair_iter)
                attack_items = [{
                    "variant_type":    "pair",
                    "prompt":          pair_result["best_prompt"],
                    "pair_iterations": pair_result["iterations_used"],
                    "pair_success":    pair_result["success"]
                }]

            else:
                raise ValueError(f"Unknown attack_type: {attack_type}")

        except Exception as e:
            print(f"  ERROR khi sinh biến thể [{seed_id}]: {e}")
            continue

        if not attack_items:
            print(f"  WARN [{seed_id}]: không có variant hợp lệ, bỏ qua.")
            continue

        # ── Gửi & đánh giá ───────────────────────────────────────────────
        for item in attack_items:
            try:
                # Gọi target model
                response = call_target(model_key, item["prompt"], system_prompt)

                if not response or not isinstance(response, str):
                    print(f"  WARN [{item['variant_type']}]: target trả về response rỗng.")
                    response = ""

                # Đánh giá response
                eval_result = evaluate_response(item["prompt"], response)

                # Guard: nếu evaluate_response trả về None hoặc không phải dict
                if not isinstance(eval_result, dict):
                    print(f"  WARN [{item['variant_type']}]: evaluate_response trả về None, dùng fallback.")
                    eval_result = EVAL_ERROR_FALLBACK.copy()

                # PAIR: tin theo PAIR judge nếu đã xác nhận bypass
                if attack_type == "pair" and item.get("pair_success"):
                    eval_result["is_success"] = True
                    eval_result["score"]      = 4
                    eval_result["verdict"]    = "BYPASSED"
                    eval_result["reasoning"]  = "PAIR judge confirmed bypass"
                    eval_result["eval_method"] = "pair_judge"

                record = {
                    "seed_id":       seed_id,
                    "category":      seed_item.get("category", "unknown"),
                    "model":         model_key,
                    "attack_type":   attack_type,
                    "variant_type":  item["variant_type"],
                    "attack_prompt": item["prompt"],
                    "response":      response,
                    **eval_result
                }
                if attack_type == "pair":
                    record["pair_iterations"] = item.get("pair_iterations")
                    record["pair_success"]     = item.get("pair_success")

                all_results.append(record)

                status = "✅ BYPASS" if eval_result["is_success"] else "❌ REFUSED"
                method = eval_result.get("eval_method", "?")
                print(f"  {item['variant_type']:30s} → "
                      f"score={eval_result['score']} [{method:10s}] {status}")

                # dolphin là local, không cần sleep
                # groq/gemini là cloud, cần throttle
                if model_key in ("groq", "gemini"):
                    time.sleep(5.0)

            except Exception as e:
                print(f"  ERROR [{item['variant_type']}]: {e}")

    # ── Lưu kết quả ──────────────────────────────────────────────────────
    summary = summarize_results(all_results)
    output  = {
        "experiment": {
            "model":       model_key,
            "attack_type": attack_type,
            "category":    category or "all",
            "n_seeds":     len(dataset),
            "timestamp":   timestamp
        },
        "summary": summary,
        "results": all_results
    }

    out_file = RESULTS_DIR / f"{timestamp}_{model_key}_{attack_type}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ASR:     {summary['asr_percent']}%")
    print(f"  Success: {summary['successful']}/{summary['total_attacks']}")
    print(f"  Saved →  {out_file}")
    print(f"{'='*60}\n")

    return output


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Red Teaming Pipeline Runner")
    parser.add_argument("--model",     default="dolphin",
                        choices=["dolphin", "groq", "gemini"])
    parser.add_argument("--attack",    default="paraphrase",
                        choices=["paraphrase", "persona", "pair"])
    parser.add_argument("--category",  default=None)
    parser.add_argument("--n-seeds",   type=int, default=None,
                        help="Giới hạn số seed (mặc định: tất cả)")
    parser.add_argument("--variants",  type=int, default=3)
    parser.add_argument("--pair-iter", type=int, default=20)
    parser.add_argument("--system",    default="",
                        help="System prompt cho target model")

    args = parser.parse_args()
    run_experiment(
        model_key=args.model,
        attack_type=args.attack,
        category=args.category,
        n_seeds=args.n_seeds,
        n_variants=args.variants,
        max_pair_iter=args.pair_iter,
        system_prompt=args.system
    )