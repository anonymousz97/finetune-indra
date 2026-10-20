"""Supervised Fine-Tuning (SFT) for Qwen2.5-0.5B Schema Extraction with Trained Abstention.
Uses LoRA, PEFT, TRL and evaluates Before vs After on held-out test data.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def evaluate_model(
    model,
    tokenizer,
    test_samples: list[dict],
    device: str = "cuda",
    max_eval: int = 35,
) -> dict:
    model.eval()
    samples = test_samples[:max_eval]
    
    total_fields = 0
    total_present = 0
    total_null = 0
    
    correct_present = 0
    correct_null = 0
    hallucinations = 0
    valid_json = 0
    exact_docs = 0

    print(f"Evaluating on {len(samples)} test samples...", flush=True)
    t0 = time.time()

    for i, s in enumerate(samples):
        messages = s["messages"]
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]
        gt_json_str = messages[2]["content"]
        gt = json.loads(gt_json_str)

        prompt_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy decoding for deterministic evaluation
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Extract JSON substring if wrapped in markdown codeblocks
        if "```json" in resp:
            resp = resp.split("```json")[-1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[-1].split("```")[0].strip()

        is_json = False
        pred = {}
        try:
            pred = json.loads(resp)
            is_json = isinstance(pred, dict)
            if is_json:
                valid_json += 1
        except Exception:
            is_json = False

        doc_all_correct = is_json
        for k, gt_val in gt.items():
            total_fields += 1
            pred_val = pred.get(k) if is_json else "__MISSING__"

            if gt_val is None:
                total_null += 1
                if pred_val is None:
                    correct_null += 1
                else:
                    hallucinations += 1
                    doc_all_correct = False
            else:
                total_present += 1
                if pred_val == gt_val or (pred_val is not None and str(gt_val).strip() == str(pred_val).strip()):
                    correct_present += 1
                else:
                    doc_all_correct = False

        if doc_all_correct:
            exact_docs += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  Processed {i + 1}/{len(samples)} samples...", flush=True)

    dt = time.time() - t0
    res = {
        "samples_evaluated": len(samples),
        "seconds_taken": round(dt, 2),
        "valid_json_rate": round(valid_json / max(1, len(samples)), 4),
        "exact_document_accuracy": round(exact_docs / max(1, len(samples)), 4),
        "field_accuracy": round(correct_present / max(1, total_present), 4),
        "abstention_accuracy": round(correct_null / max(1, total_null), 4),
        "hallucination_rate": round(hallucinations / max(1, total_null), 4),
        "overall_field_accuracy": round((correct_present + correct_null) / max(1, total_fields), 4),
        "counts": {
            "total_fields": total_fields,
            "present_fields": total_present,
            "null_fields": total_null,
            "correct_present": correct_present,
            "correct_null": correct_null,
            "hallucinations": hallucinations,
            "exact_documents": exact_docs
        }
    }
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=Path("work/schema_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("work/qwen_extractor_lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading test samples for baseline evaluation...", flush=True)
    test_file = args.data_dir / "test.jsonl"
    test_samples = [json.loads(line) for line in test_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    print("\n1. Loading Base Model for Baseline Zero-Shot Evaluation...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device
    )

    baseline_cache_file = args.out_dir / "baseline_zero_shot.json"
    if baseline_cache_file.exists():
        print(f"Loading cached baseline metrics from {baseline_cache_file}...", flush=True)
        baseline_metrics = json.loads(baseline_cache_file.read_text())
    else:
        baseline_metrics = evaluate_model(base_model, tokenizer, test_samples, device=device, max_eval=args.eval_samples)
        baseline_cache_file.write_text(json.dumps(baseline_metrics, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("BASELINE (ZERO-SHOT) RESULTS:")
    print(f"  Valid JSON Rate          : {baseline_metrics['valid_json_rate']:.1%}")
    print(f"  Field Accuracy (Present) : {baseline_metrics['field_accuracy']:.1%}")
    print(f"  Abstention Rate (Null)   : {baseline_metrics['abstention_accuracy']:.1%}")
    print(f"  Hallucination Rate       : {baseline_metrics['hallucination_rate']:.1%}")
    print(f"  Exact Document Match     : {baseline_metrics['exact_document_accuracy']:.1%}")
    print("=" * 60 + "\n", flush=True)

    print("2. Preparing Dataset for Fine-Tuning...", flush=True)
    dataset = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl")
    })

    def format_chat(sample):
        return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}

    train_ds = dataset["train"].map(format_chat, remove_columns=dataset["train"].column_names)
    val_ds = dataset["validation"].map(format_chat, remove_columns=dataset["validation"].column_names)
    print(f"  Train: {len(train_ds):,} examples | Val: {len(val_ds):,} examples", flush=True)

    print("3. Configuring LoRA (PEFT)...", flush=True)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(args.out_dir / "checkpoints"),
        dataset_text_field="text",
        max_length=2048,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        args=training_args
    )

    print("\n4. Starting SFT Training on RTX A4000...", flush=True)
    train_t0 = time.time()
    trainer.train()
    train_time = time.time() - train_t0
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.2f} min)!", flush=True)

    print(f"Saving LoRA adapter to {args.out_dir}...", flush=True)
    trainer.model.save_pretrained(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))

    print("\n5. Evaluating Fine-Tuned Model on Test Data...", flush=True)
    finetuned_metrics = evaluate_model(model, tokenizer, test_samples, device=device, max_eval=args.eval_samples)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON: BASELINE ZERO-SHOT vs FINE-TUNED MODEL:")
    print(f"  Metric                     | Base Model | Fine-Tuned | Gain")
    print(f"  ---------------------------+------------+------------+-------")
    print(f"  Valid JSON Rate            |   {baseline_metrics['valid_json_rate']:>6.1%}   |   {finetuned_metrics['valid_json_rate']:>6.1%}   | +{(finetuned_metrics['valid_json_rate'] - baseline_metrics['valid_json_rate'])*100:>4.1f}%")
    print(f"  Field Accuracy (Present)   |   {baseline_metrics['field_accuracy']:>6.1%}   |   {finetuned_metrics['field_accuracy']:>6.1%}   | +{(finetuned_metrics['field_accuracy'] - baseline_metrics['field_accuracy'])*100:>4.1f}%")
    print(f"  Abstention Rate (Null)     |   {baseline_metrics['abstention_accuracy']:>6.1%}   |   {finetuned_metrics['abstention_accuracy']:>6.1%}   | +{(finetuned_metrics['abstention_accuracy'] - baseline_metrics['abstention_accuracy'])*100:>4.1f}%")
    print(f"  Hallucination Rate (Lower) |   {baseline_metrics['hallucination_rate']:>6.1%}   |   {finetuned_metrics['hallucination_rate']:>6.1%}   | -{(baseline_metrics['hallucination_rate'] - finetuned_metrics['hallucination_rate'])*100:>4.1f}%")
    print(f"  Exact Document Match       |   {baseline_metrics['exact_document_accuracy']:>6.1%}   |   {finetuned_metrics['exact_document_accuracy']:>6.1%}   | +{(finetuned_metrics['exact_document_accuracy'] - baseline_metrics['exact_document_accuracy'])*100:>4.1f}%")
    print("=" * 65 + "\n")

    summary_file = args.out_dir / "eval_results.json"
    summary_file.write_text(json.dumps({
        "baseline_zero_shot": baseline_metrics,
        "finetuned_lora": finetuned_metrics,
        "train_time_sec": round(train_time, 2),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples_evaluated": len(test_samples[:args.eval_samples])
    }, indent=2, ensure_ascii=False))
    print(f"Detailed comparison saved to {summary_file}")
    return 0


if __name__ == "__main__":
    main()
