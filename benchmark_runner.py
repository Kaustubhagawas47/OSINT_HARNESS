from dotenv import load_dotenv
import pandas as pd
import time
import json
from memory import OSINTMemoryManager
from agent import OSINTAgentHarness
from metrics import evaluate_episode_metrics
from config import ConvergenceConfig
load_dotenv()

BENCHMARK_CASES = [
    {
        "subject": "Theranos Edison blood testing device clinical accuracy claims",
        "type": "claim",
        "ground_truth": "REFUTED"
    },
    {
        "subject": "Babson Diagnostics BetterWay blood testing clinical validity",
        "type": "claim",
        "ground_truth": "VERIFIED" # Legitimate capillary blood testing company with clinical validation
    },
    {
        "subject": "Demis Hassabis awarded the 2024 Nobel Prize in Chemistry",
        "type": "person",
        "ground_truth": "VERIFIED"
    }
]

def run_ablation_suite():
    summary_rows = []

    for mode in ["none", "short", "long"]:
        print(f"\n=== Running Benchmark with Memory Mode: {mode.upper()} ===", flush=True)
        mem_manager = OSINTMemoryManager(mode=mode)
        harness = OSINTAgentHarness(memory_manager=mem_manager)

        for case in BENCHMARK_CASES:
            mem_manager.reset_short_term()
            initial_state = {
                "subject": case["subject"],
                "investigation_type": case["type"],
                "memory_mode": mode,
                "max_steps": ConvergenceConfig.max_steps,
                "current_step": 0,
                "active_queries": [],
                "gathered_evidence": [],
                "current_confidence": 0.0,
                "current_verdict": "UNVERIFIED_INSUFFICIENT",
                "is_converged": False,
                "step_logs": [],
                "final_report": None
            }
            final_state = harness.graph.invoke(initial_state)
            ep_metrics = evaluate_episode_metrics(
                final_state["step_logs"],
                ground_truth_verdict=case["ground_truth"]
            )
            ep_metrics["subject"] = case["subject"][:35] + "..."
            ep_metrics["memory_mode"] = mode
            summary_rows.append(ep_metrics)
            report = final_state.get("final_report")
            if report:
               filename = f"report_{case['subject'].replace(' ', '_')}_{mode}.json"
               with open(filename, "w") as f:
                  json.dump(report.model_dump(), f, indent=4)
            print(" [Rate Limit] Waiting 15 seconds before the next case...", flush=True)
            time.sleep(15)
    
    df = pd.DataFrame(summary_rows)
    print("\n=== Ablation & Convergence Results ===")
    print(df.to_string(index=False))
    df.to_csv("osint_benchmark_metrics.csv", index=False)

if __name__ == "__main__":
    run_ablation_suite()