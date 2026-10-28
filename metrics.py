from dotenv import load_dotenv
from typing import List, Optional
from schemas import StepLog, EvidenceItem
from config import RewardWeights, ConvergenceConfig

load_dotenv()

def compute_step_reward(
        new_evidence: List[EvidenceItem],
        prev_confidence: float,
        curr_confidence: float,
        irrelevant_mem_count: int,
        tokens_used: int
)->float:
    if not new_evidence:
        return -0.2 - (0.01 * (tokens_used/1000.0))
    
    rw= RewardWeights
    # Average reliability of newly gatherred sources
    avg_rel= sum(e.reliability_score for e in new_evidence)/ len(new_evidence)

    # Confidence progression / information gain
    confidence_delta= curr_confidence - prev_confidence

    # Conflict penalty (uncorroborated contradiction)
    conflicts= sum(1 for e in new_evidence if e.stance == "conflicting")
    conflict_penalty= rw.w_conflict_penalty * conflicts

    # Memory interference penalty
    mem_penalty= rw.w_memory_penalty * irrelevant_mem_count

    # Token cost penalty
    token_penalty= rw.w_token_cost * (tokens_used/1000)

    reward= ( rw.w_reliability * avg_rel) + (rw.w_confidence_gain* confidence_delta) - conflict_penalty - mem_penalty - token_penalty
    return round(reward, 4)

def evaluate_episode_metrics(logs: List[StepLog], ground_truth_verdict: Optional[str]=None) ->dict:
    if not logs:
        return {}
    
    confidences= [log.confidence for log in logs]
    verdicts= [log.verdict for log in logs]
    rewards= [log.step_reward for log in logs]

    #Count verdict flips
    verdict_changes= sum(1 for i in range(1, len(verdicts)) if verdicts[i] != verdicts[i-1])

    #Compute step of convergence (where |C_t - C_{t-1}| < 0.05 and verdict stabilizes)
    convergence_step= len(logs)
    for i in range(1, len(logs)):
        if abs(confidences[i] - confidences[i-1])<ConvergenceConfig.stability_delta and all(v==verdicts[-1] for v in verdicts[i-1:]):
            convergence_step= i+1
            break
    
    total_tokens= sum(log.tokens_used for log in logs)
    total_latency= sum(log.latency_sec for log in logs)
    final_conf= confidences[-1]

    # Episode Reward: cumulative step reward + terminal bonus if high confidence & converged
    episode_reward= sum(rewards) + (0.5 * final_conf if verdict_changes <= 1 else -0.2)
    if ground_truth_verdict:
        accuracy= 1.0 if verdicts[-1] == ground_truth_verdict else 0.0
        episode_reward += (1.0 if accuracy ==1.0 else -1.0)
    else:
        accuracy =None

    return {
        "total_steps": len(logs),
        "convergence_step": convergence_step,
        "final_verdict": verdicts[-1],
        "final_confidence": round(final_conf, 3),
        "verdict_changes": verdict_changes,
        "cumulative_step_reward": round(sum(rewards), 4),
        "episode_reward": round(episode_reward, 4),
        "total_tokens": total_tokens,
        "total_latency_sec": round(total_latency, 2),
        "quality_per_1k_tokens": round(final_conf / max(total_tokens / 1000.0, 0.1), 4),
        "harmful_memory_retrievals": sum(log.harmful_memory_retrieved for log in logs),
        "accuracy": accuracy
    }
    
