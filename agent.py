import time
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, END 
from schemas import EvidenceItem, BatchEvidenceOutput,ReflectionOutput, FinalReport, StepLog
from memory import OSINTMemoryManager
from tools import search_external_sources
from metrics import compute_step_reward
from config import CONFIG 

load_dotenv()

class InvestigationState(TypedDict):
    subject: str
    investigation_type: str  # "company", "person", "claim"
    memory_mode: str         # "none", "short", "long"
    max_steps: int
    current_step: int
    active_queries: List[str]
    gathered_evidence: List[EvidenceItem]
    current_confidence: float
    current_verdict: str
    is_converged: bool
    step_logs: List[StepLog]
    final_report: Optional[FinalReport]

class OSINTAgentHarness:
    def __init__(self, memory_manager: OSINTMemoryManager):
        self.llm= CONFIG.get_llm()
        self.memory= memory_manager
        self.graph= self._build_graph()
        

    def _plan_node(self, state: InvestigationState) -> InvestigationState:
        """Initializes or adapts search queries."""
        if state["current_step"]==0:
            prompt= (
                f"You are an OSINT Lead Investigator. Subject ({state['investigation_type']}): '{state['subject']}'.\n"
                "Generate 2 initial targeted search queries to verify facts, background, and independent corroboration."
            )
            # 8 seconds sleep time to deal with rate limits
            time.sleep(8)
            
            res= self.llm.invoke(prompt)
            raw_text = res.content if isinstance(res.content, str) else "\n".join(
               part.get("text", "") if isinstance(part, dict) else str(part)
               for part in res.content
            )
    
            # 1. Strip the characters
            cleaned_queries = [q.strip("- .123.*\"'") for q in raw_text.split("\n")]
            # 2. Keep only valid strings longer than a few characters
            queries = [q for q in cleaned_queries if len(q) > 2][:2]
            state["active_queries"]= queries
        return state
    
    class BatchEvidenceOutput(BaseModel):
        items: List[EvidenceItem]
    
    def _gather_and_evaluate_node(self, state: InvestigationState) -> InvestigationState:
        """Runs searches, pulls documents, retrieves memory, and scores evidence reliability."""
        t0 = time.time()
        step_sleep_time=0
        step_idx= state["current_step"] +1
        raw_results= []
        irrelevant_mem_total = 0

        seen_urls= {e.source_url for e in state.get("gathered_evidence", [])}
        for q in state["active_queries"]:
            raw_results.extend(search_external_sources(q, seen_urls=seen_urls))
            
       # Structured extraction of EvidenceItems
        extractor = self.llm.with_structured_output(BatchEvidenceOutput, method="json_mode")
        new_evidence: List[EvidenceItem] = []
        eval_prompt=""
        if raw_results:
            docs_Formatted = "\n---\n".join([
                f"Doc [{i}] | URL: {doc['url']} | Base Rel: {doc['base_reliability']}\nContent: {doc['content'][:1000]}"
                for i, doc in enumerate(raw_results)
            ])

            eval_prompt = (
                f"Subject: {state['subject']}\n\n"
                f"Documents:\n{docs_Formatted}\n\n"
                "For each relevant document, extract the key factual finding, assess if it supports/refutes/conflicts "
                "with the subject, and adjust the reliability_score (0.0 to 1.0) based on citation quality and specificity.\n"
                "IMPORTANT: Output your response strictly as a JSON object with an 'items' list using EXACTLY these keys:\n"
                "{\n"
                '  "items": [\n'
                "    {\n"
                '      "claim_addressed": "string",\n'
                '      "extracted_fact": "string",\n'
                '      "source_url": "string",\n'
                '      "source_domain": "string",\n'
                '      "reliability_score": <float between 0.0 and 1.0>,\n'
                '      "stance": "<must be exactly one of: supports, refutes, neutral, conflicting>"\n'
                "    }\n"
                "  ]\n"
                "}"
            )

            try:
                time.sleep(8)
                step_sleep_time += 8
                ev = extractor.invoke(eval_prompt)
                if ev and ev.items:
                    new_evidence.extend(ev.items)
            except Exception as e:
                print(f"Batch extraction failed: {e}")

        # Store in Memory Manager
        self.memory.add_evidence(new_evidence, state["subject"]) 

        # If memory mode is 'none', only keep current step's evidence in state; else accumulate
        if state["memory_mode"] == "none":
            state["gathered_evidence"] = new_evidence
        else:
            state["gathered_evidence"].extend(new_evidence)
        
        # Reflect on accumulated evidence
        reflector = self.llm.with_structured_output(ReflectionOutput, method="json_mode")
        mem_context, irrelevant_mem_total = self.memory.retrieve_context(state["subject"]," ".join(state["active_queries"]))
        evidence_Pool = mem_context if state["memory_mode"] != "none" else new_evidence
        #keep at most the top 12 most reliable/recent evidence items
        evidence_Pool=evidence_Pool[-12:]
        
        evidence_text = "\n".join([
            f"- [{e.stance.upper()} | Rel: {e.reliability_score}] ({e.source_url}): {e.extracted_fact}"
            for e in evidence_Pool
        ])

        reflect_prompt = (
                f"Subject: {state['subject']}\n"
                f"Step: {step_idx}/{state['max_steps']}\n"
                f"Accumulated Evidence:\n{evidence_text}\n\n"
                "Reflect on the findings:\n"
                "1. Identify any conflicting claims across sources and weigh them by reliability_score.\n"
                "2. Determine current verdict and confidence (0.0 - 1.0).\n"
                "3. Decide if the investigation has converged (confidence >= 0.85 with corroborated sources) "
                "or if follow-up queries are needed to resolve conflicts/gaps.\n\n"
                "IMPORTANT: Output your response strictly as a JSON object using EXACTLY the following keys:\n"
                "{\n"
                '  "current_verdict": "<must be exactly one of: VERIFIED, REFUTED, UNVERIFIED_INSUFFICIENT, CONFLICTING>",\n'
                '  "current_confidence": <float between 0.0 and 1.0>,\n'
                '  "conflicts_identified": ["string", "string"],\n'
                '  "missing_information": ["string", "string"],\n'
                '  "is_converged": <boolean true or false>,\n'
                '  "next_search_queries": ["query1", "query2"],\n'
                '  "reasoning": "string"\n'
                "}"
            )
        #time.sleep(8) 
        time.sleep(8)
        step_sleep_time+=8
        reflection: ReflectionOutput = reflector.invoke(reflect_prompt) 
        
        # Estimate token usage & compute step reward
        est_tokens = (len(reflect_prompt) + len(eval_prompt)) // 4 + 400
        step_reward = compute_step_reward(
            new_evidence=new_evidence,
            prev_confidence=state["current_confidence"],
            curr_confidence=reflection.current_confidence,
            irrelevant_mem_count=irrelevant_mem_total,
            tokens_used=est_tokens
        )

        avg_rel = sum(e.reliability_score for e in new_evidence) / max(len(new_evidence), 1)
        log_entry = StepLog(
            step=step_idx,
            node="gather_evaluate_reflect",
            queries_run=state["active_queries"],
            new_evidence_count=len(new_evidence),
            avg_source_reliability=round(avg_rel, 3),
            confidence=reflection.current_confidence,
            verdict=reflection.current_verdict,
            step_reward=step_reward,
            tokens_used=est_tokens,
            latency_sec=round((time.time() - t0) - step_sleep_time,2),
            harmful_memory_retrieved=irrelevant_mem_total
        ) 
        state["current_step"] = step_idx
        state["current_confidence"] = reflection.current_confidence
        state["current_verdict"] = reflection.current_verdict
        state["is_converged"] = reflection.is_converged
        cleaned_next_queries = [q.strip("- .123.*\"'") for q in reflection.next_search_queries]
        state["active_queries"] = [q for q in cleaned_next_queries if len(q) > 2][:2]   
        state["step_logs"].append(log_entry)
        return state  
    
    def _report_node(self, state: InvestigationState) -> InvestigationState:
        """Synthesizes the final analyst-grade report with citations."""
        reporter = self.llm.with_structured_output(FinalReport, method="json_mode")
        evidence_text = "\n".join([
            f"[{i+1}] URL: {e.source_url} | Reliability: {e.reliability_score} | Stance: {e.stance} | Fact: {e.extracted_fact}"
            for i, e in enumerate(state["gathered_evidence"])
        ])
        
        prompt = (
           f"Produce a rigorous OSINT Findings Report for human analysts.\n"
           f"CRITICAL: You MUST incorporate the exact subject, verdict, and confidence values below:\n"
           f"- subject: {state['subject']}\n"
           f"- verdict: {state['current_verdict']}\n"
           f"- overall_confidence: {state['current_confidence']}\n\n"
           f"Evidence Log:\n{evidence_text}\n\n"
           "Ensure all claims are explicitly cited with URLs.\n\n"
           "IMPORTANT: Output your response strictly as a JSON object using EXACTLY the following keys:\n"
           "{\n"
           '  "subject": "string",\n'
           '  "executive_summary": "string (max 50 words)",\n'
           '  "verdict": "<must be exactly one of: VERIFIED, REFUTED, UNVERIFIED_INSUFFICIENT, CONFLICTING>",\n'
           '  "overall_confidence": <float>,\n'
           '  "key_findings": ["string", "string"],\n'
           '  "evidence_synthesis": "string (max 150 words)",\n'
           '  "conflict_resolution_notes": "string",\n'
           '  "citations": ["url1", "url2"]\n'
           "}"
        )
        
        state["final_report"] = reporter.invoke(prompt)
        return state
    
    def _should_continue(self, state: InvestigationState) -> str:
        if state["is_converged"] or state["current_step"] >= state.get("max_steps",4) or not state["active_queries"]:
            return "report"
        return "investigate"

    def _build_graph(self):
        workflow = StateGraph(InvestigationState)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("investigate", self._gather_and_evaluate_node)
        workflow.add_node("report", self._report_node)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "investigate")
        workflow.add_conditional_edges(
            "investigate",
            self._should_continue,
            {"investigate": "investigate", "report": "report"}
        )
        workflow.add_edge("report", END)
        return workflow.compile()
