from dotenv import load_dotenv
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

load_dotenv()

class EvidenceItem(BaseModel):
    claim_addressed: str
    extracted_fact: str
    source_url : str
    source_domain: str
    reliability_score: float=Field(description="0.0 to 1.0 based on domain authority and corroboration")
    stance: Literal["supports", "refutes", "neutral", "conflicting"]

class BatchEvidenceOutput(BaseModel):
      items: List[EvidenceItem]

class ReflectionOutput(BaseModel):
    current_verdict: Literal["VERIFIED", 'REFUTED', "UNVERIFIED_INSUFFICIENT", "CONFLICTING"]
    current_confidence: float= Field(ge=0.0, le=1.0)
    conflicts_identified: List[str]
    missing_information: List[str]
    is_converged: bool =Field(description="True if confidence is stable or max utility reached")
    next_search_queries: List[str]= Field(description="Empty if is_converged is True")
    reasoning:str

class FinalReport(BaseModel):
    subject: str
    executive_summary: str
    verdict: Literal["VERIFIED", "REFUTED", "UNVERIFIED_INSUFFICIENT", "CONFLICTING"]
    overall_confidence: float
    key_findings: List[str]
    evidence_synthesis: str
    conflict_resolution_notes: str
    citations: List[str]

class StepLog(BaseModel):
    step: int
    node: str
    queries_run: List[str] = []
    new_evidence_count: int = 0
    avg_source_reliability: float = 0.0
    confidence: float = 0.0
    verdict: str = ""
    step_reward: float = 0.0
    tokens_used: int = 0
    latency_sec: float = 0.0
    harmful_memory_retrieved: int = 0
