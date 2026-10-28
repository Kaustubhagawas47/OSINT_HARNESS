import os
from dataclasses import dataclass, field
from typing import Dict
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

@dataclass(frozen=True)
class RewardWeights:
    """
    Hyperparameters for the Step-Level Reward formula (R_t).
    Centralized here so you can justify and tune them in your evaluation report.
    """
    w_reliability: float = 0.40       # Weight for average source reliability
    w_confidence_gain: float = 0.50   # Weight for information gain (delta confidence)
    w_conflict_penalty: float = 0.10  # Penalty per uncorroborated conflicting claim
    w_memory_penalty: float = 0.15    # Penalty per harmful/irrelevant LTM retrieval
    w_token_cost: float = 0.02        # Penalty per 1,000 tokens consumed


@dataclass(frozen=True)
class ConvergenceConfig:
    """Thresholds governing when the investigation loop terminates."""
    max_steps: int = 4
    confidence_threshold: float = 0.85
    stability_delta: float = 0.05     # |C_t - C_{t-1}| < delta indicates convergence
    max_queries_per_step: int = 2
    max_results_per_query: int = 3
    max_scrape_chars: int = 1000


# Heuristic baseline reliability scores by domain/TLD for OSINT evaluation
DOMAIN_RELIABILITY_TIERS: Dict[str, float] = {
    # Tier 1: Primary & Regulatory Filings / Official Institutions
    "sec.gov": 0.98,
    "gov": 0.95,
    "edu": 0.90,
    "nobelprize.org": 0.98,
    # Tier 2: Established Financial & Wire Journalism
    "reuters.com": 0.90,
    "bloomberg.com": 0.90,
    "apnews.com": 0.90,
    "ft.com": 0.88,
    "wsj.com": 0.88,
    # Tier 3: Aggregators & Secondary Encyclopedias
    "wikipedia.org": 0.70,
    "crunchbase.com": 0.75,
    "techcrunch.com": 0.72,
    # Tier 4: Unverified User-Generated Content / Blogs
    "medium.com": 0.40,
    "substack.com": 0.45,
    "reddit.com": 0.25,
    "x.com": 0.20,
    "twitter.com": 0.20,
}

DEFAULT_UNKNOWN_DOMAIN_RELIABILITY: float = 0.50


@dataclass
class HarnessConfig:
    """Master configuration object for the OSINT Agent Harness."""
    provider: str = os.getenv("LLM_PROVIDER", "groq")  
    model_name: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    temperature: float = 0.0
    
    rewards: RewardWeights = field(default_factory=RewardWeights)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    
    # File paths for logs and benchmarks
    log_dir: str = "logs"
    metrics_output_csv: str = "osint_benchmark_metrics.csv"

    def get_llm(self):
        """Factory method supporting free LLM tiers (Gemini, Groq, Ollama)."""
        if self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name, # e.g., "gemini-3.6-flash"
                temperature=self.temperature,
                google_api_key=os.getenv("GEMINI_API_KEY")
            )
        elif self.provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model_name,     #"openai/gpt-oss-20b"
                temperature=self.temperature,
                api_key=os.getenv("GROQ_API_KEY"),
                max_retries=5,
                max_tokens= 4096
                )
        elif self.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self.model_name, 
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
# Instantiate a singleton config to import across tools.py, metrics.py, and agent.py
CONFIG = HarnessConfig()
os.makedirs(CONFIG.log_dir, exist_ok=True)