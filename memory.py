import numpy as numpy
from dotenv import load_dotenv
from typing import List, Dict, Literal
from schemas import EvidenceItem

load_dotenv()

class OSINTMemoryManager:
    def __init__(self, mode: Literal["none", "short", "long"] = "long"):
        self.mode= mode
        self.short_term_buffer: List[EvidenceItem]= []
        #For production, replace with FAISS or ChromaDB collection
        self.long_term_store: List[Dict]=[]
        self.STOPWORDS = {
    "the", "a", "an", "in", "on", "of", "and", "or", "to", "for", "is", "are",
    "was", "were", "be", "been", "with", "by", "at", "from", "as", "that", "this", "it"
    }

    def reset_short_term(self):
        self.short_term_buffer= []

    def add_evidence(self, items: List[EvidenceItem], subject:str):
        if self.mode in ['short', 'long']:
            self.short_term_buffer.extend(items)
        if self.mode =="long":
            for item in items:
                self.long_term_store.append({
                    "subject":subject,
                    "evidence": item,
                    "text": f"{subject} | {item.claim_addressed}: {item.extracted_fact}"
                })
    
   
    def retrieve_context(self, subject: str, query: str, max_items: int = 10) -> tuple[List[EvidenceItem], int]:
        """
        Returns (relevant_evidence, irrelevant_or_harmful_count)
        """
        if self.mode == "none":
            return [], 0

        if self.mode == "short":
            return self.short_term_buffer[-max_items:], 0

        if self.mode == "long":
            retrieved = []
            irrelevant_count = 0
            
            # Filter out stopwords and punctuation so common words don't match everything
            query_terms = {
                w.strip(".,!?:;\"'()[]{}") for w in query.lower().split()
            } - self.STOPWORDS
            query_terms = {w for w in query_terms if len(w) > 2}

            scored_entries = []
            for entry in self.long_term_store:
                text_terms = {
                    w.strip(".,!?:;\"'()[]{}") for w in entry["text"].lower().split()
                } - self.STOPWORDS
                overlap = len(query_terms.intersection(text_terms))
                
                # Require at least 2 meaningful overlapping keywords
                if overlap >= 2:
                    scored_entries.append((overlap, entry))

            # Sort by highest keyword overlap and cap to top results
            scored_entries.sort(key=lambda x: x[0], reverse=True)
            for _, entry in scored_entries[:max_items]:
                retrieved.append(entry["evidence"])
                if entry["subject"].lower() != subject.lower():
                    irrelevant_count += 1

            # Combine recent STM (capped) with top LTM and deduplicate by (source_url, extracted_fact)
            recent_stm = self.short_term_buffer[-max_items:]
            combined = list({
                (e.source_url, e.extracted_fact): e for e in (recent_stm + retrieved)
            }.values())
            
            return combined[-max_items:], irrelevant_count  