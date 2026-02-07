"""Base LLM provider interface with v2 prompts."""

from abc import ABC, abstractmethod

from polymarket_edges.models import RulesExtractionOutput, ReportFactsPayload


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def extract_rules(
        self,
        question: str,
        description: str | None,
        rules: str | None,
    ) -> RulesExtractionOutput:
        """Extract structured rules information from market data.

        Args:
            question: Market question/title
            description: Market description
            rules: Market resolution rules

        Returns:
            RulesExtractionOutput with structured data
        """
        pass

    @abstractmethod
    async def generate_report(
        self,
        facts: ReportFactsPayload,
    ) -> str:
        """Generate human-readable markdown report from facts.

        Args:
            facts: Structured facts payload

        Returns:
            Markdown report string
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name used by this provider."""
        pass


# Prompt template for rules extraction (v2)
RULES_EXTRACTION_PROMPT_V2 = """You are analysing a prediction market to extract structured information about its resolution criteria.

Market Question: {question}

Market Description: {description}

Resolution Rules: {rules}

Extract the following information and return ONLY valid JSON matching this exact schema:

{{
  "resolution_source": "string - the authoritative source that will determine resolution",
  "primary_measurement": "string - the key metric or event being measured",
  "yes_conditions": ["array of specific conditions that would result in 'Yes' resolution"],
  "no_conditions": ["array of specific conditions that would result in 'No' resolution"],
  "key_dates": ["array of important dates in ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"],
  "edge_cases": ["array of unusual scenarios or edge cases that might affect resolution"],
  "ambiguity_score": 0.0,  // float from 0.0 (crystal clear) to 1.0 (highly ambiguous)
  "unfalsifiable_flag": false,  // boolean - true if cannot be objectively verified
  "dispute_risk_notes": ["array of potential points of dispute or controversy"],
  "recommended_evidence_to_monitor": ["array of sources or events to track for resolution"]
}}

Guidelines:
- Be specific and literal when extracting conditions
- Be extremely critical and assume ambiguity exists unless the rules are mathematically precise.
- Identify ALL sources of potential ambiguity
- Consider edge cases and boundary conditions
- Assess whether the resolution source is reliable and specific
- Set unfalsifiable_flag=true for subjective or unverifiable markets
- Extract actual dates mentioned in the rules
- Identify evidence sources that would be useful to monitor

Return ONLY the JSON object, no other text."""


# Prompt template for report generation
REPORT_GENERATION_PROMPT = """Generate a concise analytical report for this prediction market outcome.

You are provided with a JSON payload containing ALL the facts you need. Do NOT invent numbers or make up data.

Facts payload:
{facts_json}

Generate a markdown report with the following structure:

# [Market Title]

## Outcome: [Outcome Name]

### Payout Conditions

[Summarise the resolution criteria from rules_structured. Be precise about what needs to happen for this outcome to resolve YES.]

### Key Numbers

Create a table with current market data:
- Best Bid: [value]
- Best Ask: [value]
- Mid Price: [value]
- Spread: [value]

### Execution Analysis

Summarise the execution metrics across different trade sizes. Highlight:
- Entry and exit VWAPs at the reference size
- Liquidity tax (entry - exit)
- Fill ratios
- Whether execution quality is good or poor

### Constraint Signals

[If constraint_violations is not empty, describe them. Otherwise state "No constraint violations detected."]

### Regime Characteristics

Summarise key regime features:
- Time to resolution
- Market age
- Price volatility
- Spread trends

Include the belief estimate if available.

### Risk Assessment

Based on the ambiguity score and dispute_risk_notes, assess the clarity of resolution criteria.

---

**Important**:
- Use only the numbers provided in the facts payload
- Do not provide trading advice or predictions
- Do not claim certainty about outcomes
- Be objective and analytical
- Use British English spelling"""
