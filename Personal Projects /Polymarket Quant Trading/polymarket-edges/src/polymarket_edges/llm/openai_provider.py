"""OpenAI LLM provider for rules extraction and report generation (v2)."""

import json
import logging

from openai import AsyncOpenAI
from pydantic import ValidationError

from polymarket_edges.config import settings
from polymarket_edges.llm.provider import LLMProvider, RULES_EXTRACTION_PROMPT_V2, REPORT_GENERATION_PROMPT
from polymarket_edges.models import RulesExtractionOutput, ReportFactsPayload

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI-based LLM provider with v2 capabilities."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialise OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model or settings.openai_model
        self.client = AsyncOpenAI(api_key=self.api_key)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.model

    async def extract_rules(
        self,
        question: str,
        description: str | None,
        rules: str | None,
    ) -> RulesExtractionOutput:
        """Extract structured rules using OpenAI API (v2)."""

        # Build prompt
        prompt = RULES_EXTRACTION_PROMPT_V2.format(
            question=question,
            description=description or "No description provided",
            rules=rules or "No specific rules provided",
        )

        # Attempt extraction with retry on validation failure
        for attempt in range(2):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise JSON extraction system for prediction market analysis. "
                                "Return ONLY valid JSON matching the schema, no markdown, no explanation."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )

                # Extract JSON from response
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from OpenAI")

                # Parse and validate
                data = json.loads(content)
                result = RulesExtractionOutput(**data)

                logger.info(
                    f"Successfully extracted rules for '{question[:50]}...' "
                    f"(ambiguity={result.ambiguity_score:.2f})"
                )
                return result

            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    f"Attempt {attempt + 1}: Failed to parse OpenAI response for "
                    f"'{question[:50]}...': {e}"
                )
                if attempt == 1:
                    # Last attempt failed, return fallback
                    logger.error(f"All attempts failed for '{question[:50]}...', using fallback")
                    return self._fallback_response(question)

            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return self._fallback_response(question)

        return self._fallback_response(question)

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
        # Build prompt with facts
        prompt = REPORT_GENERATION_PROMPT.format(
            facts_json=facts.model_dump_json(indent=2)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quantitative analyst generating concise, factual reports "
                            "for prediction market outcomes. Your reports must:\n"
                            "- Use only the numbers provided in the facts payload\n"
                            "- Not invent or hallucinate data\n"
                            "- Not provide trading advice or predictions\n"
                            "- Be written in British English\n"
                            "- Use markdown formatting\n"
                            "- Be objective and analytical"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty report from OpenAI")

            logger.info(
                f"Generated report for {facts.market_title[:50]}... "
                f"(outcome: {facts.outcome})"
            )
            return content

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return self._fallback_report(facts)

    def _fallback_response(self, question: str) -> RulesExtractionOutput:
        """Return a safe fallback response when extraction fails."""
        return RulesExtractionOutput(
            resolution_source="Unknown - extraction failed",
            primary_measurement="",
            yes_conditions=["Could not parse conditions"],
            no_conditions=["Could not parse conditions"],
            key_dates=[],
            edge_cases=[],
            ambiguity_score=1.0,
            unfalsifiable_flag=True,
            dispute_risk_notes=["LLM extraction failed"],
            recommended_evidence_to_monitor=[],
        )

    def _fallback_report(self, facts: ReportFactsPayload) -> str:
        """Return a simple fallback report when generation fails."""
        return f"""# {facts.market_title}

## Outcome: {facts.outcome}

**Report generation failed.**

## Key Numbers

- Best Bid: {facts.current_best_bid if facts.current_best_bid else 'N/A'}
- Best Ask: {facts.current_best_ask if facts.current_best_ask else 'N/A'}
- Mid Price: {facts.current_mid if facts.current_mid else 'N/A'}
- Spread: {facts.current_spread if facts.current_spread else 'N/A'}

*Automated report generation encountered an error. Please review raw data.*
"""
