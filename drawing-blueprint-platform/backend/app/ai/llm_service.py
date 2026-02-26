"""
Abstract LLM service for generating post-analysis drawing feedback.

After the blueprint pipeline runs, the LLM service looks at the results
(complexity, proportion accuracy, landmarks, shading) and produces:
  1. An overall assessment of the drawing
  2. Strengths identified in the work
  3. Specific areas for improvement
  4. Recommended practice exercises

The OpenAI implementation returns placeholders when no API key is set.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseLLMService(ABC):
    """Abstract base — swap implementations without touching endpoints."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        ...

    def build_feedback_prompt(
        self,
        complexity_score: float,
        proportion_accuracy: float,
        landmark_count: int,
        shading_region_count: int,
    ) -> str:
        return (
            "You are an expert art instructor and anatomy specialist.\n"
            "Analyse the following drawing analysis results and provide "
            "constructive feedback in JSON format with keys: "
            '"overall_assessment", "strengths", "areas_for_improvement", '
            '"recommended_exercises".\n\n'
            f"Complexity Score : {complexity_score}/100\n"
            f"Proportion Accuracy : {proportion_accuracy}%\n"
            f"Detected Landmarks : {landmark_count}\n"
            f"Shading Regions : {shading_region_count}\n"
        )

    async def get_drawing_feedback(
        self,
        complexity_score: float,
        proportion_accuracy: float,
        landmark_count: int,
        shading_region_count: int,
    ) -> Optional[Dict]:
        """High-level helper — generate and parse feedback. Returns None on failure."""
        prompt = self.build_feedback_prompt(
            complexity_score, proportion_accuracy, landmark_count, shading_region_count
        )
        try:
            raw = await self.generate(prompt)
            return json.loads(raw)
        except Exception as exc:
            logger.error("LLM feedback generation failed: %s", exc)
            return None


class OpenAILLMService(BaseLLMService):
    """
    OpenAI implementation. Returns placeholder when OPENAI_API_KEY is not set.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self.api_key:
            # Return useful placeholder feedback so the prototype still works
            return json.dumps({
                "overall_assessment": (
                    "Your drawing shows solid foundational structure. "
                    "The landmark placement and proportions demonstrate an understanding "
                    "of basic anatomy."
                ),
                "strengths": [
                    "Consistent body proportions detected",
                    "Good use of shading depth across the figure",
                    "Clean outline structure suitable for blueprint conversion",
                ],
                "areas_for_improvement": [
                    "Work on limb-to-torso ratio for more natural proportions",
                    "Add more variation in shading transitions between zones",
                    "Pay attention to joint angles at elbows and knees",
                ],
                "recommended_exercises": [
                    "Practice 2-minute gesture drawings to loosen up form",
                    "Study the Loomis 8-head proportion model",
                    "Do value studies with 3 tones: light, mid, shadow",
                ],
            })

        # When API key is available — real call would go here:
        # import openai
        # client = openai.AsyncOpenAI(api_key=self.api_key)
        # response = await client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     response_format={"type": "json_object"},
        # )
        # return response.choices[0].message.content

        return json.dumps({"overall_assessment": "LLM configured but running in dev mode."})


def get_llm_service() -> BaseLLMService:
    """Factory — returns the configured LLM service."""
    return OpenAILLMService()
