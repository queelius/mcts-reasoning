"""
Context Management and Automatic Summarization

Handles automatic context compression when reasoning states get too long.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context management."""

    # Maximum context length in tokens
    max_context_tokens: int = 8000

    # Trigger summarization when context reaches this fraction of max (0.0-1.0)
    summarize_threshold: float = 0.8

    # Target size after summarization (as fraction of max)
    summarize_target: float = 0.5

    # Use token counting (more accurate) vs character estimation
    use_token_counting: bool = True

    # Characters per token (rough estimate when token counting unavailable)
    chars_per_token: float = 4.0

    def should_summarize(self, current_tokens: int) -> bool:
        """Check if summarization should be triggered."""
        threshold_tokens = int(self.max_context_tokens * self.summarize_threshold)
        return current_tokens >= threshold_tokens

    def get_target_tokens(self) -> int:
        """Get target token count after summarization."""
        return int(self.max_context_tokens * self.summarize_target)


class ContextManager:
    """
    Manages context length and automatic summarization.

    Monitors reasoning state length and triggers summarization when needed.
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize context manager.

        Args:
            config: Context configuration (uses defaults if not provided)
        """
        self.config = config or ContextConfig()
        self._tokenizer = None
        self._summarization_count = 0
        self._last_summarization_tokens = None

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if self.config.use_token_counting and self._tokenizer is not None:
            # Use actual tokenizer
            try:
                return len(self._tokenizer.encode(text))
            except:
                pass

        # Fallback: estimate based on character count
        return int(len(text) / self.config.chars_per_token)

    def load_tokenizer(self, model_name: str):
        """
        Load tokenizer for a specific model.

        Args:
            model_name: Name of the model (e.g., "gpt-4", "claude-3")
        """
        if not self.config.use_token_counting:
            return

        try:
            import tiktoken

            # Try to get encoding for the model
            if "gpt" in model_name.lower():
                try:
                    self._tokenizer = tiktoken.encoding_for_model(model_name)
                    logger.info(f"Loaded tiktoken tokenizer for {model_name}")
                    return
                except:
                    pass

            # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo, claude, etc.)
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info(f"Loaded cl100k_base tokenizer (generic)")

        except ImportError:
            logger.warning("tiktoken not installed. Using character-based estimation.")
            logger.warning("Install tiktoken for more accurate token counting: pip install tiktoken")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}. Using character-based estimation.")

    def should_summarize(self, state: str) -> bool:
        """
        Check if state should be summarized.

        Args:
            state: Current reasoning state

        Returns:
            True if summarization should be triggered
        """
        tokens = self.estimate_tokens(state)
        should = self.config.should_summarize(tokens)

        if should:
            logger.info(f"Context size: {tokens} tokens (threshold: {int(self.config.max_context_tokens * self.config.summarize_threshold)})")
            logger.info("Triggering automatic summarization")

        return should

    def summarize_state(self, state: str, llm, original_question: str) -> str:
        """
        Summarize a state using the LLM.

        Args:
            state: Current reasoning state to summarize
            llm: LLM provider to use for summarization
            original_question: Original question/problem

        Returns:
            Summarized state
        """
        self._summarization_count += 1

        current_tokens = self.estimate_tokens(state)
        target_tokens = self.config.get_target_tokens()

        logger.info(f"Summarization #{self._summarization_count}: {current_tokens} → ~{target_tokens} tokens")

        # Build summarization prompt
        prompt = f"""You are helping with step-by-step reasoning. The reasoning context has grown too large and needs to be compressed.

Original Question:
{original_question}

Current Reasoning State (to be summarized):
{state}

Task: Create a concise summary that preserves:
1. The original question/problem
2. Key insights and findings so far
3. Important intermediate results
4. Current progress and next steps

Target length: Approximately {target_tokens} tokens (about {int(target_tokens * self.config.chars_per_token)} characters).

Provide ONLY the summary, without meta-commentary:"""

        try:
            # Use LLM to summarize
            summary = llm.generate(
                prompt,
                max_tokens=int(target_tokens * 1.5),  # Allow some buffer
                temperature=0.3  # Lower temp for more focused summarization
            )

            summary_tokens = self.estimate_tokens(summary)
            self._last_summarization_tokens = summary_tokens

            logger.info(f"Summary generated: {summary_tokens} tokens")

            # Add marker to indicate summarization happened
            summary_header = f"[Context summarized - compression #{self._summarization_count}]\n\n"
            return summary_header + summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            logger.warning("Falling back to truncation")

            # Fallback: just truncate to target length
            target_chars = int(target_tokens * self.config.chars_per_token)
            return f"[Context truncated]\n\n{state[-target_chars:]}"

    def get_stats(self) -> Dict[str, Any]:
        """Get summarization statistics."""
        return {
            'summarization_count': self._summarization_count,
            'last_summarization_tokens': self._last_summarization_tokens,
            'max_context_tokens': self.config.max_context_tokens,
            'summarize_threshold': self.config.summarize_threshold,
            'summarize_target': self.config.summarize_target,
            'use_token_counting': self.config.use_token_counting and self._tokenizer is not None
        }


def configure_context_for_model(model_name: str) -> ContextConfig:
    """
    Create context configuration based on known model limits.

    Args:
        model_name: Name of the model

    Returns:
        ContextConfig with appropriate limits for the model
    """
    model_lower = model_name.lower()

    # GPT models
    if "gpt-4" in model_lower:
        if "turbo" in model_lower or "1106" in model_lower or "0125" in model_lower:
            return ContextConfig(max_context_tokens=128000, summarize_threshold=0.8)
        else:
            return ContextConfig(max_context_tokens=8192, summarize_threshold=0.8)

    elif "gpt-3.5" in model_lower:
        if "16k" in model_lower:
            return ContextConfig(max_context_tokens=16384, summarize_threshold=0.8)
        else:
            return ContextConfig(max_context_tokens=4096, summarize_threshold=0.8)

    # Claude models
    elif "claude-3" in model_lower or "claude-opus" in model_lower or "claude-sonnet" in model_lower:
        return ContextConfig(max_context_tokens=200000, summarize_threshold=0.8)

    elif "claude-2" in model_lower:
        return ContextConfig(max_context_tokens=100000, summarize_threshold=0.8)

    elif "claude" in model_lower:
        return ContextConfig(max_context_tokens=9000, summarize_threshold=0.8)

    # Ollama models (varies widely, use conservative default)
    elif any(name in model_lower for name in ["llama", "mistral", "gemma", "phi", "qwen"]):
        # Most local models have 4k-32k context
        # Use conservative 8k default
        return ContextConfig(max_context_tokens=8000, summarize_threshold=0.8)

    # Default fallback
    else:
        return ContextConfig(max_context_tokens=8000, summarize_threshold=0.8)


__all__ = ['ContextManager', 'ContextConfig', 'configure_context_for_model']
