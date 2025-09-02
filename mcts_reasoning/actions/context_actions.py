"""
Context manipulation actions for tree-of-thought exploration.

These actions transform or analyze the context in various ways.
"""

from typing import Dict, Any, Optional
from .base_action import BaseAction, ActionResult


class SummarizeContextAction(BaseAction):
    """Summarize the current context."""
    
    def __init__(self):
        super().__init__(
            action_type="summarize_context",
            description="Summarize the current context to key points"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Current context:
{context}

Please provide a concise summary of the key points and progress so far."""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Summarize the context."""
        prompt = f"""Original question: {original_question}

Current context:
{context}

Please provide a concise summary of the key points and progress so far."""
        
        response = llm_manager.reason(prompt)
        
        # Create summarized context
        new_context = f"Summary: {response}\n\nOriginal question: {original_question}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "summarize_context"}
        )


class DistillContextAction(BaseAction):
    """Distill the essence from the context."""
    
    def __init__(self):
        super().__init__(
            action_type="distill_context",
            description="Distill the context to its essence"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Current context:
{context}

Distill this down to the most essential insights and conclusions."""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Distill the context to its essence."""
        prompt = f"""Original question: {original_question}

Current context:
{context}

Distill this down to the most essential insights and conclusions."""
        
        response = llm_manager.reason(prompt)
        
        new_context = f"Distilled insights: {response}\n\nOriginal question: {original_question}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "distill_context"}
        )


class ExtractMainIdeaAction(BaseAction):
    """Extract the main idea from the context."""
    
    def __init__(self):
        super().__init__(
            action_type="extract_main_idea",
            description="Extract the single most important idea"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Current context:
{context}

What is the single most important idea or insight here?"""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Extract the main idea."""
        prompt = f"""Original question: {original_question}

Current context:
{context}

What is the single most important idea or insight here?"""
        
        response = llm_manager.reason(prompt)
        
        new_context = f"Main idea: {response}\n\nOriginal question: {original_question}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "extract_main_idea"}
        )


class ExtractAssumptionsAction(BaseAction):
    """Extract assumptions from the context."""
    
    def __init__(self):
        super().__init__(
            action_type="extract_assumptions",
            description="Identify underlying assumptions"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Current context:
{context}

What assumptions are being made in this reasoning? List them explicitly."""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Extract assumptions."""
        prompt = f"""Original question: {original_question}

Current context:
{context}

What assumptions are being made in this reasoning? List them explicitly."""
        
        response = llm_manager.reason(prompt)
        
        new_context = f"Assumptions identified: {response}\n\nContext: {context}\n\nOriginal question: {original_question}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "extract_assumptions"}
        )


class ExtractEvidenceAction(BaseAction):
    """Extract evidence and support from the context."""
    
    def __init__(self):
        super().__init__(
            action_type="extract_evidence",
            description="Extract key evidence and support"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Current context:
{context}

What evidence or support has been provided? Extract the key facts and reasoning."""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Extract evidence."""
        prompt = f"""Original question: {original_question}

Current context:
{context}

What evidence or support has been provided? Extract the key facts and reasoning."""
        
        response = llm_manager.reason(prompt)
        
        new_context = f"Evidence: {response}\n\nOriginal question: {original_question}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "extract_evidence"}
        )


class ReframeContextAction(BaseAction):
    """Reframe the context from a different perspective."""
    
    def __init__(self):
        super().__init__(
            action_type="reframe_context",
            description="Reframe from a different perspective"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Current context:
{context}

Reframe this problem or reasoning from a completely different perspective or angle."""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Reframe the context."""
        prompt = f"""Original question: {original_question}

Current context:
{context}

Reframe this problem or reasoning from a completely different perspective or angle."""
        
        response = llm_manager.reason(prompt)
        
        new_context = f"Reframed perspective: {response}\n\nOriginal context: {context}\n\nOriginal question: {original_question}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "reframe_context"}
        )


class IsolateSubproblemContextAction(BaseAction):
    """Isolate a specific subproblem from the context."""
    
    def __init__(self):
        super().__init__(
            action_type="isolate_subproblem",
            description="Isolate the next key subproblem"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Current context:
{context}

Identify and isolate the most important subproblem that needs to be solved next."""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Isolate a subproblem."""
        prompt = f"""Original question: {original_question}

Current context:
{context}

Identify and isolate the most important subproblem that needs to be solved next."""
        
        response = llm_manager.reason(prompt)
        
        new_context = f"Isolated subproblem: {response}\n\nOriginal question: {original_question}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "isolate_subproblem"}
        )


class ProduceFinalAnswerAction(BaseAction):
    """Produce a final answer based on the context."""
    
    def __init__(self):
        super().__init__(
            action_type="produce_final_answer",
            description="Produce comprehensive final answer"
        )
    
    def get_prompt_template(self) -> str:
        return """Original question: {original_question}

Based on all the reasoning and context so far:
{context}

Now provide a clear, comprehensive final answer to the original question."""
    
    def execute(self, context: str, original_question: str, 
                llm_manager: Any, **kwargs) -> ActionResult:
        """Produce final answer."""
        prompt = f"""Original question: {original_question}

Based on all the reasoning and context so far:
{context}

Now provide a clear, comprehensive final answer to the original question."""
        
        response = llm_manager.reason(prompt)
        
        new_context = f"Final answer: {response}\n\nBased on: {context}"
        
        return ActionResult(
            response=response,
            new_context=new_context,
            metadata={"action": "produce_final_answer", "is_final": True}
        )