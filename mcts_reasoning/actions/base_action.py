from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..core.node import Node
from ..llm.ollama_client import LLMManager, LLMResponse


@dataclass
class ActionResult:
    """Result of executing an action."""
    response: str
    new_context: str
    children_actions: List[str] = None  # For actions that spawn sub-actions
    requires_composition: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.children_actions is None:
            self.children_actions = []
        if self.metadata is None:
            self.metadata = {}


class BaseAction(ABC):
    """Base class for all thought actions."""
    
    def __init__(self, action_type: str, description: str):
        self.action_type = action_type
        self.description = description
    
    @abstractmethod
    def execute(self, context: str, original_question: str, 
                llm_manager: LLMManager, **kwargs) -> ActionResult:
        """Execute the action and return the result."""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get the prompt template for this action."""
        pass
    
    def should_preserve_original_question(self) -> bool:
        """Whether this action should preserve the original question in context."""
        return True
    
    def can_have_children(self) -> bool:
        """Whether this action can spawn child actions."""
        return False
    
    def get_followup_actions(self, response: str, context: str) -> List[str]:
        """Get potential follow-up actions based on the response."""
        return []


class ComposableAction(BaseAction):
    """Base class for actions that can have children and need composition."""
    
    def __init__(self, action_type: str, description: str):
        super().__init__(action_type, description)
    
    def can_have_children(self) -> bool:
        return True
    
    @abstractmethod
    def compose_results(self, children_responses: List[str], 
                       original_context: str, llm_manager: LLMManager) -> str:
        """Compose results from child actions."""
        pass