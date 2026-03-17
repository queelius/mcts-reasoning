# MCTS Action Space Design Notes

## The Problem

Currently, the MCTS action space has one action: "generate a continuation."
Branching comes only from LLM temperature randomness: different samples
of the same prompt. This is weak because:

1. The branches are semantically similar (same prompt, different random seeds)
2. max_tokens=150 was needed to prevent the model from solving in one step
3. The tree doesn't represent genuinely different reasoning strategies

## The Insight

The action space should be **different types of reasoning moves**. Each action
is a different way to prompt the LLM for its next step. MCTS then searches
the space of reasoning strategy sequences, not just random LLM outputs.

## Proposed Action Space

```python
class ReasoningAction(ABC):
    """A type of reasoning move the LLM can make."""
    @abstractmethod
    def prompt(self, question: str, state: State) -> list[Message]:
        """Build the prompt for this action type."""

    @property
    @abstractmethod
    def name(self) -> str: ...
```

Implementations:

### DECOMPOSE
"Break this problem into sub-problems. What are the key things we need to figure out?"
- Good early in the tree (planning phase)
- Produces structured sub-problem lists

### DEDUCE
"Given what we know so far, what follows logically? State one deduction."
- The workhorse action for logical reasoning
- Each deduction builds on prior state

### ASSUME
"Make a specific assumption (e.g., 'Assume A is a knight') and derive one consequence."
- Excellent for case analysis (knights/knaves, proof by contradiction)
- MCTS naturally explores different assumptions in different branches

### VERIFY
"Check the reasoning so far for contradictions or errors."
- Critical for catching mistakes before going deeper
- Can terminate a branch early if contradiction found

### CALCULATE
"Perform the next arithmetic or algebraic step."
- For math problems, forces step-by-step calculation
- Prevents the model from doing mental math badly

### CONCLUDE
"Based on all the reasoning so far, state the final answer."
- Explicitly asks for the answer (not a sub-question answer)
- Should only succeed if the reasoning is sufficient

## How MCTS Uses Actions

At each node, instead of calling `generator.generate(question, state, n=1)`,
the MCTS selects which ACTION to take. Each action produces a different prompt,
which produces a different continuation.

```
Root: "What is 15*7+23?"
├── DECOMPOSE: "Break into sub-problems: (1) compute 15*7, (2) add 23"
│   ├── CALCULATE: "15*7 = 105"
│   │   ├── CALCULATE: "105 + 23 = 128"
│   │   │   └── CONCLUDE: "ANSWER: 128"
│   │   └── VERIFY: "Let me check: 15*7... 10*7=70, 5*7=35, 70+35=105. Correct."
│   │       └── CONCLUDE: "ANSWER: 128"
│   └── DEDUCE: "First, 15*7. I can compute this as (10+5)*7 = 70+35 = 105"
│       └── ...
├── CALCULATE: "15*7 = 105" (jumped straight to calculation)
│   └── ...
└── DEDUCE: "The expression has two operations: multiplication first, then addition"
    └── ...
```

UCB1 learns: DECOMPOSE -> CALCULATE -> CONCLUDE works better than
random DEDUCE chains for arithmetic. ASSUME -> DEDUCE -> VERIFY works
better for logic puzzles.

## Key Design Decision: Action Selection

Option A: **Random action selection** at each node. Simple. MCTS explores
different action sequences by chance, UCB1 favors the ones that work.

Option B: **UCB1 over actions at each node.** Each node maintains visit
counts per action type. UCB1 selects which action to try next. This is
true MCTS over the action space.

Option C: **Problem-type-dependent action sets.** Logic problems get
{ASSUME, DEDUCE, VERIFY, CONCLUDE}. Math problems get {DECOMPOSE,
CALCULATE, VERIFY, CONCLUDE}. The action set is a parameter.

Recommendation: Start with Option A (random), measure if action diversity
helps. Then move to Option B if it does.

## Relationship to PromptStrategy ABC

Each ReasoningAction is essentially a specialized PromptStrategy. But
instead of one PromptStrategy for the whole search, we have a SET of
them, and MCTS picks which one to use at each node.

The Generator interface stays the same. What changes is HOW the generator
picks its prompt. Instead of always using the same PromptStrategy, it
receives an action parameter:

```python
class ActionAwareGenerator(Generator):
    def __init__(self, provider, actions: list[ReasoningAction],
                 terminal_detector, ...):
        self.actions = actions

    def generate(self, question, state, n=1, action=None):
        if action is None:
            action = random.choice(self.actions)
        messages = action.prompt(question, state)
        response = self.provider.generate(messages, ...)
        # Parse, check terminal, build Continuation
        ...
```

## No Token Limit Needed

With proper action design, each action naturally produces a focused response:
- CALCULATE: one computation
- DEDUCE: one logical step
- ASSUME: one assumption + consequence
- VERIFY: one consistency check
- CONCLUDE: the final answer

The model can use as many tokens as it needs. The ACTIONS constrain the
reasoning granularity, not the token count.
