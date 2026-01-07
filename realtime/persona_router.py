"""
Persona Router: Context-Aware Voice Selection
=============================================

The Cognitive Agent's "Voice Selector" - chooses the right persona (voice buffer)
based on species, context, arousal level, and intent.

This is the critical bridge between:
- Brain: Cognitive decision-making (what to say)
- Mouth: Granular Synthesizer (how to sound it)

Author: Sheel Morjaria (sheelmorjaria@gmail.com)
License: CC BY-ND 4.0 International
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextState(Enum):
    """Behavioral context states for synthesis."""

    SILENCE = "silence"
    CONTACT = "contact"
    ALARM = "alarm"
    AGGRESSIVE = "aggressive"
    FOOD = "food"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    SUBMISSION = "submission"
    URGENCY = "urgency"


@dataclass
class PersonaDefinition:
    """Complete definition of a vocal persona."""

    persona_id: str
    species: str
    cluster_id: int
    source_file: str
    usage: str
    acoustic_profile: Dict[str, Any]
    synthesis_params: Dict[str, Any]


@dataclass
class RoutingDecision:
    """Result of persona routing decision."""

    persona_id: str
    source_file: str
    synthesis_params: Dict[str, Any]
    acoustic_profile: Dict[str, Any]
    context_vector: Optional[Dict[str, float]]
    reasoning: str


class PersonaRouter:
    """
    Context-Aware Persona Router

    Selects the appropriate voice (persona) based on:
    1. Species (marmoset, egyptian_bat, etc.)
    2. Context (contact, alarm, aggression, etc.)
    3. Arousal level (0.0 to 1.0)
    4. Communication distance (roost, close, navigation)
    5. Social complexity (low, medium, high)

    This enables "Voice Switching Synthesis" - the synthesizer can switch
    between different voices seamlessly within a conversation.
    """

    def __init__(self, persona_map_path: Optional[str] = None):
        """
        Initialize Persona Router.

        Args:
            persona_map_path: Path to persona_source_map.json
        """
        self.persona_map_path = persona_map_path or self._find_persona_map()
        self.personas: Dict[str, PersonaDefinition] = {}
        self.routing_rules: Dict[str, Dict] = {}
        self.context_vectors: Dict[str, Dict[str, float]] = {}

        self._load_persona_map()

        logger.info(f"PersonaRouter initialized with {len(self.personas)} personas")

    def _find_persona_map(self) -> str:
        """Find persona_source_map.json in project directory."""
        # Try current directory first
        current_dir = Path(__file__).parent.parent
        path = current_dir / "persona_source_map.json"

        if path.exists():
            return str(path)

        # Fallback to src directory
        path = Path.cwd() / "persona_source_map.json"
        if path.exists():
            return str(path)

        raise FileNotFoundError(
            "Could not find persona_source_map.json. Expected location: ./persona_source_map.json"
        )

    def _load_persona_map(self):
        """Load persona definitions and routing rules from JSON."""
        try:
            with open(self.persona_map_path, "r") as f:
                data = json.load(f)

            # Load personas
            for persona_id, persona_data in data.get("personas", {}).items():
                self.personas[persona_id] = PersonaDefinition(
                    persona_id=persona_id,
                    species=persona_data["species"],
                    cluster_id=persona_data["cluster_id"],
                    source_file=persona_data["source_file"],
                    usage=persona_data["usage"],
                    acoustic_profile=persona_data["acoustic_profile"],
                    synthesis_params=persona_data.get("synthesis_params", {}),
                )

            # Load routing rules
            self.routing_rules = data.get("routing_rules", {})

            # Load context vectors for extrapolation
            extrapolation_data = data.get("contextual_extrapolation", {})
            self.context_vectors = extrapolation_data.get("context_vectors", {})

            logger.info(
                f"Loaded {len(self.personas)} personas, "
                f"{len(self.routing_rules)} species routing rules, "
                f"{len(self.context_vectors)} context vectors"
            )

        except Exception as e:
            logger.error(f"Failed to load persona map: {e}")
            raise

    def select_persona(
        self,
        species: str,
        context: ContextState,
        arousal_level: float = 0.5,
        comm_distance: str = "navigation",
        social_complexity: str = "medium",
    ) -> RoutingDecision:
        """
        Select the appropriate persona based on context.

        Args:
            species: Target species ('marmoset', 'egyptian_bat', etc.)
            context: Behavioral context (contact, alarm, aggressive, etc.)
            arousal_level: Arousal intensity (0.0 = calm, 1.0 = extreme)
            comm_distance: Communication context ('roost', 'close', 'navigation')
            social_complexity: Social complexity ('low', 'medium', 'high')

        Returns:
            RoutingDecision with selected persona and reasoning
        """
        # Normalize species name
        species = self._normalize_species(species)

        # Get routing rules for this species
        species_rules = self.routing_rules.get(species, {})
        default_persona = species_rules.get("default_persona", f"{species.upper()}_DEFAULT")
        exceptions = species_rules.get("exceptions", [])

        # Evaluate exceptions in order
        selected_persona = default_persona
        reasoning = f"Using default persona: {default_persona}"

        for exception in exceptions:
            condition = exception["condition"]
            target_persona = exception["persona"]

            if self._evaluate_condition(
                condition,
                context=context,
                arousal_level=arousal_level,
                comm_distance=comm_distance,
                social_complexity=social_complexity,
            ):
                selected_persona = target_persona
                reasoning = (
                    f"Exception matched: '{condition}' -> "
                    f"Using {target_persona} instead of {default_persona}"
                )
                break

        # Get persona definition
        if selected_persona not in self.personas:
            logger.warning(f"Persona {selected_persona} not found, using default")
            selected_persona = default_persona

        persona_def = self.personas[selected_persona]

        # Get context vector for extrapolation
        context_vector = self._get_context_vector(context.value)

        return RoutingDecision(
            persona_id=selected_persona,
            source_file=persona_def.source_file,
            synthesis_params=persona_def.synthesis_params,
            acoustic_profile=persona_def.acoustic_profile,
            context_vector=context_vector,
            reasoning=reasoning,
        )

    def _normalize_species(self, species: str) -> str:
        """Normalize species name to match routing rules."""
        species_map = {
            "marmoset": "marmoset",
            "egyptian_bat": "egyptian_bat",
            "bat": "egyptian_bat",
            "dolphin": "dolphin",
            "finch": "finch",
            "sperm_whale": "sperm_whale",
        }
        return species_map.get(species.lower(), species.lower())

    def _evaluate_condition(
        self,
        condition: str,
        context: ContextState,
        arousal_level: float,
        comm_distance: str,
        social_complexity: str,
    ) -> bool:
        """
        Evaluate routing exception condition.

        Supports:
        - Comparisons: arousal_level > 0.8
        - Equality: context == 'alarm'
        - Logical OR: context == 'aggressive' or context == 'alarm'
        - Logical AND: comm_distance == 'close' and social_complexity == 'high'
        """
        # Build evaluation context
        eval_context = {
            "context": context.value,
            "arousal_level": arousal_level,
            "comm_distance": comm_distance,
            "social_complexity": social_complexity,
        }

        try:
            # Safe evaluation: only allow specific variables and operations
            result = eval(condition, {"__builtins__": {}}, eval_context)
            return bool(result)
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False

    def _get_context_vector(self, context: str) -> Optional[Dict[str, float]]:
        """Get contextual extrapolation vector for a context."""
        # Map context states to vector names
        vector_map = {
            "neutral": "neutral",
            "contact": "contact",
            "aggressive": "aggression",
            "alarm": "alarm",
            "submission": "submission",
            "urgency": "urgency",
            "silence": "neutral",
            "uncertain": "neutral",
            "food": "neutral",
        }

        vector_name = vector_map.get(context, "neutral")
        return self.context_vectors.get(vector_name)

    def get_persona_definition(self, persona_id: str) -> Optional[PersonaDefinition]:
        """Get complete definition for a specific persona."""
        return self.personas.get(persona_id)

    def list_personas(self, species: Optional[str] = None) -> List[str]:
        """List available personas, optionally filtered by species."""
        if species:
            species = self._normalize_species(species)
            return [p_id for p_id, p_def in self.personas.items() if p_def.species == species]
        return list(self.personas.keys())

    def get_routing_rules(self, species: str) -> Dict:
        """Get routing rules for a species."""
        species = self._normalize_species(species)
        return self.routing_rules.get(species, {})


# ============================================================================
# High-Level Interface
# ============================================================================


def get_persona_router(persona_map_path: Optional[str] = None) -> PersonaRouter:
    """
    Get singleton PersonaRouter instance.

    Args:
        persona_map_path: Optional path to persona_source_map.json

    Returns:
        Configured PersonaRouter instance
    """
    return PersonaRouter(persona_map_path)


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create router
    router = PersonaRouter()

    print("\n" + "=" * 60)
    print("Persona Router Demo")
    print("=" * 60)

    # Test 1: Marmoset neutral contact
    print("\n--- Test 1: Marmoset Neutral Contact ---")
    decision = router.select_persona(
        species="marmoset", context=ContextState.CONTACT, arousal_level=0.3
    )
    print(f"Persona: {decision.persona_id}")
    print(f"Source: {decision.source_file}")
    print(f"Reasoning: {decision.reasoning}")

    # Test 2: Marmoset high-arousal alarm
    print("\n--- Test 2: Marmoset High-Arousal Alarm ---")
    decision = router.select_persona(
        species="marmoset", context=ContextState.ALARM, arousal_level=0.9
    )
    print(f"Persona: {decision.persona_id}")
    print(f"Source: {decision.source_file}")
    print(f"Reasoning: {decision.reasoning}")

    # Test 3: Bat roost communication
    print("\n--- Test 3: Bat Roost Communication ---")
    decision = router.select_persona(
        species="egyptian_bat", context=ContextState.CONTACT, comm_distance="roost"
    )
    print(f"Persona: {decision.persona_id}")
    print(f"Source: {decision.source_file}")
    print(f"Reasoning: {decision.reasoning}")

    # Test 4: Bat close-range high-complexity social
    print("\n--- Test 4: Bat Close-Range High-Complexity Social ---")
    decision = router.select_persona(
        species="bat", context=ContextState.CONTACT, comm_distance="close", social_complexity="high"
    )
    print(f"Persona: {decision.persona_id}")
    print(f"Source: {decision.source_file}")
    print(f"Reasoning: {decision.reasoning}")

    print("\n" + "=" * 60)
    print("Available Personas:")
    print("=" * 60)
    for species in ["marmoset", "egyptian_bat"]:
        personas = router.list_personas(species)
        print(f"\n{species.title()}: {', '.join(personas)}")

    print("\n" + "=" * 60)
    print("Context Vectors Available:")
    print("=" * 60)
    for name, vector in router.context_vectors.items():
        print(f"\n{name}:")
        for key, value in vector.items():
            print(f"  {key}: {value}")
