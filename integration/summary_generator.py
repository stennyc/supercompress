import logging
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
REQUIRED_SECTIONS = ['## Standing facts & constraints', '## Goal', '## Current Task', '## Pending & next step']
TCMNET_SECTIONS = ['## 情感高光事件', '## 归因修正', '## 建议策略']
class EnhancedSummaryGenerator:
    def __init__(self, bridge=None): self.bridge = bridge
    def generate_enhanced_summary(self, base_summary: str, hint: Dict[str, Any], messages: Optional[List[Dict]] = None) -> str:
        enhanced = base_summary
        high_energy = hint.get('high_energy_events', [])
        if high_energy: enhanced += self._inject_high_energy_events(high_energy)
        strategy = hint.get('strategy_notes', '')
        if strategy: enhanced += f'\n\n## 建议策略\n{strategy}'
        language = hint.get('language', 'en')
        if language != 'en': enhanced += f'\n\n## Language Hint\nThis summary contains {language} content.'
        return enhanced
    def _inject_high_energy_events(self, events: List[Dict]) -> str:
        lines = ['\n## 情感高光事件', '']
        for event in sorted(events, key=lambda x: x.get('energy_score', 0), reverse=True)[:3]:
            energy = event.get('energy_score', 0)
            action = event.get('anchor_action', 'Unknown')
            importance = '🔴 CRITICAL' if energy >= 0.8 else '🟠 HIGH' if energy >= 0.6 else '🟡 MEDIUM'
            lines.append(f'- {importance} {action}')
        return '\n'.join(lines)
    def validate_summary_structure(self, summary: str) -> bool:
        for section in REQUIRED_SECTIONS:
            if section not in summary: return False
        return True
_generator: Optional[EnhancedSummaryGenerator] = None
def get_generator(bridge=None) -> EnhancedSummaryGenerator:
    global _generator
    if _generator is None: _generator = EnhancedSummaryGenerator(bridge)
    return _generator
