import asyncio, json, logging, time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)
DEFAULT_SUPERCOMPRESS_PATH = Path('~/.hermes/supercompress_store/').expanduser()
DEFAULT_TCMNET_API_URL = 'http://localhost:8000'
DEFAULT_ENERGY_THRESHOLD = 0.5
DEFAULT_DREAM_INTERVAL_SECONDS = 3600 * 24

@dataclass
class CompressionHint:
    high_energy_events: List[Dict] = field(default_factory=list)
    suggested_detail_level: str = 'normal'
    protected_chunks: List[str] = field(default_factory=list)
    strategy_notes: str = ''
    language: str = 'en'
    def to_dict(self) -> Dict:
        return {'high_energy_events': [{'event_id': e.get('event_id'), 'energy_score': e.get('energy_score'), 'anchor_action': e.get('anchor_action'), 'time': e.get('created_at', 'unknown')} for e in self.high_energy_events], 'suggested_detail_level': self.suggested_detail_level, 'strategy_notes': self.strategy_notes, 'language': self.language}

class SupercompressClient:
    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or DEFAULT_SUPERCOMPRESS_PATH
    def connect(self) -> bool:
        try:
            self.store_path.mkdir(parents=True, exist_ok=True)
            return self.store_path.exists()
        except: return False
    def get_current_chunk_info(self) -> Dict:
        if not self.store_path.exists(): return {'chunk_id': None, 'timestamp': time.time()}
        chunk_files = list(self.store_path.glob('chunk_*.jsonl'))
        if chunk_files:
            latest = max(chunk_files, key=lambda f: f.stat().st_mtime)
            return {'chunk_id': latest.name, 'timestamp': latest.stat().st_mtime}
        return {'chunk_id': None, 'timestamp': time.time()}
    def get_messages_by_time_range(self, start_ts: float, end_ts: float, max_messages: int = 100) -> List[Dict]:
        if not self.store_path.exists(): return []
        messages = []
        for chunk_file in sorted(self.store_path.glob('chunk_*.jsonl')):
            if len(messages) >= max_messages: break
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                msg = json.loads(line)
                                if start_ts <= msg.get('timestamp', 0) <= end_ts:
                                    messages.append(msg)
                                    if len(messages) >= max_messages: break
                            except: continue
            except: continue
        return messages[:max_messages]

class TCMNetBridge:
    def __init__(self, tcmnet_api_url: str = DEFAULT_TCMNET_API_URL, supercompress_client: Optional[SupercompressClient] = None, energy_threshold: float = DEFAULT_ENERGY_THRESHOLD):
        self.tcmnet_api_url = tcmnet_api_url
        self.supercompress = supercompress_client or SupercompressClient()
        self.energy_threshold = energy_threshold
        self.last_dream_time = 0.0
        self._state_file = Path('~/.hermes/shared_state/compression_events.jsonl').expanduser()
    async def check_and_trigger_dream(self) -> Optional[CompressionHint]:
        now = time.time()
        if now - self.last_dream_time < DEFAULT_DREAM_INTERVAL_SECONDS: return None
        pending = await self._get_pending_high_energy_events()
        if pending == 0: return None
        hint = await self._run_dream()
        if hint: self.last_dream_time = now
        return hint
    async def _run_dream(self) -> Optional[CompressionHint]:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f'{self.tcmnet_api_url}/api/v1/dream', json={'force_run': True})
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('success') and data.get('events_scanned', 0) > 0:
                        return CompressionHint(high_energy_events=data.get('events', []), strategy_notes=data.get('strategy_notes', ''))
        except Exception as e: logger.error(f'Dream failed: {e}')
        return None
    async def _get_pending_high_energy_events(self) -> int:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f'{self.tcmnet_api_url}/api/v1/events/pending', params={'min_energy': self.energy_threshold})
                if resp.status_code == 200: return resp.json().get('count', 0)
        except: pass
        return 0

_bridge_instance: Optional[TCMNetBridge] = None
def get_bridge() -> Optional[TCMNetBridge]:
    global _bridge_instance
    if _bridge_instance is None:
        try:
            _bridge_instance = TCMNetBridge()
            if not _bridge_instance.supercompress.connect(): _bridge_instance = None
        except: _bridge_instance = None
    return _bridge_instance
def reset_bridge(): global _bridge_instance; _bridge_instance = None
