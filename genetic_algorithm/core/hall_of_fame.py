"""
Strategy Hall of Fame

Persistent archive of the best strategies discovered across all GA runs.
Saves top strategies to disk so good genetic material is never lost.
Hall of fame members can be re-injected into future evolution runs.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from genetic_algorithm.core.individual import Individual
from genetic_algorithm.core.strategy_gene import StrategyGene

logger = logging.getLogger(__name__)

DEFAULT_HOF_DIR = "genetic_algorithm/data/hall_of_fame"
DEFAULT_MAX_SIZE = 50


@dataclass
class HallOfFameEntry:
    """A single hall of fame entry."""
    strategy_gene_dict: Dict[str, Any]
    fitness: float
    metrics: Dict[str, Any]
    generation_found: int
    run_timestamp: float
    run_id: str = ""
    individual_id: int = 0
    entry_id: str = ""
    
    def __post_init__(self):
        """Generate a stable entry_id if not provided."""
        if not self.entry_id:
            import hashlib
            fp = json.dumps(self.strategy_gene_dict, sort_keys=True, default=str)
            self.entry_id = f"hof_{hashlib.sha256(fp.encode()).hexdigest()[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.entry_id,
            'strategy_gene': self.strategy_gene_dict,
            'fitness': self.fitness,
            'metrics': self.metrics,
            'generation_found': self.generation_found,
            'individual_id': self.individual_id,
            'run_timestamp': self.run_timestamp,
            'run_id': self.run_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HallOfFameEntry':
        return cls(
            strategy_gene_dict=data['strategy_gene'],
            fitness=data['fitness'],
            metrics=data.get('metrics', {}),
            generation_found=data.get('generation_found', 0),
            run_timestamp=data.get('run_timestamp', 0),
            run_id=data.get('run_id', ''),
            individual_id=data.get('individual_id', 0),
            entry_id=data.get('id', ''),
        )


class HallOfFame:
    """
    Persistent archive of top-performing strategies.
    
    Maintains a ranked list of the best strategies ever discovered,
    persisted to a JSON file on disk. Supports:
    - Adding new candidates after each generation 
    - Re-injecting hall of fame members into new GA runs
    - Deduplication by strategy structure similarity
    """
    
    def __init__(self, 
                 directory: str = DEFAULT_HOF_DIR,
                 max_size: int = DEFAULT_MAX_SIZE,
                 min_fitness: float = 0.0,
                 run_id: Optional[str] = None):
        """
        Args:
            directory: Directory to store hall of fame files.
            max_size: Maximum number of strategies to keep.
            min_fitness: Minimum fitness threshold to enter the hall.
            run_id: Optional run identifier; auto-generated from timestamp if omitted.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.min_fitness = min_fitness
        self.entries: List[HallOfFameEntry] = []
        self.run_id = run_id or f"run_{int(time.time())}"
        
        # Load existing hall of fame
        self._load()
    
    @property
    def filepath(self) -> Path:
        return self.directory / "hall_of_fame.json"
    
    def _load(self) -> None:
        """Load hall of fame from disk."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self.entries = [HallOfFameEntry.from_dict(e) for e in data.get('entries', [])]
                logger.info(f"Loaded hall of fame with {len(self.entries)} entries from {self.filepath}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load hall of fame: {e}. Starting fresh.")
                self.entries = []
        else:
            logger.info("No existing hall of fame found. Starting fresh.")
    
    def _save(self) -> None:
        """Save hall of fame to disk."""
        data = {
            'version': 1,
            'last_updated': time.time(),
            'total_entries': len(self.entries),
            'entries': [e.to_dict() for e in self.entries],
        }
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save hall of fame: {e}")
    
    def _is_duplicate(self, gene_dict: Dict[str, Any]) -> bool:
        """
        Check if a strategy is structurally similar to an existing entry.
        
        Uses indicator types + condition operators as a fingerprint.
        """
        new_fp = self._fingerprint(gene_dict)
        for entry in self.entries:
            if self._fingerprint(entry.strategy_gene_dict) == new_fp:
                return True
        return False
    
    def _fingerprint(self, gene_dict: Dict[str, Any]) -> str:
        """Create a structural fingerprint of a strategy gene dict.
        
        Includes indicator types + parameters, condition operators +
        thresholds + indicator refs, and timeframes so that strategies
        differing only in numeric values are NOT treated as duplicates.
        """
        # Indicator: type + sorted params + timeframe
        ind_parts = []
        for ind in gene_dict.get('indicators', []):
            params = ind.get('parameters', {})
            param_str = str(sorted(params.items())) if params else ''
            tf = ind.get('timeframe', '')
            ind_parts.append(f"{ind.get('type', '')}({param_str})@{tf}")
        ind_parts.sort()

        # Conditions: indicator ref + operator + threshold (rounded)
        # Use 1-decimal rounding so near-identical strategies (30.0 vs 30.0001)
        # are properly deduplicated while meaningfully different ones stay separate.
        def _cond_key(c):
            thr = round(c.get('threshold', 0), 1)
            return f"{c.get('indicator', '')}:{c.get('operator', '')}:{thr}"

        entry_keys = sorted(_cond_key(c) for c in gene_dict.get('entry_conditions', []))
        exit_keys = sorted(_cond_key(c) for c in gene_dict.get('exit_conditions', []))

        return f"{'|'.join(ind_parts)}::{'|'.join(entry_keys)}::{'|'.join(exit_keys)}"
    
    def update(self, population, generation: int) -> int:
        """
        Consider top individuals from a population for hall of fame induction.
        
        Args:
            population: Evaluated Population object.
            generation: Current generation number.
            
        Returns:
            Number of new entries added.
        """
        added = 0
        candidates = sorted(
            [ind for ind in population if ind.evaluated and ind.fitness is not None],
            key=lambda x: x.fitness or 0,
            reverse=True
        )
        
        # Consider top 20% as candidates
        n_candidates = max(1, int(len(candidates) * 0.2))
        
        for ind in candidates[:n_candidates]:
            if ind.fitness is None or ind.fitness < self.min_fitness:
                continue
            
            gene_dict = ind.strategy_gene.to_dict()
            
            if self._is_duplicate(gene_dict):
                continue
            
            # Check if it qualifies — use raw_fitness (pre-sharing) for true performance
            actual_fitness = getattr(ind, 'raw_fitness', None) or ind.fitness
            if len(self.entries) < self.max_size or actual_fitness > self.entries[-1].fitness:
                entry = HallOfFameEntry(
                    strategy_gene_dict=gene_dict,
                    fitness=actual_fitness,
                    metrics=ind.metrics or {},
                    generation_found=generation,
                    run_timestamp=time.time(),
                    run_id=self.run_id,
                    individual_id=getattr(ind, 'id', 0) if hasattr(ind, 'id') else 0,
                )
                self.entries.append(entry)
                added += 1
        
        if added > 0:
            # Sort by fitness descending, trim to max_size
            self.entries.sort(key=lambda e: e.fitness, reverse=True)
            self.entries = self.entries[:self.max_size]
            self._save()
            logger.info(f"[HALL OF FAME] Added {added} new entries. Total: {len(self.entries)}")
        
        return added
    
    def get_individuals(self, count: int) -> List[Individual]:
        """
        Create Individual objects from hall of fame entries for re-injection.
        
        Args:
            count: Maximum number of individuals to return.
            
        Returns:
            List of Individual objects with strategy genes from the hall of fame.
        """
        individuals = []
        for entry in self.entries[:count]:
            try:
                gene = StrategyGene.from_dict(entry.strategy_gene_dict)
                ind = Individual(strategy_gene=gene)
                ind.metadata = {'source': 'hall_of_fame', 'original_fitness': entry.fitness}
                individuals.append(ind)
            except Exception as e:
                logger.debug(f"Failed to restore hall of fame entry: {e}")
                continue
        
        return individuals
    
    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the hall of fame."""
        if not self.entries:
            return {'size': 0, 'entries': []}
        
        return {
            'size': len(self.entries),
            'best_fitness': self.entries[0].fitness,
            'worst_fitness': self.entries[-1].fitness,
            'avg_fitness': sum(e.fitness for e in self.entries) / len(self.entries),
            'unique_runs': len(set(e.run_id for e in self.entries if e.run_id)),
            'top_5': [
                {
                    'fitness': round(e.fitness, 4),
                    'profit': round(e.metrics.get('profit', 0), 2),
                    'sharpe': round(e.metrics.get('sharpe_ratio', 0), 2),
                    'generation': e.generation_found,
                    'run_id': e.run_id[:16],
                }
                for e in self.entries[:5]
            ],
        }
