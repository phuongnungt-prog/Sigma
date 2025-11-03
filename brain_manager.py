"""
?? AI BRAIN MANAGER v16.0
Export/Import AI brain v?i backup v? versioning
"""

import json
import os
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path


class BrainManager:
    """
    ?? AI BRAIN MANAGER
    Qu?n l? export/import/backup AI brain
    """
    
    def __init__(self, brain_dir: str = "./ai_brains"):
        self.brain_dir = Path(brain_dir)
        self.brain_dir.mkdir(exist_ok=True)
        
        self.current_brain_file = "ai_brain_memory.json"
        self.backup_dir = self.brain_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def export_brain(self, 
                     filename: Optional[str] = None,
                     include_metadata: bool = True) -> str:
        """
        Export AI brain ra file
        Returns: path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"brain_export_{timestamp}.json"
        
        # Load current brain
        if not os.path.exists(self.current_brain_file):
            raise FileNotFoundError("No brain file found to export")
        
        with open(self.current_brain_file, 'r', encoding='utf-8') as f:
            brain_data = json.load(f)
        
        # Add export metadata
        if include_metadata:
            brain_data['export_info'] = {
                'exported_at': datetime.now().isoformat(),
                'version': '16.0',
                'exporter': 'BrainManager'
            }
        
        # Save to export location
        export_path = self.brain_dir / filename
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(brain_data, f, indent=2, ensure_ascii=False)
        
        return str(export_path)
    
    def import_brain(self, 
                     import_path: str,
                     backup_current: bool = True) -> bool:
        """
        Import AI brain t? file
        Returns: success
        """
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {import_path}")
        
        # Backup current brain first
        if backup_current and os.path.exists(self.current_brain_file):
            self.create_backup("before_import")
        
        # Load imported brain
        with open(import_path, 'r', encoding='utf-8') as f:
            imported_data = json.load(f)
        
        # Validate
        if not self._validate_brain_data(imported_data):
            raise ValueError("Invalid brain data format")
        
        # Save as current brain
        with open(self.current_brain_file, 'w', encoding='utf-8') as f:
            json.dump(imported_data, f, indent=2, ensure_ascii=False)
        
        return True
    
    def create_backup(self, label: str = "manual") -> str:
        """
        T?o backup c?a brain hi?n t?i
        Returns: backup file path
        """
        if not os.path.exists(self.current_brain_file):
            raise FileNotFoundError("No brain file to backup")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{label}_{timestamp}.json"
        backup_path = self.backup_dir / backup_filename
        
        shutil.copy2(self.current_brain_file, backup_path)
        
        return str(backup_path)
    
    def auto_backup(self, keep_last_n: int = 10) -> Optional[str]:
        """
        Auto backup v?i cleanup old backups
        Returns: backup path or None if no brain exists
        """
        if not os.path.exists(self.current_brain_file):
            return None
        
        # Create backup
        backup_path = self.create_backup("auto")
        
        # Cleanup old backups
        self._cleanup_old_backups(keep_last_n)
        
        return backup_path
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all backups
        """
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("backup_*.json"), reverse=True):
            stat = backup_file.stat()
            
            backups.append({
                'filename': backup_file.name,
                'path': str(backup_file),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'age_hours': (datetime.now().timestamp() - stat.st_ctime) / 3600
            })
        
        return backups
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore t? backup
        """
        return self.import_brain(backup_path, backup_current=True)
    
    def get_brain_info(self, brain_path: Optional[str] = None) -> Dict[str, Any]:
        """
        L?y th?ng tin v? brain file
        """
        if brain_path is None:
            brain_path = self.current_brain_file
        
        if not os.path.exists(brain_path):
            return {'exists': False}
        
        with open(brain_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = data.get('metadata', {})
        
        return {
            'exists': True,
            'version': metadata.get('version', 'unknown'),
            'total_rounds': metadata.get('total_rounds', 0),
            'accuracy': metadata.get('accuracy', 0.0),
            'last_updated': metadata.get('last_updated', 'unknown'),
            'file_size': os.path.getsize(brain_path),
            'path': brain_path
        }
    
    def compare_brains(self, 
                       brain_path_1: str,
                       brain_path_2: str) -> Dict[str, Any]:
        """
        So s?nh 2 brain files
        """
        info1 = self.get_brain_info(brain_path_1)
        info2 = self.get_brain_info(brain_path_2)
        
        if not info1['exists'] or not info2['exists']:
            return {'error': 'One or both brains not found'}
        
        comparison = {
            'brain_1': info1,
            'brain_2': info2,
            'differences': {
                'rounds_diff': info2['total_rounds'] - info1['total_rounds'],
                'accuracy_diff': info2['accuracy'] - info1['accuracy'],
                'size_diff': info2['file_size'] - info1['file_size']
            },
            'recommendation': self._get_comparison_recommendation(info1, info2)
        }
        
        return comparison
    
    def _validate_brain_data(self, data: Dict[str, Any]) -> bool:
        """Validate brain data format"""
        required_keys = ['metadata', 'online_learner', 'pattern_learner', 
                        'adaptive_strategy', 'memory_learner']
        
        for key in required_keys:
            if key not in data:
                return False
        
        return True
    
    def _cleanup_old_backups(self, keep_last_n: int):
        """X?a backup c?, gi? l?i N backups g?n nh?t"""
        backups = list(self.backup_dir.glob("backup_auto_*.json"))
        
        if len(backups) <= keep_last_n:
            return
        
        # Sort by creation time (oldest first)
        backups.sort(key=lambda x: x.stat().st_ctime)
        
        # Remove oldest
        for backup in backups[:-keep_last_n]:
            backup.unlink()
    
    def _get_comparison_recommendation(self, 
                                       info1: Dict[str, Any],
                                       info2: Dict[str, Any]) -> str:
        """?? xu?t brain n?o t?t h?n"""
        # More rounds = more learning
        if info2['total_rounds'] > info1['total_rounds'] * 1.5:
            rounds_vote = 'brain_2'
        elif info1['total_rounds'] > info2['total_rounds'] * 1.5:
            rounds_vote = 'brain_1'
        else:
            rounds_vote = 'tie'
        
        # Higher accuracy = better
        if info2['accuracy'] > info1['accuracy'] + 0.05:
            accuracy_vote = 'brain_2'
        elif info1['accuracy'] > info2['accuracy'] + 0.05:
            accuracy_vote = 'brain_1'
        else:
            accuracy_vote = 'tie'
        
        # Determine recommendation
        if rounds_vote == 'brain_2' and accuracy_vote in ['brain_2', 'tie']:
            return "Brain 2 is better (more learning + higher/equal accuracy)"
        elif rounds_vote == 'brain_1' and accuracy_vote in ['brain_1', 'tie']:
            return "Brain 1 is better (more learning + higher/equal accuracy)"
        elif accuracy_vote == 'brain_2':
            return "Brain 2 is better (higher accuracy)"
        elif accuracy_vote == 'brain_1':
            return "Brain 1 is better (higher accuracy)"
        else:
            return "Both brains are similar - use the one with more rounds"
    
    def merge_brains(self,
                     brain_path_1: str,
                     brain_path_2: str,
                     output_path: Optional[str] = None) -> str:
        """
        Merge 2 brains (experimental)
        Combine knowledge from both
        """
        with open(brain_path_1, 'r') as f:
            brain1 = json.load(f)
        
        with open(brain_path_2, 'r') as f:
            brain2 = json.load(f)
        
        # Merge metadata (use higher values)
        merged = {
            'metadata': {
                'version': '16.0',
                'last_updated': datetime.now().isoformat(),
                'total_rounds': brain1['metadata']['total_rounds'] + brain2['metadata']['total_rounds'],
                'accuracy': max(brain1['metadata']['accuracy'], brain2['metadata']['accuracy']),
                'merged_from': [brain_path_1, brain_path_2]
            }
        }
        
        # Merge online learner (average weights)
        ol1 = brain1['online_learner']['feature_weights']
        ol2 = brain2['online_learner']['feature_weights']
        
        merged['online_learner'] = {
            'feature_weights': {
                k: (ol1.get(k, 0) + ol2.get(k, 0)) / 2
                for k in set(list(ol1.keys()) + list(ol2.keys()))
            },
            'learning_rate': min(brain1['online_learner']['learning_rate'], 
                               brain2['online_learner']['learning_rate']),
            'round_count': merged['metadata']['total_rounds'],
            'correct_predictions': brain1['online_learner']['correct_predictions'] + 
                                 brain2['online_learner']['correct_predictions']
        }
        
        # Merge patterns (combine sequences)
        merged['pattern_learner'] = brain1['pattern_learner'].copy()
        # Simple merge - could be more sophisticated
        
        # Merge strategies (combine stats)
        merged['adaptive_strategy'] = brain1['adaptive_strategy'].copy()
        
        # Merge memory (keep both)
        merged['memory_learner'] = brain1['memory_learner'].copy()
        
        # Save merged brain
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.brain_dir / f"merged_brain_{timestamp}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        
        return output_path
