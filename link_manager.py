"""
? QUANTUM LINK MANAGER v1.0 ?
Qu?n l? l?u tr? link game v?i m? h?a
"""

import json
import base64
import os
from pathlib import Path
from typing import Optional, Dict


class QuantumLinkManager:
    """
    ?? Qu?n l? link game v?i m? h?a
    """
    
    def __init__(self, config_file: str = ".quantum_game_links.dat"):
        self.config_file = Path.home() / config_file
        
    def _encode(self, data: str) -> str:
        """M? h?a ??n gi?n (base64)"""
        return base64.b64encode(data.encode('utf-8')).decode('utf-8')
    
    def _decode(self, data: str) -> str:
        """Gi?i m?"""
        try:
            return base64.b64decode(data.encode('utf-8')).decode('utf-8')
        except Exception:
            return ""
    
    def save_link(self, game_link: str, game_name: str = "xworld") -> bool:
        """
        ?? L?u link game
        
        Args:
            game_link: Link game t? xworld.info
            game_name: T?n game (default: xworld)
        
        Returns:
            True n?u l?u th?nh c?ng
        """
        try:
            # Parse link ?? l?y userId v? secretKey
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(game_link)
            qs = parse_qs(parsed.query)
            
            user_id = qs.get('userId', [''])[0] or qs.get('userid', [''])[0]
            secret_key = qs.get('secretKey', [''])[0] or qs.get('secretkey', [''])[0]
            
            if not user_id or not secret_key:
                return False
            
            # T?o data object
            data = {
                "game_name": game_name,
                "user_id": self._encode(user_id),
                "secret_key": self._encode(secret_key),
                "full_link": self._encode(game_link),
                "saved_at": str(__import__('datetime').datetime.now())
            }
            
            # L?u v?o file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            # Set permission (ch? user ??c ???c)
            os.chmod(self.config_file, 0o600)
            
            return True
            
        except Exception as e:
            print(f"? L?i khi l?u link: {e}")
            return False
    
    def load_link(self) -> Optional[str]:
        """
        ?? Load link game ?? l?u
        
        Returns:
            Link game n?u c?, None n?u kh?ng
        """
        try:
            if not self.config_file.exists():
                return None
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Gi?i m? link
            full_link = self._decode(data.get('full_link', ''))
            
            if full_link:
                return full_link
            else:
                return None
                
        except Exception as e:
            print(f"? L?i khi load link: {e}")
            return None
    
    def get_saved_info(self) -> Optional[Dict[str, str]]:
        """
        ??  L?y th?ng tin link ?? l?u (kh?ng decode secret)
        
        Returns:
            Dict v?i game_name, saved_at, user_id (partial)
        """
        try:
            if not self.config_file.exists():
                return None
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            user_id = self._decode(data.get('user_id', ''))
            
            # Ch? hi?n th? 4 k? t? ??u c?a user_id (b?o m?t)
            user_id_masked = user_id[:4] + "****" if len(user_id) > 4 else "****"
            
            return {
                "game_name": data.get('game_name', 'Unknown'),
                "saved_at": data.get('saved_at', 'Unknown'),
                "user_id": user_id_masked
            }
            
        except Exception:
            return None
    
    def delete_saved_link(self) -> bool:
        """
        ???  X?a link ?? l?u
        
        Returns:
            True n?u x?a th?nh c?ng
        """
        try:
            if self.config_file.exists():
                self.config_file.unlink()
                return True
            return False
        except Exception:
            return False
    
    def has_saved_link(self) -> bool:
        """
        ? Check xem c? link ?? l?u kh?ng
        
        Returns:
            True n?u c? link ?? l?u
        """
        return self.config_file.exists()
