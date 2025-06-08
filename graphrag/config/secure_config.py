# graphrag/config/secure_config.py
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        encryption_key = os.getenv("ENCRYPTION_KEY")
        if not encryption_key:
            raise ValueError("ENCRYPTION_KEY must be set in the environment.")
        self.cipher = Fernet(encryption_key)

    def get_api_key(self, service: str) -> str:
        encrypted = os.getenv(f"{service.upper()}_API_KEY_ENCRYPTED")
        if not encrypted:
            raise ValueError(f"Encrypted API key for {service} not found in environment.")
        return self.cipher.decrypt(encrypted.encode()).decode()
