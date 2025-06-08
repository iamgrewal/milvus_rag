import os
from cryptography.fernet import Fernet

# Load your encryption key
fernet = Fernet(os.getenv("ENCRYPTION_KEY"))

# Encrypt your API key
plain_key = "sk-xxxxxYOURKEYxxxxx"
encrypted_key = fernet.encrypt(plain_key.encode())

print(encrypted_key.decode())  # Paste this into your .env
