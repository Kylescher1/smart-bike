#!/usr/bin/env python3
"""Quick test to verify password authentication is working."""

from werkzeug.security import generate_password_hash, check_password_hash

# Test password
test_password = "my-new-password"

# Generate hash
password_hash = generate_password_hash(test_password)
print(f"Password: {test_password}")
print(f"Hash: {password_hash}")

# Test check
result = check_password_hash(password_hash, test_password)
print(f"Check result: {result}")

# Test wrong password
wrong_result = check_password_hash(password_hash, "wrong-password")
print(f"Wrong password check: {wrong_result}")

if result and not wrong_result:
    print("✅ Authentication is working correctly!")
else:
    print("❌ Authentication has issues!")

