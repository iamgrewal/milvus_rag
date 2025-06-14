import re

# Read current file
with open('pyproject.toml', 'r') as f:
    content = f.read()

# Fix protobuf constraint - allow 5.x but block 6.x
content = re.sub(
    r'"protobuf>=4\.25\.0,<5\.0\.0"',
    '"protobuf>=4.25.0,<6.0.0"',
    content
)

# Update grpcio-tools constraint to be more flexible
content = re.sub(
    r'"grpcio-tools>=1\.67\.0,<1\.74\.0"',
    '"grpcio-tools>=1.62.0,<1.74.0"',
    content
)

# Write fixed file
with open('pyproject.toml', 'w') as f:
    f.write(content)

print("✅ Fixed protobuf constraint to allow 5.x series")
