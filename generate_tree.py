import os

EXCLUDE = {"node_modules", ".git", "evenv", "__pycache__", "dist", ".idea"}


def print_tree(start_path, indent=""):
    for item in sorted(os.listdir(start_path)):
        if item in EXCLUDE or item.startswith("."):
            continue
        full_path = os.path.join(start_path, item)
        print(f"{indent}├── {item}")
        if os.path.isdir(full_path):
            print_tree(full_path, indent + "│   ")


with open("structure.txt", "w", encoding="utf-8") as f:
    from contextlib import redirect_stdout

    with redirect_stdout(f):
        print("📁 Project Structure:\n")
        print_tree(".")
