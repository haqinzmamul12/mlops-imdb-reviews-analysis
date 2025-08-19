from bowler import Query
import re
import os


# Convert CamelCase or mixedCase to snake_case
def to_snake_case(name: str) -> str:
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


# Convert CamelCase to PascalCase (for classes)
def to_pascal_case(name: str) -> str:
    return "".join(word.capitalize() for word in re.split("_+", name))


# Project root folder
PROJECT_ROOT = "."  # current folder

# Folders to skip
SKIP_FOLDERS = {".git", "__pycache__", "venv", ".github"}


def should_skip(path):
    return any(part in SKIP_FOLDERS for part in path.split(os.sep))


# Function to run Bowler query on all Python files recursively
def refactor_folder(folder):
    for root, dirs, files in os.walk(folder):
        # Remove skipped folders from traversal
        dirs[:] = [d for d in dirs if d not in SKIP_FOLDERS]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                # Functions & methods
                (
                    Query(file_path)
                    .select_function("name")
                    .modify(
                        lambda node, capture: node.value.replace(
                            node.value, to_snake_case(node.value)
                        )
                    )
                    .execute(interactive=False)
                )

                # Classes
                (
                    Query(file_path)
                    .select_class("name")
                    .modify(
                        lambda node, capture: node.value.replace(
                            node.value, to_pascal_case(node.value)
                        )
                    )
                    .execute(interactive=False)
                )


# Run the refactor
refactor_folder(PROJECT_ROOT)
