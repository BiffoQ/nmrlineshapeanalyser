"""
Generate the code reference pages and navigation.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Explicitly looking for the two Python files we know exist
python_files = [
    "src/nmrlineshapeanalyser/__init__.py",
    "src/nmrlineshapeanalyser/core.py"
]

for path_str in python_files:
    path = Path(path_str)
    module_path = path.relative_to("src").with_suffix("")
    doc_path = path.relative_to("src").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        continue  # Skip __init__.py

    nav[parts] = doc_path.as_posix()

    # Create the necessary directories
    full_doc_path.parent.mkdir(parents=True, exist_ok=True)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Write the page content
        fd.write(f"# {parts[-1]}\n\n")
        fd.write(f"::: nmrlineshapeanalyser.{parts[-1]}\n")
        fd.write(f"    :members:\n")
        fd.write(f"    :show-inheritance:\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Generate the navigation summary
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())