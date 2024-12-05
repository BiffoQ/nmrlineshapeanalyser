"""
Generate the code reference pages and navigation.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Change path to look in src/nmrlineshapeanalyser
python_src = Path("src/nmrlineshapeanalyser")

for path in sorted(python_src.rglob("*.py")):
    # Get the relative path within the package
    relative_path = path.relative_to(Path("src"))
    module_path = relative_path.with_suffix("")
    doc_path = relative_path.with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Ensure the package name is included in the import path
        identifier = "nmrlineshapeanalyser." + ".".join(parts[1:]) if len(parts) > 1 else "nmrlineshapeanalyser"
        fd.write(f"# {parts[-1]}\n\n")
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Generate the navigation summary
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())