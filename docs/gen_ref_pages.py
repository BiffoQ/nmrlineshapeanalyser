"""
Generate the code reference pages and navigation.

You never have to run this manually!
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Change this line to match your project's package location
for path in sorted(Path("nmrlineshapeanalyser").rglob("*.py")):  # Changed from "src" to "nmrlineshapeanalyser"
    module_path = path.relative_to("nmrlineshapeanalyser").with_suffix("")  # Changed here too
    doc_path = path.relative_to("nmrlineshapeanalyser").with_suffix(".md")  # And here
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] in ("__init__", "__main__"):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = "nmrlineshapeanalyser." + ".".join(parts)  # Added package prefix
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())