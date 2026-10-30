"""Tiny helpers used by the notebook builder (kept separate so cell text stays readable)."""
import textwrap

import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip("\n"))
