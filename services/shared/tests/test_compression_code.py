"""The code compressor drops bodies and keeps everything a reader navigates by.

Two properties matter more than the ratio:

* **A language we cannot parse is never guessed at.** Detection is confirmed
  by an error-free parse, so a Ruby file or a unified diff comes back
  byte-identical rather than mangled by whichever grammar matched loosest.
* **Signatures, imports and structure survive exactly.** The point of an
  AST-aware compressor over a line-count heuristic is that the surviving text
  is still true. If a parameter list or an import path changes, the output is
  a plausible lie.
"""

import pytest

from shared.compression.code import (
    ELISION_BLOCK,
    ELISION_PYTHON,
    compress_code,
    detect_language,
)

PYTHON_SOURCE = '''\
import os
from typing import Optional


class Loader:
    """Docstring stays: it is interface, not implementation."""

    def load(self, path: str, strict: bool = False) -> Optional[str]:
        with open(path) as handle:
            body = handle.read()
        if strict and not body:
            raise ValueError("empty")
        return body


def main(argv):
    loader = Loader()
    for arg in argv:
        print(loader.load(arg))
    return 0
'''

GO_SOURCE = """\
package loader

import (
\t"fmt"
\t"os"
)

type Loader struct {
\tRoot string
}

func (l *Loader) Load(path string) (string, error) {
\tdata, err := os.ReadFile(path)
\tif err != nil {
\t\treturn "", fmt.Errorf("read %s: %w", path, err)
\t}
\treturn string(data), nil
}

func main() {
\tfmt.Println("hi")
}
"""

TS_SOURCE = """\
import { readFile } from "fs/promises";

export interface LoaderOptions {
  strict: boolean;
}

export class Loader {
  constructor(private readonly root: string) {}

  async load(path: string, options: LoaderOptions): Promise<string> {
    const body = await readFile(path, "utf8");
    if (options.strict && body.length === 0) {
      throw new Error("empty");
    }
    return body;
  }
}

export function makeLoader(root: string): Loader {
  return new Loader(root);
}
"""


# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (PYTHON_SOURCE, "python"),
        (GO_SOURCE, "go"),
        (TS_SOURCE, "typescript"),
        ("const add = (a, b) => a + b;\nexport default add;\n", "javascript"),
    ],
)
def test_detects_the_three_supported_language_families(source, expected):
    assert detect_language(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "class Loader\n  def load(path)\n    File.read(path)\n  end\nend\n",  # Ruby
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n-old\n+new\n",  # unified diff
        "The quick brown fox jumps over the lazy dog. Nothing here parses.",
        "",
    ],
)
def test_an_unsupported_language_is_not_guessed_at(source):
    assert detect_language(source) is None


def test_broken_python_is_not_claimed_as_python():
    """A parse full of ERROR nodes is not a match, however Python-shaped."""
    broken = "def f(:\n    ???? >>> <<<\nclass ???\n" * 4
    assert detect_language(broken) is None


# --------------------------------------------------------------------------
# Compression
# --------------------------------------------------------------------------


def test_python_keeps_imports_signatures_and_classes_and_drops_bodies():
    out = compress_code(PYTHON_SOURCE)
    assert "import os" in out
    assert "from typing import Optional" in out
    assert "class Loader:" in out
    assert "def load(self, path: str, strict: bool = False) -> Optional[str]:" in out
    assert "def main(argv):" in out
    assert ELISION_PYTHON in out
    # Body statements are gone.
    assert "handle.read()" not in out
    assert "raise ValueError" not in out
    assert "loader = Loader()" not in out


def test_go_keeps_package_imports_types_and_receivers():
    out = compress_code(GO_SOURCE)
    assert "package loader" in out
    assert '"fmt"' in out
    assert "type Loader struct {" in out
    assert "Root string" in out
    assert "func (l *Loader) Load(path string) (string, error)" in out
    assert ELISION_BLOCK in out
    assert "os.ReadFile" not in out
    assert "fmt.Errorf" not in out


def test_typescript_keeps_interfaces_and_method_signatures():
    out = compress_code(TS_SOURCE)
    assert 'import { readFile } from "fs/promises";' in out
    assert "export interface LoaderOptions {" in out
    assert "strict: boolean;" in out
    assert "async load(path: string, options: LoaderOptions): Promise<string>" in out
    assert ELISION_BLOCK in out
    assert "await readFile" not in out


def test_keep_bodies_returns_the_source_unchanged():
    """The caller can ask for structure without paying the lossy price."""
    assert compress_code(PYTHON_SOURCE, keep_bodies=True) == PYTHON_SOURCE


def test_nested_functions_are_elided_once_not_twice():
    source = "def outer(a):\n    def inner(b):\n        return b\n    return inner\n"
    out = compress_code(source)
    assert "def outer(a):" in out
    assert out.count(ELISION_PYTHON) == 1
    assert "def inner" not in out


def test_output_is_smaller_than_the_input():
    for source in (PYTHON_SOURCE, GO_SOURCE, TS_SOURCE):
        assert len(compress_code(source)) < len(source)


# --------------------------------------------------------------------------
# Correctness floor
# --------------------------------------------------------------------------


def test_unsupported_language_passes_through_byte_for_byte():
    ruby = "class Loader\n  def load(path)\n    File.read(path)\n  end\nend\n"
    assert compress_code(ruby) == ruby


def test_a_unified_diff_passes_through_byte_for_byte():
    """A diff is not a parse tree. Guessing at one would corrupt a patch."""
    diff = (
        "diff --git a/app/main.py b/app/main.py\n"
        "--- a/app/main.py\n"
        "+++ b/app/main.py\n"
        "@@ -10,7 +10,7 @@ def handler(request):\n"
        "-    return None\n"
        "+    return Response(200)\n"
    )
    assert compress_code(diff) == diff


def test_signatures_survive_byte_for_byte():
    """Every kept signature must appear in the output exactly as written."""
    source = (
        "def parse_config(path: str, *, strict: bool = True, retries: int = 3) -> dict:\n"
        "    return {}\n"
    )
    out = compress_code(source)
    assert (
        "def parse_config(path: str, *, strict: bool = True, retries: int = 3) -> dict:"
        in out
    )


def test_module_level_constants_and_paths_survive():
    source = (
        "CONFIG_PATH = '/etc/klio/config.yaml'\n"
        "ENDPOINT = 'https://api.klio.tech/v1'\n"
        "VERSION = '1.2.3'\n"
        "\n"
        "def load():\n"
        "    return CONFIG_PATH\n"
    )
    out = compress_code(source)
    assert "'/etc/klio/config.yaml'" in out
    assert "'https://api.klio.tech/v1'" in out
    assert "VERSION = '1.2.3'" in out


def test_a_stack_trace_is_not_code_and_is_never_touched():
    trace = (
        "Traceback (most recent call last):\n"
        '  File "/Users/dev/app/main.py", line 42, in handler\n'
        "    return loader.load(path)\n"
        '  File "/Users/dev/app/loader.py", line 17, in load\n'
        '    raise ValueError("empty")\n'
        "ValueError: empty\n"
    )
    assert compress_code(trace) == trace


# --------------------------------------------------------------------------
# Fail-open
# --------------------------------------------------------------------------


def test_compressor_failure_returns_the_input_unchanged(monkeypatch):
    """Compression is never allowed to raise into a request path."""
    from shared.compression import code as code_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("grammar exploded")

    monkeypatch.setattr(code_module, "_elide_bodies", boom)
    assert compress_code(PYTHON_SOURCE) == PYTHON_SOURCE


def test_missing_grammar_degrades_to_passthrough(monkeypatch):
    from shared.compression import code as code_module

    monkeypatch.setattr(code_module, "_parsers", lambda: {})
    assert detect_language(PYTHON_SOURCE) is None
    assert compress_code(PYTHON_SOURCE) == PYTHON_SOURCE
