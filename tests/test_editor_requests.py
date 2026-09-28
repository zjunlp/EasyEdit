"""Regression tests for edit() argument forwarding; only stdlib is required.

Run from the repository root:
    python3 -m unittest discover -s tests -p 'test_editor_requests.py' -v

Extract the actual source functions to avoid importing model dependencies or
initializing a model. Model editing itself is outside the scope of these tests.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import Mock


EDITORS = Path(__file__).resolve().parents[1] / "easyeditor" / "editors"


def load_functions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = tree.body
    if class_name is not None:
        nodes = next(
            node for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ).body
    functions = [
        node for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == set(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)


class EditRequestsTests(unittest.TestCase):
    def setUp(self):
        namespace = dict(vars(typing))
        namespace["BatchEditor"] = SimpleNamespace(is_batchable_method=lambda name: False)
        load_functions(
            EDITORS / "utils.py", ["normalize_ground_truths", "_prepare_requests"], namespace
        )
        load_functions(EDITORS / "editor.py", ["edit"], namespace, "BaseEditor")
        self.edit = namespace["edit"]
        self.prepare_requests = Mock(wraps=namespace["_prepare_requests"])
        namespace["_prepare_requests"] = self.prepare_requests
        self.calls = []
        self.result = object()

        # Keep an explicit requests parameter: a generic Mock would accept the
        # duplicate positional/keyword argument that this regression tests.
        def edit_requests(requests, sequential_edit=False, verbose=True, **kwargs):
            self.calls.append((requests, sequential_edit, verbose, kwargs))
            return self.result

        self.editor = SimpleNamespace(
            hparams=SimpleNamespace(batch_size=1),
            alg_name="ROME",
            edit_requests=edit_requests,
        )

    def test_custom_requests_are_forwarded_once(self):
        requests = [{"prompt": "Custom question", "target_new": "Custom answer"}]
        result = self.edit(self.editor, "Q", "A", requests=requests)
        self.assertIs(result, self.result)
        self.assertEqual(len(self.calls), 1)
        self.assertIs(self.calls[0][0], requests)
        self.assertEqual(self.calls[0][1:], (False, True, {"test_generation": False}))
        self.prepare_requests.assert_not_called()

    def test_ordinary_inputs_still_build_requests(self):
        result = self.edit(self.editor, "Q", "A")
        self.assertIs(result, self.result)
        self.prepare_requests.assert_called_once()
        self.assertEqual(self.calls, [([
            {"prompt": "Q", "target_new": "A", "ground_truth": None,
             "locality": {}, "portability": {}}
        ], False, True, {"test_generation": False})])

    def test_other_options_are_preserved(self):
        requests = [{"prompt": "Q", "target_new": "A"}]
        self.edit(
            self.editor, "Q", "A", requests=requests,
            sequential_edit=True, verbose=False, test_generation=True,
            eval_metric="ppl",
        )
        self.assertEqual(self.calls, [
            (requests, True, False, {"test_generation": True, "eval_metric": "ppl"})
        ])
        self.assertIs(self.calls[0][0], requests)


if __name__ == "__main__":
    unittest.main()
