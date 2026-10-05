"""Only the released numerical core belongs in the active measurement module."""
import ast
from pathlib import Path
import re
import unittest

REPO = Path(__file__).resolve().parents[1]


class CoreScopeTests(unittest.TestCase):
    def test_legacy_entry_point_is_not_in_active_core(self):
        tree = ast.parse((REPO / 'src/measurement_core.py').read_text(encoding='utf-8'))
        definitions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(len(definitions), 24)
        self.assertNotIn('main', definitions)
        self.assertIn('measure_rois', definitions)
        self.assertIn('refine_similarity', definitions)

    def test_sources_have_no_personal_filesystem_roots(self):
        pattern = re.compile(rb'[A-Za-z]:[\\/]+Users[\\/]+|/(?:home|Users)/', re.I)
        for path in REPO.rglob('*.py'):
            if any(part in {'.venv', 'venv', '.git', '__pycache__', '.pytest_cache'} for part in path.relative_to(REPO).parts):
                continue
            self.assertIsNone(pattern.search(path.read_bytes()), str(path.relative_to(REPO)))
