"""Offline consistency checks for the self-contained review source package."""

import hashlib
from pathlib import Path
import re
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
STYLE_DIGEST = 'ebf666a7cc5cff96c51f6b50ae43eb843692a3bcaf238bd8d7066d050327bbde'


class PaperPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = (PAPER / 'main.tex').read_text(encoding='utf-8')

    def test_delivery_contains_only_tex_and_figure_folder(self):
        self.assertEqual({p.name for p in PAPER.iterdir()}, {'main.tex', 'figures'})
        self.assertEqual({p.name for p in (PAPER / 'figures').iterdir()},
                         {'architecture.pdf', 'mamba_loss.png', 'lstm_loss.png'})

    def test_official_style_is_unmodified(self):
        style = self.tex.split('\\begin{filecontents*}{tmlr.sty}\n', 1)[1]
        style = style.split('\\end{filecontents*}', 1)[0].rstrip('\n')
        self.assertEqual(hashlib.sha256(style.encode()).hexdigest(), STYLE_DIGEST)

    def test_default_is_anonymous_submission(self):
        source = '\n'.join(line for line in self.tex.splitlines()
                           if not line.lstrip().startswith('%'))
        self.assertIn('\\usepackage{tmlr}', source)
        self.assertNotRegex(source, r'\\usepackage\[(?:accepted|preprint)\]\{tmlr\}')
        self.assertIn('pdfauthor={}', source)
        self.assertIn('\\author{\\name Anonymous authors}', source)

    def test_all_citations_resolve_in_embedded_bibliography(self):
        cited = set()
        for group in re.findall(r'\\cite[pt]?\{([^}]+)\}', self.tex):
            cited.update(group.split(','))
        entries = re.findall(r'\\bibitem\[[^\]]+\]\{([^}]+)\}', self.tex)
        self.assertEqual(len(entries), len(set(entries)))
        self.assertEqual(cited, set(entries))

    def test_figure_references_exist(self):
        figures = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', self.tex)
        self.assertEqual(set(figures), {'architecture.pdf', 'mamba_loss.png', 'lstm_loss.png'})
        for name in figures:
            self.assertTrue((PAPER / 'figures' / name).is_file())

    def test_anonymous_link_is_explicitly_pending_or_valid_mirror(self):
        match = re.search(r'\\newcommand\{\\AnonymousCodeURL\}\{([^}]*)\}', self.tex)
        self.assertIsNotNone(match)
        url = match.group(1)
        if url:
            self.assertRegex(url, r'^https://anonymous\.4open\.science/r/[^\s]+$')
        else:
            self.assertIn('[Anonymous repository URL to be inserted before submission]', self.tex)
        self.assertIn('\\ifdefempty{\\AnonymousCodeURL}', self.tex)

    def test_original_loss_images_have_no_identifying_metadata_chunks(self):
        for name in ['mamba_loss.png', 'lstm_loss.png']:
            payload = (PAPER / 'figures' / name).read_bytes()
            self.assertEqual(payload[:8], b'\x89PNG\r\n\x1a\n')
            pos = 8
            while pos < len(payload):
                length = struct.unpack('>I', payload[pos:pos + 4])[0]
                kind = payload[pos + 4:pos + 8]
                self.assertNotIn(kind, (b'tEXt', b'zTXt', b'iTXt', b'eXIf'))
                pos += 12 + length
            self.assertEqual(pos, len(payload))


if __name__ == '__main__':
    unittest.main()