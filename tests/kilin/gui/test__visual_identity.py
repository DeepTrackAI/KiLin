"""Verify KíLín's packaged mosaic artwork and presentation contracts.

Check the assets without depending on image-rendering libraries. The GUI
continues to use Qt for its window icon and CSS for its themed artwork;
branding must not add remote resources or application logic.

"""

# pylint: disable=C0115:missing-class-docstring
# pylint: disable=C0116:missing-function-docstring
# pylint: disable=C0103:invalid-name

from pathlib import Path
import re
import struct
import unittest
from xml.etree import ElementTree

_ROOT = Path(__file__).resolve().parents[3]
_ICONS = _ROOT / "kilin/gui/icons"
_BRAND = _ROOT / "assets/brand"
_WEB = _ROOT / "kilin/gui/web"


def _png_header(path: Path) -> tuple[int, int, int]:
    """Read PNG width, height, and color type from its fixed IHDR header.

    Parameters
    ----------
    path : Path
        Image file to inspect.

    Returns
    -------
    tuple[int, int, int]
        Width, height, and PNG color type.

    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG signature: {path}")
    width, height = struct.unpack(">II", data[16:24])
    return width, height, data[25]


def _contrast_ratio(first: str, second: str) -> float:
    """Compute relative WCAG contrast between two opaque sRGB colors.

    Parameters
    ----------
    first : str
        First six-digit hex color.
    second : str
        Second six-digit hex color.

    Returns
    -------
    float
        Contrast ratio, at least one.

    """
    def luminance(value: str) -> float:
        channels = [int(value[index:index + 2], 16) / 255.0
                    for index in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else
                  ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return sum(a * b for a, b in zip(linear, (0.2126, 0.7152, 0.0722)))

    a, b = sorted((luminance(first), luminance(second)))
    return (b + 0.05) / (a + 0.05)


class TestVisualIdentity(unittest.TestCase):
    """Check accessibility and consistency of the installed GUI resources."""

    def test_mascot_images_are_transparent_and_theme_specific(self):
        """Full-body mascot retains its transparent background."""
        names = ("kilin-qilin.png", "kilin-qilin-dark.png")
        for name in names:
            width, height, color_type = _png_header(_ICONS / name)
            self.assertGreaterEqual(width, 800)
            self.assertGreaterEqual(height, 500)
            self.assertEqual(color_type, 6)
        self.assertNotEqual(
            (_ICONS / names[0]).read_bytes(),
            (_ICONS / names[1]).read_bytes(),
        )

    def test_native_icons_use_existing_window_contract(self):
        """Native theme selection resolves both rounded dragon-head icons."""
        for name in ("kilin-light.png", "kilin-dark.png"):
            self.assertEqual(_png_header(_ICONS / name), (1024, 1024, 6))
        window = (_ROOT / "kilin/gui/window.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"kilin-dark.png" if dark else "kilin-light.png"',
                      window)

    def test_wordmarks_have_even_spacing_and_only_i_accents(self):
        """Outlined letters have equal ink gaps and no K embellishment."""
        namespace = "{http://www.w3.org/2000/svg}"
        expected = ["K", "i", "L", "i", "n"]
        for name in ("kilin-wordmark-light.svg",
                     "kilin-wordmark-dark.svg"):
            root = ElementTree.parse(_ICONS / name).getroot()
            self.assertEqual(root.attrib["aria-label"], "KíLín")
            self.assertIsNone(root.find(f"{namespace}text"))
            letters = root.findall(f"{namespace}g")
            self.assertEqual([g.attrib["data-letter"] for g in letters],
                             expected)
            self.assertEqual([int(g.attrib["data-index"])
                              for g in letters], list(range(5)))
            bounds = [
                (float(g.attrib["data-ink-left"]),
                 float(g.attrib["data-ink-right"]))
                for g in letters
            ]
            gaps = [right[0] - left[1]
                    for left, right in zip(bounds, bounds[1:])]
            for gap in gaps:
                self.assertAlmostEqual(gap, 17.0, places=2)
            accents = root.findall(f"{namespace}path")
            self.assertEqual(len(accents), 2)
            self.assertEqual(
                [p.attrib["data-over"] for p in accents], ["1", "3"]
            )
            self.assertTrue(all(p.attrib["class"] == "kilin-accent"
                                for p in accents))
            # The glyph K is a single outline, not a letter plus a slash.
            self.assertEqual(len(letters[0].findall(f"{namespace}path")),
                             1)
            self.assertEqual(len(letters[0]), 1)

    def test_wordmark_variants_and_lockups_share_the_same_letters(self):
        """Brand masters match the GUI logo without changing mascot art."""
        for name, number in (("light", "08"), ("dark", "09")):
            gui = (_ICONS / f"kilin-wordmark-{name}.svg").read_bytes()
            brand = (_BRAND / f"{number}-wordmark-{name}.svg")
            self.assertEqual(gui, brand.read_bytes())
            for prefix in ("horizontal", "stacked"):
                lockup_number = {
                    ("light", "horizontal"): "10",
                    ("dark", "horizontal"): "11",
                    ("light", "stacked"): "12",
                    ("dark", "stacked"): "13",
                }[(name, prefix)]
                path = _BRAND / (
                    f"{lockup_number}-lockup-{prefix}-{name}.svg"
                )
                root = ElementTree.parse(path).getroot()
                namespace = "{http://www.w3.org/2000/svg}"
                self.assertEqual(len(root.findall(
                    f".//{namespace}g[@class='kilin-letter']")), 5)
                self.assertEqual(len(root.findall(
                    f".//{namespace}path[@class='kilin-accent']")), 2)

    def test_sidebar_and_home_use_full_mascot(self):
        """Project navigation and launch both use the unchanged mascot."""
        full = _png_header(_ICONS / "kilin-qilin.png")
        for name in ("kilin-qilin-head.png",
                     "kilin-qilin-head-dark.png"):
            width, height, color_type = _png_header(_ICONS / name)
            self.assertLess(width, full[0])
            self.assertLess(height, full[1])
            self.assertGreater(width, height)
            self.assertEqual(color_type, 6)
        css = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(".setup-brand-figure {", css)
        self.assertIn("background: var(--logo-image)", css)
        self.assertIn(".sidebar-logo-mark {", css)
        self.assertIn("background: var(--logo-image)", css)
        self.assertIn(".sidebar-logo { display: flex;", css)
        self.assertIn("justify-content: center;", css)
        self.assertIn(".sidebar-logo-wordmark { display: none; }", css)

    def test_light_dark_and_system_theme_use_matching_assets(self):
        """Theme choice changes both wordmark and mascot in pure CSS."""
        css = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertEqual(css.count(
            '--logo-image: url("../icons/kilin-qilin.png");'), 1)
        self.assertEqual(css.count(
            '--logo-image: url("../icons/kilin-qilin-dark.png");'), 2)
        self.assertEqual(css.count(
            '--wordmark-image: url("../icons/kilin-wordmark-light.svg");'),
            1)
        self.assertEqual(css.count(
            '--wordmark-image: url("../icons/kilin-wordmark-dark.svg");'),
            2)
        self.assertIn(':root[data-appearance="dark"]', css)
        self.assertIn(':root[data-appearance="system"]', css)
        self.assertIn('var(--logo-image)', css)
        self.assertIn('var(--wordmark-image)', css)
        self.assertIn('var(--mosaic-image)', css)

    def test_primary_button_contrast_is_readable(self):
        """Brand orange uses dark text instead of low-contrast white."""
        css = (_WEB / "styles.css").read_text(encoding="utf-8")
        bg = re.findall(r"--primary:\s*(#[0-9a-f]{6});", css)
        fg = re.findall(r"--primary-ink:\s*(#[0-9a-f]{6});", css)
        self.assertEqual(len(bg), 3)
        self.assertEqual(len(fg), 3)
        for back, text in zip(bg, fg):
            self.assertGreaterEqual(_contrast_ratio(back, text), 4.5)
        self.assertIn('color: var(--primary-ink)', css)
        self.assertIn('color: var(--accent-ink)', css)

    def test_start_and_sidebar_branding_keep_accessible_names(self):
        """Illustrative marks do not replace the screen reader's title."""
        html = (_WEB / "index.html").read_text(encoding="utf-8")
        self.assertIn('class="setup-brand-figure" aria-hidden="true"',
                      html)
        self.assertIn('class="setup-brand-wordmark" aria-hidden="true"',
                      html)
        self.assertIn('<span class="visually-hidden">KíLín</span>', html)
        self.assertIn('id="home-logo"', html)
        self.assertIn('aria-label="Back to the start screen"', html)
        self.assertIn('Your AI. Your Knowledge. Your Control.', html)


    def test_launch_layout_is_compact_enough_for_initial_view(self):
        """Landing screen reduces vertical padding to avoid initial scroll."""
        css = (_WEB / "styles.css").read_text(encoding="utf-8")
        self.assertIn(
            '.setup { width: min(100%, 650px); margin: 0 auto auto; '
            'padding: 16px 0 10px; }',
            css,
        )
        self.assertIn(
            '.setup-brand { position: relative; display: flex; '
            'align-items: center; gap: 12px; min-height: 146px;',
            css,
        )
        self.assertIn(
            '.setup-card { display: grid; gap: 8px; padding: 20px; }',
            css,
        )
        self.assertIn(
            '.drop-zone { width: 100%; min-height: 102px; padding: 24px;',
            css,
        )

    def test_monochrome_marks_remain_monochrome(self):
        """Black and white brand marks do not inherit colored accents."""
        for name, color in (("03-mark-black.svg", "#12161B"),
                            ("04-mark-white.svg", "#FFFFFF")):
            svg = (_BRAND / name).read_text(encoding="utf-8")
            paints = set(re.findall(r'(?:fill|stroke)="(#[0-9A-Fa-f]{6})"',
                                    svg))
            self.assertEqual(paints, {color})

    def test_portable_brand_files_and_packaging(self):
        """All assets used by CSS are in the installed package data."""
        for name in ("kilin-mosaic-accent.svg",
                     "kilin-wordmark-light.svg",
                     "kilin-wordmark-dark.svg",
                     "kilin-qilin-head.png",
                     "kilin-qilin-head-dark.png"):
            self.assertTrue((_ICONS / name).is_file())
        project = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"icons/*.svg",', project)
        self.assertIn('"icons/*.png",', project)
        self.assertIn('"icons/README.md",', project)
        for name in ("kilin-mascot-master.png", "kilin-head-master.png",
                     "README.md"):
            self.assertTrue((_BRAND / name).is_file())
        for name in ("10-lockup-horizontal-light.svg",
                     "11-lockup-horizontal-dark.svg",
                     "12-lockup-stacked-light.svg",
                     "13-lockup-stacked-dark.svg"):
            content = (_BRAND / name).read_text(encoding="utf-8")
            self.assertIn('data:image/png;base64,', content)
        for name in ("kilin-mosaic-accent.svg",
                     "kilin-wordmark-light.svg"):
            content = (_ICONS / name).read_text(encoding="utf-8")
            self.assertNotIn("http://", content.replace(
                'http://www.w3.org/2000/svg', ''))
            self.assertNotIn("https://", content)


if __name__ == "__main__":
    unittest.main()
