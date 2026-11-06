# KíLín portable brand assets

This directory collects master artwork and portable exports for the KíLín
visual identity. It is a source-asset directory; the GUI uses packaged
copies under `kilin/gui/icons/` instead of loading files from this folder.

## Files

- `kilin-mascot-master.png` and `kilin-head-master.png`: full mascot and
  head artwork from which presentation and icon variants are prepared.
- `kilin-light.png` and `kilin-dark.png`: light/dark application icon exports.
- `01-mark-light.svg` through `04-mark-white.svg`: mascot-and-mark variants,
  including monochrome black and white versions.
- `05-icon-light.svg`, `06-icon-dark.svg`, `07-icon-small.svg`: icon variants.
- `08-wordmark-light.svg`, `09-wordmark-dark.svg`: stand-alone wordmarks.
- `10-lockup-horizontal-light.svg` through `13-lockup-stacked-dark.svg`:
  horizontal and stacked compositions for light and dark backgrounds.
- `kilin-mosaic-accent.svg`: decorative mosaic accent.

Some SVG compositions embed PNG imagery as data URIs so they remain portable
without external image paths. When preparing a GUI release, use the copies in
`kilin/gui/icons/`; do not add runtime references back to this directory.

For the earlier Qilin illustration's author and licensing history, see
[`kilin/gui/icons/README.md`](../../kilin/gui/icons/README.md). The historical
attribution does not by itself establish the provenance or licence of the
newer mosaic-style artwork.
