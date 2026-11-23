# README links, package metadata and release tags

## Ownership and purpose

`README.md` at the repository root is the **only hand-maintained README**.
It serves GitHub's branch and tag views. `README.pypi.md` is a generated,
version-controlled distribution description, not an independent document.
The generator is `setup/generate_pypi_readme.py`. `pyproject.toml` points
its `[project].readme` field to `README.pypi.md`; `MANIFEST.in` includes
both README files in source distributions.

This distinction is necessary because GitHub resolves repository-relative
links in the current branch or tag, while a PyPI project description has no
repository-relative base URL. A PyPI link must identify an actual external
GitHub resource. Maintaining the two descriptions manually would permit
citation, installation and documentation claims to drift apart.

## Writing the source README

Use paths **relative to the repository root** for all repository-owned files,
images and documentation directories. Do not hard-code `main`, `develop`,
or a release tag in these links:

```markdown
[Model management](docs/model_management.md)
[Tutorials](tutorials/)
![Logo](assets/brand/10-lockup-horizontal-light.svg)
```

HTML `src`, `srcset` and `href` attributes in the root README follow the
same rule. In particular, the `<picture>` logo must use repository-relative
paths for its light- and dark-mode images. Fragment-only links such as
`#installation` should remain fragment-only.

External sites and repository-wide services remain absolute: GitHub Releases,
GitHub Issues, PyPI, shields.io badges and other independent websites do
not represent a file in the selected revision. Do not attempt to make those
URLs branch-relative or rewrite their targets during generation.

Use percent-encoded filenames for unusual Markdown URLs, with ordinary
Markdown inline links/images or HTML `src`/`srcset`/`href` attributes. The
standard-library generator intentionally supports these documented forms;
code fences and inline code are preserved. Local paths must exist within the
checkout. If a new README construct is needed, extend the generator and
its tests **before** relying on that syntax in a release description.

## What generation does

The generator reads the package version from `pyproject.toml` and derives
`v<version>` automatically. For version `0.1.0b1`, the guide link
shown above becomes:

```markdown
[guide](https://github.com/DeepTrackAI/KiLin/blob/v0.1.0b1/docs/model_management.md)
```

Markdown file links use GitHub's `/blob/` route; directory links use
`/tree/`. Images use `raw.githubusercontent.com` at the same tag so PyPI
can request the image bytes instead of a GitHub HTML page. External URLs,
same-page anchors and code examples are preserved. Missing or escaping
local paths fail generation rather than publish a broken reference.

The output starts with a generated-file notice. It must contain **only**
content derived from `README.md`; do not edit it by hand. There is no link
resolution against the currently selected GitHub default branch.

## When and how to regenerate

From the repository root, after changing `README.md` **or** the version in
`pyproject.toml`, run:

```sh
python -m setup.generate_pypi_readme
python -m setup.generate_pypi_readme --check
python -m unittest tests.setup.test__pypi_readme -v
```

Commit the resulting `README.pypi.md` together with the source/version
change. The `--check` command performs a byte-for-byte comparison without
writing; it is run in CI on **every PR** and therefore also catches changes
that update only the version. Documentation tests verify README-local links
and check the generated copy. The package workflow also verifies the
installed wheel's long-description metadata against `README.pypi.md`.

`README.pypi.md` **must exist in the checkout** before building a wheel or
source distribution. The clean source exporter in
`assets/clean_repository.ipynb` must include both `README.pypi.md` and
`MANIFEST.in` as selected, Git-tracked root files. An unselected root
file is omitted, while an untracked selected file rejects the export.
After first adding these files, run
`git add README.pypi.md MANIFEST.in` before exporting. Inspect the ZIP
to verify their presence. Do not rely on GitHub Actions to commit the
generated README, and do not generate it only when a PR touches
`README.md`. Both
local and CI builds must consume the same committed description. The
release pipeline should check, not silently regenerate, the file.

## Branch, tag and publication policy

- Beta and release-candidate versions are developed and tagged from
  `develop`, for example `0.1.0b1` → `v0.1.0b1` and `0.1.0rc1` →
  `v0.1.0rc1`.
- Stable versions are merged to `main`, validated there, and tagged from
  `main`, for example `0.1.0` → `v0.1.0`.
- The branch README always follows its own branch. A tag-view README always
  follows that exact immutable tag. The PyPI description for that version
  points to that **same tag**, regardless of the GitHub default branch.
- The tag need not exist when checking a prerelease PR: CI validates paths
  in the current checkout. Create the annotated/published tag only after
  committing and validating the exact release content.
- Before publishing to PyPI, verify that the tag exists and identifies
  the **exact commit used to build** the wheel and source distribution.
  Do not move or reuse a published tag.

After building, inspect the source distribution for both README files,
inspect the wheel's `METADATA` long description, and run `twine check`
when preparing a public upload. Verify the rendered description and images
on the actual PyPI release after publication. Rendering and external asset
availability are not proved merely by an offline generator test.

## Tests and future maintenance

Changes to this contract must be reviewed together with `README.md`, the
generator, `README.pypi.md`, packaging metadata, CI workflows, release
checks, and the focused tests in `tests/setup/test__pypi_readme.py` and
`tests/setup/test__documentation_assets.py`. Never replace the tests with
hard-coded assertions for `main`: the contract is **version-pinned on PyPI
and revision-relative on GitHub**.

To investigate a broken PyPI link, first check the version and tag, then
confirm the target path exists in the corresponding tagged tree. If a
release was built from the wrong commit, issue a new version; do not
mutate an already published tag or attempt to change immutable PyPI files.

## Related

[Development](development.md), [Release procedures](release.md),
[Technical documentation index](README.md).
