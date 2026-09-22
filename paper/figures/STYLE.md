# Manuscript figure style contract

Status: frozen default approved by the owner on 2026-09-22

## Backend and scope

- Use Python with Matplotlib for all manuscript plotting, previews, exports, and
  visual QA.
- Do not use R unless the owner explicitly changes this preference.
- Keep one consistent visual system across the manuscript and supplement.

## Scientific and layout rules

- Start every figure from one Results-level claim and retain only panels with a
  distinct inferential role.
- Target IEEE-style final widths of 89 mm (single column) or 183 mm (double column).
- Use a white background, sans-serif text, restrained colors, no decorative boxes,
  no rainbow maps, and normally no top or right spines.
- Use 7--9 pt text at final size and require every rendered PDF glyph to be at least
  5 pt. Panel labels are bold lowercase letters near the upper-left plot boundary.
- Reuse semantic colors: oracle/reference in neutral dark gray, audited model in blue,
  coarsened or supplementary quantities in muted teal, and threshold boundaries in a
  restrained red accent. Color must not be the only differentiator.
- For comparable stochastic summaries, show the same interval definition and state
  the replicate unit, sample size, center statistic, and interval in the legend.
- Select representative series only through the frozen deterministic rule; never by
  visual appeal.

## Export rules

- Export line, bar, scatter, and other ordinary chart panels as vector PDF and
  editable-text SVG.
- Use high-resolution PNG only when raster content is scientifically necessary, such
  as sample images or dense heatmaps. Default to 600 dpi for those exports.
- Keep Matplotlib PDF fonts editable (`pdf.fonttype = 42`) and SVG text editable
  (`svg.fonttype = 'none'`).
- Preserve plotting source, clean source-data tables, final PDF/SVG, optional PNG
  preview, alignment report, collision report, and panel-level QA notes.

## Blocking QA

- Audit multi-panel plot-area alignment at the final physical dimensions with a
  1.5-point tolerance.
- Audit rendered PDF text for the 5-point glyph floor and inspect collisions,
  clipping, label clearance, grayscale legibility, and color-vision robustness.
- Inspect each panel and the assembled figure at final size after every
  layout-affecting revision. A figure with unresolved overlap, clipping, unreadable
  labels, or ambiguous uncertainty is not submission-ready.
