---
version: alpha
name: "VisionGuard AI"
description: "A local-first computer-vision operations studio with quiet, evidence-led controls."
colors:
  primary: "#D7A99E"
  background: "#080807"
  panel: "#0F0F0D"
  card: "#141310"
  text: "#F2EFEB"
  muted: "#97928D"
  accent: "#D7A99E"
  accent-soft: "#EAD0C8"
  success: "#62B7B2"
  scrollbar-thumb: "#5F5954"
  scrollbar-thumb-hover: "#8B7E77"
  scrollbar-track: "#0B0B0A"
rounded:
  DEFAULT: "0.8125rem"
  card: "1.3125rem"
  pill: "999px"
spacing:
  page-inline: "2.4rem"
  section-gap: "2.1rem"
  page-max: "92.5rem"
typography:
  sans:
    fontFamily: "Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
components:
  button: {}
  card: {}
  input: {}
  navigation: {}
---

# VisionGuard AI Design System

## Overview

### Creative North Star

The interface borrows from a low-light computer-vision control room: matte instrument panels, precise evidence overlays, and a restrained warm signal color. It is a product surface, not a cinematic security prop.

### Product context and register

- **Audience and primary job:** students, educators, and project evaluators operating local face/object detection and reviewing its evidence.
- **Target market and evidence:** English-language engineering-classroom demonstrations, based on the repository README and student/employee identifiers.
- **Locale and language policy:** English UI with plain, action-led labels; stored timestamps remain database values until rendered by Streamlit.
- **Usage scene:** laptop/desktop use in classrooms and demonstrations, with camera, upload, and data-review tasks.
- **Register:** product UI throughout.
- **Memorable signature:** the concentric scanner-line hero and compact icon navigation.
- **Restraint:** forms, attendance rosters, errors, and privacy controls remain familiar and low-motion.
- **Anti-references:** neon “hacker” dashboards, generic blue SaaS cards, and police-surveillance theatrics.
- **Token ownership/runtime mapping:** `assets/styles.css` is the runtime source; this file documents its durable semantic tokens. UI changes must update both when those tokens change.

## Colors

The near-black background and warm neutral surfaces keep imagery dominant. Rose is the primary action/focus accent; teal is reserved for successful recognition and positive status. Red remains an error/unknown signal. Text and border opacity must retain WCAG 2.2 AA contrast where required.

Product-owned scroll surfaces inherit the documented thumb, hover, and track tokens globally. Forced-colors mode returns scrollbar and focus rendering to system colors.

## Typography

Inter and system sans fallbacks are canonical. Large hero headings carry the visual identity; task labels and body copy use conventional sentence case. IDs, percentages, and counts use compact numeric presentation without decorative monospace.

## Layout

The application uses Streamlit's wide shell, a persistent sidebar, and a 1480px content maximum. Task content follows natural document scrolling. Media columns collapse at Streamlit's responsive breakpoint; no camera or roster control may depend on horizontal scrolling.

## Elevation & Depth

Hierarchy comes from tonal layers, translucent borders, and one soft shadow on major cards/media. Routine inputs and tables remain flat. Blur is limited to the application header and sidebar shell.

## Shapes

Cards use 18–21px radii, controls use 13px radii, and action buttons use pill geometry. Face boxes stay square and functional so the evidence overlay remains precise.

## Components

### Foundational visual states

Enabled controls show hover and visible focus; disabled controls remain non-interactive. Busy work uses Streamlit's stable spinner/status treatment. Success, warning, and error states combine text with color.

### Buttons and actions

Use one rose-accented primary action per decision area. Secondary actions use the same component at lower emphasis. Labels name the outcome: “Record class attendance,” “Mark student present,” and “Finish and mark remaining absent.”

### Navigation and data display

The sidebar is canonical. The compact navigation mirrors every route. Rosters use Streamlit dataframes with explicit column labels, stable widths, and downloadable CSV where useful.

### Forms and overlays

Native Streamlit fields, camera inputs, uploads, tabs, alerts, and progress feedback are canonical. Validation stays inline and preserves capture context. Accepted image types appear beside upload controls.

### Iconography

Material Symbols Rounded is canonical for navigation. Icons supplement readable text on task actions.

### Motion

Hover movement is limited to 1–3px over roughly 200ms. Functional state changes are immediate. Reduced-motion users receive no decorative transforms.

### Content and data visualization

Copy is direct and specific about recognition uncertainty. Percentages are confidence evidence, never guarantees. Tables remain the accessible alternative to visual summaries.

## Do's and Don'ts

- **Do:** keep biometric processing and retention language visible near capture workflows.
- **Do:** reuse the hero, metric cards, Streamlit fields, and shared navigation.
- **Don't:** imply that a face match is infallible or a certified identity decision.
- **Don't:** add decorative neon, dense glass effects, or motion that competes with camera evidence.
