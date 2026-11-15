# UX Contract

## Product context

- Audience: educators, students, engineering-project evaluators, and local operators.
- Primary jobs: register faces, detect media, record attendance, and review/export evidence.
- Target markets: classroom/demo use; no production identity-assurance claim.
- Active locales: English.
- Timezone/calendar policy: application display follows the host/session locale; database timestamps use the existing SQLAlchemy model policy.
- Accessibility target: WCAG 2.2 AA.

## Business-context sources

| Domain / scope | Authoritative source | Source type | Reviewed date |
|---|---|---|---|
| Biometric lifecycle and limits | `README.md` Privacy and Limitations | Product brief | 2026-11-16 |
| Attendance behavior | Current user request | Current-task decision | 2026-11-16 |
| Detection/recognition data model | `src/database.py`, `src/face_recognizer.py` | Verified domain implementation | 2026-11-16 |
| Production access and retention | Deployment environment + OIDC secrets | Operator configuration | 2026-11-16 |

## Visual contract

- Project `DESIGN.md`: `DESIGN.md`.
- Token ownership model: existing runtime canonical.
- Runtime design-system/token source: `assets/styles.css` and shared helpers in `src/ui.py`.
- Supported themes: dark only.
- Design-context review policy: update documentation and runtime tokens together for durable changes.

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Form | Streamlit native fields/forms plus page validation | this contract | registration / capture | unit + browser flow |
| Scrollbar | browser/Streamlit global surface | `assets/styles.css` | none | browser inspection |
| Toast | Streamlit status/alert primitives | this contract | success / warning / info / error | browser flow |
| CRUD | `Database` methods plus page actions | `src/database.py` | create / delete / finalize | unit + full flow |

## Flow ledger

| Operation | Trigger | Pending | Success destination | Success feedback | Failure recovery | Focus outcome | Source ref |
|---|---|---|---|---|---|---|---|
| Register face | Register face | model spinner | same page | inline success | preserve inputs; inline error | form context | `pages/4_Face_Registration.py` |
| Full-class attendance | Analyze class photo | recognition spinner | review roster | annotated evidence + editable roster | keep image; retry/discard | review region | `pages/5_Attendance.py` |
| Save reviewed class | Save reviewed attendance | atomic local mutation | finalized roster | metrics + roster | preserve review; retry | results region | `pages/5_Attendance.py` |
| Individual check-in | Mark student present | recognition spinner | same active session | named success | retain session and retry | next capture | `pages/5_Attendance.py` |
| Finalize attendance | Finalize reviewed attendance | atomic local mutation | finalized roster | present/absent summary | session remains recoverable on failure | summary | `pages/5_Attendance.py` |
| Delete face profile | Delete then confirm | confirmation dialog | same list | profile disappears | cancel keeps profile | surviving list | `pages/4_Face_Registration.py` |

## Navigation and responsive behavior

- Every route sets a unique title through `setup_page`.
- Sidebar navigation is canonical; compact icon navigation mirrors it when collapsed.
- Camera/upload pairs use responsive columns and fall back to natural document flow.
- Focused content must not be obscured by sticky UI.

## Overlays and feedback

- Streamlit inline alerts are canonical; no browser alert/confirm/prompt.
- Biometric and attendance errors remain in context with a retry path.
- Face-profile deletion uses an app-owned confirmation naming the person and irreversible biometric consequence.
- Production access is fail-closed behind Streamlit OIDC plus an explicit email allowlist.

## Async and resilience

- Local mutations are pessimistic and buttons are activated explicitly.
- Streamlit reruns own pending-state completion; individual attendance session IDs persist in session state.
- Recognition failures preserve the active attendance session and allow retry.
- Finalization is idempotent; repeated finalization does not create duplicate rows.
- Full-class attendance and final review commit in one database transaction.
- Unfinished one-by-one sessions can be resumed or explicitly cancelled.
- Production retention runs only from an explicit positive day count and excludes in-progress sessions.

## Validation

- Required names, identifiers, class labels, captures, and one-face constraints use inline validation.
- Database uniqueness protects student identifiers and one attendance row per student/session.
- Attendance capture is blocked when any roster face embedding is missing or invalid; history stays available and Face Registration identifies the profile that must be repaired.
- Face images and embeddings remain local and must never appear in toast/error text as raw data.
- Automated matches are always reviewable before a session becomes final.

## Permission and deployment policy

- Development mode is local-only and identifies mutations as `local-development`.
- Production mode requires OIDC, `VISIONGUARD_ALLOWED_EMAILS`, and `VISIONGUARD_ATTENDANCE_RETENTION_DAYS`; missing configuration locks the app.
- The allowlist is enforced before any page runs. Institution-specific roles remain unresolved until an authoritative permission policy is provided.
- Production is single-instance while SQLite is the persistence layer.

## Verification

- Required static commands: compileall, unittest, premium static audit.
- Browser matrix: current desktop browser plus a narrow responsive viewport.
- Canonical sibling flow: Face Registration and Image Detection.
- Failure paths: no image, no face, multiple faces in one-by-one mode, unknown face, duplicate check-in, and empty history.
