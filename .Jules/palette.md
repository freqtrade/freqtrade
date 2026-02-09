## 2026-05-21 - Print Styles Matter
**Learning:** Default CSS often hides critical content in print view. The fallback UI hid the installation command when printing, making the printed page useless.
**Action:** Always verify `@media print` styles for instructional content to ensure key information (commands, codes) remains visible.

## 2026-06-15 - Interactive Code Affordance
**Learning:** Adding "terminal window controls" (red/yellow/green dots) to code blocks instantly communicates "this is a command" and "this is interactive", improving scannability.
**Action:** Style critical CLI commands as terminal windows to reinforce their purpose and interactivity.

## 2026-06-15 - Contextual Copy Feedback
**Learning:** Users often look at the text they are copying, not the button they clicked. Flashing the container of the copied text (e.g., green border) provides stronger confirmation than just button text changes.
**Action:** Enhance copy actions by visually confirming the *source* element changed state, not just the trigger button.

## 2026-06-15 - Lightweight Accessibility
**Learning:** In dependency-free environments (like standalone HTML), CSS-only tooltips using `attr(aria-label)` provide excellent accessibility feedback without JavaScript overhead or external libraries.
**Action:** Use `[aria-label]:hover::after` pattern for lightweight, accessible tooltips in static pages to enhance UX without bloat.
