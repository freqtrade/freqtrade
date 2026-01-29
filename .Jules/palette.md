## 2026-01-29 - Telegram Visual Parsing
**Learning:** In chat-based interfaces (Telegram), distinct geometric shapes and colors (🔴, 🟢, 🛑, 🔻) parse faster than generic icons (❌, ⚠) for critical trading signals.
**Action:** Prioritize unique shape/color combinations for distinct trading states (Entry, Exit, Stoploss).

## 2026-01-29 - CLI Feedback
**Learning:** Silent long-running operations (like UI download) cause user uncertainty ("Did it hang?").
**Action:** Always implement a progress bar (e.g., via `rich`) for network operations > 2 seconds.
