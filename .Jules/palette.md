## 2026-01-29 - Telegram Visual Parsing
**Learning:** In chat-based interfaces (Telegram), distinct geometric shapes and colors (🔴, 🟢, 🛑, 🔻) parse faster than generic icons (❌, ⚠) for critical trading signals.
**Action:** Prioritize unique shape/color combinations for distinct trading states (Entry, Exit, Stoploss).

## 2026-01-29 - CLI Feedback
**Learning:** Silent long-running operations (like UI download) cause user uncertainty ("Did it hang?").
**Action:** Always implement a progress bar (e.g., via `rich`) for network operations > 2 seconds.

## 2026-01-30 - Dead-end Fallback UX
**Learning:** Static error pages (like "UI not installed") act as dead ends, forcing users to manually refresh. This breaks the flow during setup.
**Action:** Always provide a "Check Again" or "Retry" action on state-dependent error pages to keep the user in the flow.

## 2026-01-30 - Text-based Card UI
**Learning:** In constraint-heavy text interfaces (Telegram), dense Key:Value lists are hard to scan. Grouping related data (Header with ID, PnL with Emoji, Details below) mimics a "Card" UI pattern effectively.
**Action:** Use whitespace and grouping to create visual "cards" in text messages instead of flat lists.

## 2026-02-01 - CLI Visual Rhythm
**Learning:** Consistent visual rhythm (zebra striping) in CLI tables significantly reduces eye strain when scanning long lists of data (like exchanges or strategies).
**Action:** Enforce `row_styles=["", "dim"]` for all `rich` tables in the CLI.

## 2026-02-01 - Command Feedback Emojis
**Learning:** Adding status emojis (✅, 🛑) to bot control command responses gives immediate positive feedback and reassurance compared to plain text status.
**Action:** Use status-indicating emojis for all state-change command responses.
