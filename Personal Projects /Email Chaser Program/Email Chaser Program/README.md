# WhatsApp Chaser Assistant (CLI)

This tool helps you draft and send WhatsApp chaser messages from a Google Sheet, with manual approval before any messages are sent.

## Requirements
- `OPENAI_API_KEY` set in your environment
- Google service account `credentials.json` in the repo root

## Quick start
```bash
export OPENAI_API_KEY="your_key"
python cli_chaser.py
```

You will be prompted for:
- Google Sheet URL or ID
- A natural language instruction, for example:
  - `Message anyone who said "I'm unsure because selection has not yet occurred" to ask if they’re going up that evening.`

## Cooldown guard
By default, recipients messaged in the last 7 days are excluded unless you override.

Flags:
```bash
python cli_chaser.py --cooldown-days 7
python cli_chaser.py --include-recent
python cli_chaser.py --dry-run
```

## History database
Message history is stored in:
```
.app_state/history.db
```

It tracks the last message per phone number to support cooldown warnings and previews.

## Notes
- Manual approval is always required before sending.
- The WhatsApp sending mechanism is unchanged.
