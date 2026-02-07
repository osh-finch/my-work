# X (Twitter) Pre-break Scraper

Lightweight Playwright scraper for early signal detection. It polls the Latest search feed for configured queries, stores samples in SQLite, and surfaces velocity spikes versus a baseline. This is for early signal detection, not perfect coverage.

## Install

```bash
nvm use 20
npm install
npx playwright install chromium
```

Use Node 20 for the most reliable Playwright + native module compatibility.

## Login (once)

```bash
npm run login
```

A browser window opens using a persistent Chrome profile. Log in to X manually, then close the window (or wait for X Home to load) to save the session.

The default profile directory is `~/.playwright-profiles/x-profile`. You can override it by setting `X_PROFILE_DIR` in a `.env` file, for example:

```bash
X_PROFILE_DIR=~/.playwright-profiles/x-profile
```

Run `npm run test:session` to confirm the session is active.

```bash
npm run test:session
```

### Cookie-based login (use if the login form fails)

If X shows an error like "Could not log you in now. Please try again later", use a cookie-based session instead of the login form.

1) Log in to X in your normal browser.
2) Open DevTools -> Application/Storage -> Cookies -> `https://x.com`.
3) Copy the `auth_token` value (and `ct0` if present).
4) Run:

```bash
npm run login:cookie
```

This saves `X_AUTH_TOKEN` (and optionally `X_CT0`) to `.env`, and the scraper will inject them on launch. Then re-run:

```bash
npm run test:session
```

## Configure queries

Edit `config/queries.json` and set `enabled` to `true` for the queries you want. X search operators are supported.

## Collect

```bash
npm run collect
```

The collector:
- Uses low-frequency polling and small per-query limits.
- Deduplicates by tweet ID.
- Saves structured JSON summaries per run and stores tweets in `data/x.sqlite`.

## Anomalies

```bash
npm run anomalies
```

This prints a small table plus JSON for the top 10 velocity spikes. Velocity is measured as new tweets in the last 10 minutes, with a baseline mean over the previous 6 hours (or the last 50 windows if there is not enough history).

## Notes

- This is sampling for early signal detection. It will miss tweets and is not intended for exhaustive coverage.
- If you see a login wall, run `npm run login` again.
- Run `npm run login` once per machine, then run other scripts.
