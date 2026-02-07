const fs = require('fs');
const path = require('path');
const Database = require('better-sqlite3');

const WINDOW_MINUTES = 10;
const BASELINE_HOURS = 6;
const MAX_FALLBACK_WINDOWS = 50;

function toMs(minutes) {
  return minutes * 60 * 1000;
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function pad(value, length) {
  const text = String(value);
  return text.length >= length ? text : text + ' '.repeat(length - text.length);
}

async function main() {
  const dbPath = path.resolve('data', 'x.sqlite');
  if (!fs.existsSync(dbPath)) {
    console.error('No data found. Run npm run collect first.');
    process.exitCode = 1;
    return;
  }

  const db = new Database(dbPath);
  const queryNames = db.prepare('SELECT DISTINCT query_name AS name FROM tweets').all();

  if (!queryNames.length) {
    console.error('No tweets stored yet. Run npm run collect.');
    db.close();
    process.exitCode = 1;
    return;
  }

  const now = Date.now();
  const windowMs = toMs(WINDOW_MINUTES);
  const currentWindowStart = now - windowMs;

  const results = [];

  for (const row of queryNames) {
    const name = row.name;
    const minRow = db.prepare('SELECT MIN(collected_at) AS earliest FROM tweets WHERE query_name = ?').get(name);
    if (!minRow || !minRow.earliest) continue;

    const earliestTs = new Date(minRow.earliest).getTime();
    if (!Number.isFinite(earliestTs)) continue;

    const maxAvailableWindows = Math.max(0, Math.floor((currentWindowStart - earliestTs) / windowMs));
    let baselineWindowCount = BASELINE_HOURS * 60 / WINDOW_MINUTES;
    if (maxAvailableWindows < baselineWindowCount) {
      baselineWindowCount = Math.min(MAX_FALLBACK_WINDOWS, maxAvailableWindows);
    }

    if (baselineWindowCount < 1) continue;

    const baselineStart = currentWindowStart - baselineWindowCount * windowMs;
    const rows = db.prepare(
      'SELECT collected_at FROM tweets WHERE query_name = ? AND collected_at >= ?'
    ).all(name, new Date(baselineStart).toISOString());

    const baselineCounts = new Array(baselineWindowCount).fill(0);
    let currentCount = 0;

    for (const tweet of rows) {
      const ts = new Date(tweet.collected_at).getTime();
      if (!Number.isFinite(ts)) continue;
      if (ts >= currentWindowStart) {
        currentCount += 1;
      } else if (ts >= baselineStart) {
        const index = Math.floor((ts - baselineStart) / windowMs);
        if (index >= 0 && index < baselineWindowCount) {
          baselineCounts[index] += 1;
        }
      }
    }

    const mean = baselineCounts.reduce((a, b) => a + b, 0) / baselineCounts.length;
    const variance = baselineCounts.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / baselineCounts.length;
    const stdDev = Math.sqrt(variance);

    let score = 0;
    let zScore = null;
    if (stdDev > 0) {
      zScore = (currentCount - mean) / stdDev;
      score = zScore;
    } else if (mean > 0) {
      score = currentCount / mean;
    } else {
      score = currentCount;
    }

    results.push({
      query_name: name,
      current_10m: currentCount,
      baseline_mean: mean,
      baseline_std: stdDev,
      z_score: zScore,
      score
    });
  }

  results.sort((a, b) => b.score - a.score);
  const top = results.slice(0, 10);

  if (!top.length) {
    console.log('No anomalies calculated yet.');
    db.close();
    return;
  }

  const headers = [
    pad('Query', 26),
    pad('Now(10m)', 9),
    pad('BaseMean', 9),
    pad('Z', 7),
    pad('Score', 7)
  ];
  console.log(headers.join(' '));
  console.log('-'.repeat(headers.join(' ').length));

  for (const row of top) {
    console.log([
      pad(row.query_name, 26),
      pad(row.current_10m, 9),
      pad(formatNumber(row.baseline_mean, 2), 9),
      pad(formatNumber(row.z_score, 2), 7),
      pad(formatNumber(row.score, 2), 7)
    ].join(' '));
  }

  console.log(JSON.stringify({
    generated_at: new Date().toISOString(),
    window_minutes: WINDOW_MINUTES,
    baseline_hours: BASELINE_HOURS,
    results: top
  }, null, 2));

  db.close();
}

main().catch((err) => {
  console.error('Anomalies failed:', err);
  process.exitCode = 1;
});
