const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const Database = require('better-sqlite3');
const { ensureLoggedIn, isLoginUrl, launchContext } = require('./x-session');

const RUNTIME_DEFAULTS = {
  pollSeconds: 120,
  maxTweetsPerQuery: 20,
  headless: true,
  minDelayMs: 800,
  maxDelayMs: 2000,
  maxScrolls: 3
};

function readJson(filePath, fallback) {
  if (!fs.existsSync(filePath)) return fallback;
  const raw = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(raw);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomBetween(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

async function saveDebugArtifacts(page, label, content) {
  if (!process.env.X_DEBUG) return;
  const debugDir = path.resolve('data', 'debug');
  if (!fs.existsSync(debugDir)) fs.mkdirSync(debugDir, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const safeLabel = label.replace(/[^a-z0-9-_]+/gi, '-').slice(0, 60);
  const base = path.join(debugDir, `${timestamp}-${safeLabel}`);

  try {
    const html = content || await page.content();
    fs.writeFileSync(`${base}.html`, html, 'utf8');
  } catch (err) {
    // Best-effort debug capture.
  }

  try {
    await page.screenshot({ path: `${base}.png`, fullPage: true });
  } catch (err) {
    // Best-effort debug capture.
  }

  try {
    const meta = {
      url: page.url(),
      title: await page.title()
    };
    fs.writeFileSync(`${base}.json`, JSON.stringify(meta, null, 2), 'utf8');
  } catch (err) {
    // Best-effort debug capture.
  }
}

async function handleCookieBanner(page) {
  const accept = page.locator('button:has-text("Accept all cookies")');
  const refuse = page.locator('button:has-text("Refuse non-essential cookies")');
  try {
    if (await accept.isVisible({ timeout: 1500 })) {
      await accept.click({ timeout: 1500 });
      await page.waitForTimeout(500);
      return true;
    }
  } catch (err) {
    // ignore
  }
  try {
    if (await refuse.isVisible({ timeout: 1500 })) {
      await refuse.click({ timeout: 1500 });
      await page.waitForTimeout(500);
      return true;
    }
  } catch (err) {
    // ignore
  }
  return false;
}

async function handleScriptLoadFailure(page, content) {
  if (!content || !content.includes('Something went wrong')) return false;
  const retryButton = page.locator('button:has-text("Try again")');
  try {
    if (await retryButton.isVisible({ timeout: 1500 })) {
      await retryButton.click({ timeout: 1500 });
      await page.waitForTimeout(1000);
      return true;
    }
  } catch (err) {
    // ignore
  }
  return false;
}

async function detectRateLimit(page) {
  try {
    return await page.evaluate(() => {
      const needles = [/rate limit/i, /too many requests/i, /try again later/i];
      const candidates = Array.from(document.querySelectorAll('[role="alert"], [data-testid="toast"], [data-testid="error-detail"]'));
      return candidates.some((el) => {
        const text = (el.textContent || '').trim();
        if (!text) return false;
        return needles.some((re) => re.test(text));
      });
    });
  } catch (err) {
    return false;
  }
}

function attachDebugNetworkLogging(page) {
  if (!process.env.X_DEBUG) return;

  page.on('requestfailed', (req) => {
    const failure = req.failure();
    const message = failure ? failure.errorText : 'unknown error';
    console.error(`[requestfailed] ${req.method()} ${req.url()} -> ${message}`);
  });

  page.on('response', (res) => {
    const status = res.status();
    if (status >= 400) {
      console.error(`[response ${status}] ${res.url()}`);
    }
  });
}

function ensureSchema(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS tweets (
      id TEXT PRIMARY KEY,
      query_name TEXT,
      query_string TEXT,
      created_at TEXT,
      collected_at TEXT,
      author_handle TEXT,
      text TEXT,
      url TEXT
    );

    CREATE TABLE IF NOT EXISTS runs (
      run_id TEXT PRIMARY KEY,
      started_at TEXT,
      finished_at TEXT,
      notes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_tweets_query_collected ON tweets(query_name, collected_at);
    CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at);
  `);
}

async function navigateWithRetry(page, url) {
  const maxAttempts = 3;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    } catch (err) {
      if (attempt === maxAttempts) throw err;
    }

    await handleCookieBanner(page);

    const currentUrl = page.url();
    if (isLoginUrl(currentUrl)) {
      return { status: 'login' };
    }

    const content = await page.content();
    if (content.includes('Something went wrong')) {
      const retried = await handleScriptLoadFailure(page, content);
      if (retried) {
        const retriedContent = await page.content();
        if (!retriedContent.includes('Something went wrong')) {
          return { status: 'ok' };
        }
      }
      await saveDebugArtifacts(page, 'something-went-wrong', content);
      if (attempt === maxAttempts) {
        return { status: 'error', message: 'Something went wrong' };
      }
      const backoff = 1000 * Math.pow(2, attempt - 1) + randomBetween(250, 750);
      await sleep(backoff);
      continue;
    }

    const rateLimited = await detectRateLimit(page);
    if (rateLimited) {
      await saveDebugArtifacts(page, 'rate-limit', content);
      if (attempt === maxAttempts) {
        return { status: 'error', message: 'Rate limiting detected' };
      }
      const backoff = 2000 * Math.pow(2, attempt - 1) + randomBetween(500, 1500);
      await sleep(backoff);
      continue;
    }

    return { status: 'ok' };
  }

  return { status: 'error', message: 'Navigation failed' };
}

async function extractTweets(page) {
  return page.$$eval('article[role="article"]', (articles) => {
    return articles.map((article) => {
      const statusLink = article.querySelector('a[href*="/status/"]');
      if (!statusLink) return null;
      const href = statusLink.getAttribute('href');
      const idMatch = href ? href.match(/\/status\/(\d+)/) : null;
      const id = idMatch ? idMatch[1] : null;
      const timeEl = article.querySelector('time');
      const createdAt = timeEl ? timeEl.getAttribute('datetime') : null;
      const textEl = article.querySelector('[data-testid="tweetText"]');
      const text = textEl ? textEl.textContent : null;
      const handleEl = Array.from(article.querySelectorAll('a[role="link"]')).find((a) => {
        const textContent = (a.textContent || '').trim();
        return textContent.startsWith('@');
      });
      const authorHandle = handleEl ? handleEl.textContent.trim() : null;
      const url = href ? `https://x.com${href}` : null;
      return {
        id,
        createdAt,
        text,
        authorHandle,
        url
      };
    }).filter(Boolean);
  });
}

function buildSearchQuery(queryObj) {
  const lang = queryObj.lang || 'en';
  if (/\blang:/i.test(queryObj.query || '')) return queryObj.query;
  return `${queryObj.query} lang:${lang}`.trim();
}

async function collectQuery(page, queryObj, runtime, insertStmt) {
  const searchQuery = buildSearchQuery(queryObj);
  const url = `https://x.com/search?q=${encodeURIComponent(searchQuery)}&src=typed_query&f=live`;

  const navResult = await navigateWithRetry(page, url);
  if (navResult.status === 'login') {
    return { status: 'login_required' };
  }
  if (navResult.status !== 'ok') {
    return { status: 'error', error: navResult.message || 'Navigation error' };
  }

  const seenThisRun = new Set();
  let totalProcessed = 0;
  let newCount = 0;
  let scrolls = 0;

  while (totalProcessed < runtime.maxTweetsPerQuery) {
    await page.waitForTimeout(randomBetween(600, 1200));

    const content = await page.content();
    if (content.includes('Something went wrong')) {
      const retried = await handleScriptLoadFailure(page, content);
      if (retried) {
        const retriedContent = await page.content();
        if (!retriedContent.includes('Something went wrong')) {
          continue;
        }
      }
      await saveDebugArtifacts(page, 'something-went-wrong', content);
      return { status: 'error', error: 'Something went wrong' };
    }
    const rateLimited = await detectRateLimit(page);
    if (rateLimited) {
      await saveDebugArtifacts(page, 'rate-limit', content);
      return { status: 'error', error: 'Rate limiting detected' };
    }

    const tweets = await extractTweets(page);
    if (!tweets.length && scrolls === 0) {
      await page.waitForTimeout(1500);
    }

    for (const tweet of tweets) {
      if (!tweet.id || seenThisRun.has(tweet.id)) continue;
      seenThisRun.add(tweet.id);

      const collectedAt = new Date().toISOString();
      const result = insertStmt.run(
        tweet.id,
        queryObj.name,
        queryObj.query,
        tweet.createdAt,
        collectedAt,
        tweet.authorHandle,
        tweet.text,
        tweet.url
      );

      if (result.changes > 0) newCount += 1;
      totalProcessed += 1;

      if (totalProcessed >= runtime.maxTweetsPerQuery) break;
    }

    if (totalProcessed >= runtime.maxTweetsPerQuery) break;

    if (scrolls >= runtime.maxScrolls) break;

    await page.evaluate(() => {
      window.scrollBy(0, window.innerHeight * 2);
    });

    scrolls += 1;
  }

  if (totalProcessed === 0) {
    return { status: 'empty', newCount };
  }

  return { status: 'ok', processed: totalProcessed, newCount };
}

async function main() {
  const queriesPath = path.resolve('config', 'queries.json');
  const runtimePath = path.resolve('config', 'runtime.json');
  const queries = readJson(queriesPath, []);
  const runtime = { ...RUNTIME_DEFAULTS, ...readJson(runtimePath, {}) };

  if (!Array.isArray(queries) || queries.length === 0) {
    console.error('No queries found. Add items to config/queries.json.');
    process.exitCode = 1;
    return;
  }

  const enabledQueries = queries.filter((q) => q.enabled);
  if (enabledQueries.length === 0) {
    console.error('No enabled queries. Set enabled=true in config/queries.json.');
    process.exitCode = 1;
    return;
  }

  const dataDir = path.resolve('data');
  if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
  const dbPath = path.join(dataDir, 'x.sqlite');

  const db = new Database(dbPath);
  ensureSchema(db);

  const runId = crypto.randomUUID();
  const startedAt = new Date().toISOString();
  const insertRunStmt = db.prepare(
    'INSERT INTO runs (run_id, started_at, finished_at, notes) VALUES (?, ?, ?, ?)'
  );
  insertRunStmt.run(runId, startedAt, null, null);

  const insertTweetStmt = db.prepare(
    `INSERT OR IGNORE INTO tweets
      (id, query_name, query_string, created_at, collected_at, author_handle, text, url)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
  );

  const context = await launchContext();
  const page = context.pages()[0] || await context.newPage();
  attachDebugNetworkLogging(page);

  const loggedIn = await ensureLoggedIn(page);
  if (!loggedIn) {
    await context.close();
    db.close();
    console.error('Not logged in. Run npm run login.');
    process.exitCode = 1;
    return;
  }

  const summary = {
    run_id: runId,
    started_at: startedAt,
    finished_at: null,
    runtime,
    queries: [],
    errors: []
  };

  let loginRequired = false;

  for (let i = 0; i < enabledQueries.length; i += 1) {
    const queryObj = enabledQueries[i];

    if (i > 0) {
      await sleep(randomBetween(runtime.minDelayMs, runtime.maxDelayMs));
    }

    const queryResult = {
      name: queryObj.name,
      query: queryObj.query,
      status: 'ok',
      new_tweets: 0,
      processed: 0,
      error: null
    };

    try {
      const result = await collectQuery(page, queryObj, runtime, insertTweetStmt);
      if (result.status === 'login_required') {
        loginRequired = true;
        queryResult.status = 'login_required';
        queryResult.error = 'Run npm run login';
      } else if (result.status !== 'ok') {
        queryResult.status = result.status;
        queryResult.error = result.error || 'Unknown error';
      } else {
        queryResult.new_tweets = result.newCount || 0;
        queryResult.processed = result.processed || 0;
      }
    } catch (err) {
      queryResult.status = 'error';
      queryResult.error = err.message;
    }

    summary.queries.push(queryResult);

    if (queryResult.status !== 'ok') {
      summary.errors.push({ query: queryObj.name, error: queryResult.error });
    }

    if (loginRequired) break;
  }

  await context.close();

  const finishedAt = new Date().toISOString();
  summary.finished_at = finishedAt;

  const updateRunStmt = db.prepare(
    'UPDATE runs SET finished_at = ?, notes = ? WHERE run_id = ?'
  );
  updateRunStmt.run(finishedAt, JSON.stringify(summary), runId);

  db.close();

  console.log(JSON.stringify(summary, null, 2));

  if (loginRequired) {
    console.error('Login required. Run npm run login.');
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error('Collector failed:', err);
  process.exitCode = 1;
});
