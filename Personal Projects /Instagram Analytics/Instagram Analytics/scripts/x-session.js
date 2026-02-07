const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const { chromium, firefox } = require('playwright');
const { getProfileDir, loadDotEnv } = require('./profile');

const LOGGED_IN_SELECTORS = [
  '[data-testid="SideNav_AccountSwitcher_Button"]',
  '[data-testid="AppTabBar_Home_Link"]'
];

function isLoginUrl(url) {
  return url.includes('/login') || url.includes('/i/flow/login');
}

async function launchContext() {
  const userDataDir = getProfileDir();
  cleanupChromeProfileLocks(userDataDir);
  const contextOptions = {
    locale: 'en-GB',
    timezoneId: 'Europe/London'
  };
  const cookies = buildAuthCookies();
  const userAgent = resolveUserAgent();

  const cdpEndpoint = resolveCdpEndpoint();
  if (cdpEndpoint) {
    return connectOverCdp(cdpEndpoint, contextOptions, cookies, userAgent);
  }
  if (process.env.X_CDP_AUTOLAUNCH === '1') {
    const cdpContext = await autoLaunchAndConnectCdp(userDataDir, contextOptions, cookies, userAgent);
    return cdpContext;
  }

  const launchOptions = buildLaunchOptions(userDataDir);
  const executablePath = resolveChromiumExecutable();
  if (executablePath) {
    launchOptions.executablePath = executablePath;
  }

  const preferredBrowser = (process.env.X_BROWSER || '').toLowerCase().trim();
  const tryChromiumFirst = !preferredBrowser || preferredBrowser === 'chromium';
  const noFallback = preferredBrowser === 'chromium' || process.env.X_NO_FALLBACK === '1';

  if (tryChromiumFirst) {
    const context = await tryLaunchChromium(userDataDir, launchOptions, contextOptions, cookies, userAgent);
    if (context) return context;
    if (noFallback) {
      throw new Error('Chromium launch failed and fallback is disabled (X_NO_FALLBACK=1 or X_BROWSER=chromium).');
    }
  }

  const firefoxPreferred = preferredBrowser === 'firefox' || !preferredBrowser || preferredBrowser === 'chromium';
  if (firefoxPreferred) {
    const context = await tryLaunchFirefox(userDataDir, contextOptions, cookies, userAgent);
    if (context) return context;
  }

  const context = await launchEphemeralContext(launchOptions, contextOptions, cookies, userAgent);
  return context;
}

async function launchEphemeralContext(launchOptions, contextOptions, cookies, userAgent) {
  const headlessExecutable = resolveHeadlessShellExecutable();
  const browser = await chromium.launch({
    ...launchOptions,
    headless: true,
    executablePath: headlessExecutable || launchOptions.executablePath
  });
  const context = await browser.newContext(buildContextOptions(contextOptions, userAgent));

  if (cookies.length) {
    await context.addCookies(cookies);
  }

  const originalClose = context.close.bind(context);
  context.close = async () => {
    await originalClose();
    await browser.close();
  };

  return context;
}

async function tryLaunchChromium(userDataDir, launchOptions, contextOptions, cookies, userAgent) {
  try {
    const context = await chromium.launchPersistentContext(userDataDir, {
      ...launchOptions,
      ...buildContextOptions(contextOptions, userAgent)
    });

    if (cookies.length) {
      await context.addCookies(cookies);
    }

    return context;
  } catch (err) {
    if (process.env.X_DEBUG) {
      console.error('[chromium] launchPersistentContext failed:', err.message);
    }
    return null;
  }
}

async function tryLaunchFirefox(userDataDir, contextOptions, cookies, userAgent) {
  try {
    const context = await firefox.launchPersistentContext(userDataDir, {
      headless: false,
      ...buildContextOptions(contextOptions, userAgent)
    });

    if (cookies.length) {
      await context.addCookies(cookies);
    }

    return context;
  } catch (err) {
    return null;
  }
}

function buildContextOptions(contextOptions, userAgent) {
  if (!userAgent) return contextOptions;
  return {
    ...contextOptions,
    userAgent
  };
}

function resolveChromiumExecutable() {
  loadDotEnv();

  const explicitPath = process.env.X_CHROMIUM_EXECUTABLE_PATH || process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;
  if (explicitPath && explicitPath.trim()) {
    return explicitPath.trim();
  }

  const cacheRoot = path.join(process.env.HOME || '', 'Library', 'Caches', 'ms-playwright');
  const candidates = [
    path.join(cacheRoot, 'chromium-1208', 'chrome-mac-arm64', 'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing'),
    path.join(cacheRoot, 'chromium-1208', 'chrome-mac-x64', 'Google Chrome for Testing.app', 'Contents', 'MacOS', 'Google Chrome for Testing')
  ];

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }

  return undefined;
}

function resolveHeadlessShellExecutable() {
  const cacheRoot = path.join(process.env.HOME || '', 'Library', 'Caches', 'ms-playwright');
  const candidates = [
    path.join(cacheRoot, 'chromium_headless_shell-1208', 'chrome-headless-shell-mac-arm64', 'chrome-headless-shell'),
    path.join(cacheRoot, 'chromium_headless_shell-1208', 'chrome-headless-shell-mac-x64', 'chrome-headless-shell')
  ];

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }

  return undefined;
}

function cleanupChromeProfileLocks(userDataDir) {
  const lockFiles = ['SingletonLock', 'SingletonCookie', 'SingletonSocket'];
  for (const name of lockFiles) {
    const fullPath = path.join(userDataDir, name);
    try {
      if (fs.existsSync(fullPath)) fs.rmSync(fullPath);
    } catch (err) {
      // Best effort; stale locks shouldn't block launch if removal fails.
    }
  }
}

function buildLaunchOptions(userDataDir) {
  const minFlags = process.env.X_MIN_FLAGS === '1';
  const ignoreDefaultArgs = process.env.X_IGNORE_DEFAULT_ARGS === '1';
  const args = [];

  if (!minFlags) {
    args.push(
      '--disable-crashpad',
      '--disable-crash-reporter',
      `--crash-dumps-dir=${path.join(userDataDir, 'crashpad')}`
    );
  }

  return {
    headless: false,
    args,
    ignoreDefaultArgs: ignoreDefaultArgs || undefined
  };
}

function resolveCdpEndpoint() {
  const endpoint = process.env.X_CDP_ENDPOINT;
  if (endpoint && endpoint.trim()) return endpoint.trim();
  const port = process.env.X_CDP_PORT;
  if (port && port.trim()) return `http://127.0.0.1:${port.trim()}`;
  return null;
}

async function connectOverCdp(endpoint, contextOptions, cookies, userAgent) {
  const browser = await chromium.connectOverCDP(endpoint);
  let context = browser.contexts()[0];
  if (!context) {
    context = await browser.newContext(buildContextOptions(contextOptions, userAgent));
  }

  if (cookies.length) {
    await context.addCookies(cookies);
  }

  const originalClose = context.close.bind(context);
  context.close = async () => {
    try {
      const pages = context.pages();
      await Promise.all(pages.map((page) => page.close().catch(() => {})));
    } catch (err) {
      // best effort
    }

    if (process.env.X_CDP_CLOSE_BROWSER === '1') {
      await originalClose();
      await browser.close();
    }
  };

  return context;
}

async function autoLaunchAndConnectCdp(userDataDir, contextOptions, cookies, userAgent) {
  const port = process.env.X_CDP_PORT || '9222';
  const chromePath = resolveChromiumExecutable() || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
  const profileDir = path.resolve(process.env.X_CDP_PROFILE_DIR || path.join(userDataDir, 'cdp-profile'));

  const args = [
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profileDir}`,
    '--no-first-run',
    '--no-default-browser-check',
    'about:blank'
  ];

  const child = spawn(chromePath, args, {
    stdio: 'ignore',
    detached: false
  });

  const endpoint = `http://127.0.0.1:${port}`;
  await waitForCdp(endpoint, 15000);
  const context = await connectOverCdp(endpoint, contextOptions, cookies, userAgent);

  const originalClose = context.close.bind(context);
  context.close = async () => {
    await originalClose();
    if (process.env.X_CDP_CLOSE_BROWSER === '1') {
      try {
        child.kill();
      } catch (err) {
        // best effort
      }
    }
  };

  return context;
}

function waitForCdp(endpoint, timeoutMs) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const req = http.get(`${endpoint}/json/version`, (res) => {
        res.resume();
        if (res.statusCode === 200) {
          resolve();
        } else {
          retry();
        }
      });
      req.on('error', retry);
    };

    const retry = () => {
      if (Date.now() - start > timeoutMs) {
        reject(new Error('Timed out waiting for CDP endpoint'));
        return;
      }
      setTimeout(attempt, 300);
    };

    attempt();
  });
}

function resolveUserAgent() {
  loadDotEnv();
  const ua = process.env.X_USER_AGENT;
  if (ua && ua.trim()) return ua.trim();
  return 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36';
}

function buildAuthCookies() {
  loadDotEnv();
  const authToken = process.env.X_AUTH_TOKEN || process.env.AUTH_TOKEN;
  const ct0 = process.env.X_CT0 || process.env.CT0;
  const cookies = [];

  if (authToken && authToken.trim()) {
    cookies.push({
      name: 'auth_token',
      value: authToken.trim(),
      domain: '.x.com',
      path: '/'
    });
  }

  if (ct0 && ct0.trim()) {
    cookies.push({
      name: 'ct0',
      value: ct0.trim(),
      domain: '.x.com',
      path: '/'
    });
  }

  return cookies;
}

async function isLoggedIn(page) {
  const url = page.url();
  if (isLoginUrl(url)) return false;

  for (const selector of LOGGED_IN_SELECTORS) {
    const handle = await page.$(selector);
    if (handle) return true;
  }

  return !isLoginUrl(url);
}

async function ensureLoggedIn(page) {
  await page.goto('https://x.com/home', { waitUntil: 'domcontentloaded' });
  return isLoggedIn(page);
}

async function waitForLoggedIn(page) {
  const homePromise = page.waitForURL(/https:\/\/x\.com\/home.*/, { timeout: 0 }).then(() => true);
  const selectorPromise = page.waitForSelector(LOGGED_IN_SELECTORS.join(','), { timeout: 0 }).then(() => true);
  return Promise.race([homePromise, selectorPromise]);
}

module.exports = {
  ensureLoggedIn,
  isLoggedIn,
  isLoginUrl,
  launchContext,
  waitForLoggedIn
};
