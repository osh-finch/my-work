const { launchContext, waitForLoggedIn } = require('./x-session');

async function main() {
  const context = await launchContext();
  const page = context.pages()[0] || await context.newPage();

  await page.goto('https://x.com/login', { waitUntil: 'domcontentloaded' });

  console.log('Log in to X in the opened browser window.');
  console.log('Close the window when finished, or wait until X home loads.');

  const closePromise = new Promise((resolve) => {
    context.on('close', resolve);
  });

  const loginPromise = waitForLoggedIn(page)
    .then(() => 'home')
    .catch(() => null);

  const result = await Promise.race([closePromise, loginPromise]);
  if (result === 'home') {
    await context.close();
  }
}

main().catch((err) => {
  console.error('Login failed:', err);
  process.exitCode = 1;
});
