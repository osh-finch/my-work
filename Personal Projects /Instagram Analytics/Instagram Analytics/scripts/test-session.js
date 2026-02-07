const { launchContext, isLoggedIn } = require('./x-session');

async function main() {
  const context = await launchContext();
  const page = context.pages()[0] || await context.newPage();

  await page.goto('https://x.com/home', { waitUntil: 'domcontentloaded' });
  const loggedIn = await isLoggedIn(page);

  if (loggedIn) {
    console.log('Logged in');
  } else {
    console.log('Not logged in');
  }

  await context.close();

  if (!loggedIn) {
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error('Session test failed:', err);
  process.exitCode = 1;
});
