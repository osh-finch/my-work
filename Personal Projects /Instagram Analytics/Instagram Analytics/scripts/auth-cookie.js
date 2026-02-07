const fs = require('fs');
const path = require('path');
const readline = require('readline');

function setEnvValue(lines, key, value) {
  let replaced = false;
  const nextLines = lines.map((line) => {
    if (line.trim().startsWith(`${key}=`)) {
      replaced = true;
      return `${key}=${value}`;
    }
    return line;
  });

  if (!replaced) {
    nextLines.push(`${key}=${value}`);
  }

  return nextLines;
}

async function prompt(question) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const answer = await new Promise((resolve) => {
    rl.question(question, (value) => resolve(value));
  });

  rl.close();
  return answer;
}

async function main() {
  const authToken = (await prompt('Paste auth_token cookie value: ')).trim();
  if (!authToken) {
    console.error('auth_token is required.');
    process.exitCode = 1;
    return;
  }

  const ct0 = (await prompt('Paste ct0 cookie value (optional, press enter to skip): ')).trim();

  const envPath = path.resolve('.env');
  const existing = fs.existsSync(envPath) ? fs.readFileSync(envPath, 'utf8') : '';
  const lines = existing ? existing.split(/\r?\n/) : [];
  let nextLines = setEnvValue(lines, 'X_AUTH_TOKEN', authToken);
  if (ct0) {
    nextLines = setEnvValue(nextLines, 'X_CT0', ct0);
  }

  const output = nextLines.filter((line, index, arr) => {
    return index < arr.length - 1 || line.trim().length > 0;
  }).join('\n');

  fs.writeFileSync(envPath, `${output}\n`, 'utf8');

  console.log('Saved to .env. You can now run: npm run test:session');
}

main().catch((err) => {
  console.error('Failed to save auth cookies:', err);
  process.exitCode = 1;
});
