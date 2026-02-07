const fs = require('fs');
const os = require('os');
const path = require('path');

let envLoaded = false;

function loadDotEnv() {
  if (envLoaded) return;
  envLoaded = true;

  const envPath = path.resolve('.env');
  if (!fs.existsSync(envPath)) return;

  const lines = fs.readFileSync(envPath, 'utf8').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIndex = trimmed.indexOf('=');
    if (eqIndex === -1) continue;
    const key = trimmed.slice(0, eqIndex).trim();
    let value = trimmed.slice(eqIndex + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}

function getProfileDir() {
  loadDotEnv();
  const envDir = process.env.X_PROFILE_DIR;
  if (envDir && envDir.trim()) {
    const trimmed = envDir.trim();
    if (trimmed.startsWith('~/')) {
      return path.join(os.homedir(), trimmed.slice(2));
    }
    return path.resolve(trimmed);
  }
  return path.join(os.homedir(), '.playwright-profiles', 'x-profile');
}

module.exports = {
  getProfileDir,
  loadDotEnv
};
