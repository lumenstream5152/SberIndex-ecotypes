// Сборка деки: deck.html → presentation.pdf (playwright-core + системный
// chrome-headless-shell; браузер не скачиваем — спека 35).
// Запуск: node build.mjs
import { chromium } from 'playwright-core';
import { existsSync, readdirSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

const SHELL = join(homedir(),
  'Library/Caches/ms-playwright/chromium_headless_shell-1234/' +
  'chrome-headless-shell-mac-arm64/chrome-headless-shell');
if (!existsSync(SHELL)) {
  // fallback: любая версия headless-shell в кэше
  const root = join(homedir(), 'Library/Caches/ms-playwright');
  const alt = readdirSync(root).find(d => d.startsWith('chromium_headless_shell'));
  if (!alt) throw new Error('chrome-headless-shell не найден в ms-playwright');
}
const exe = existsSync(SHELL) ? SHELL : (() => {
  const root = join(homedir(), 'Library/Caches/ms-playwright');
  const dir = readdirSync(root).find(d => d.startsWith('chromium_headless_shell'));
  return join(root, dir, 'chrome-headless-shell-mac-arm64', 'chrome-headless-shell');
})();

const browser = await chromium.launch({ executablePath: exe });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const url = 'file://' + new URL('./deck.html', import.meta.url).pathname;
await page.goto(url, { waitUntil: 'networkidle' });
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(600);
const n = await page.locator('.slide').count();
await page.pdf({
  path: 'presentation.pdf',
  width: '1280px', height: '720px',
  printBackground: true,
  pageRanges: Array.from({ length: n }, (_, i) => String(i + 1)).join(','),
});
console.log(`presentation.pdf: ${n} слайдов`);
await browser.close();
