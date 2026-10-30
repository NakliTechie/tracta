"""Render marketing/social-card.html to marketing/social.png (1280x640) and copy it to site/social.png.

    python3 marketing/render-card.py
"""
import os
import pathlib
import shutil

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).resolve().parent
out = HERE / "social.png"
with sync_playwright() as p:
    # Any recent Chromium works; set TRACTA_CHROMIUM to use one other than Playwright's default build.
    exe = os.environ.get("TRACTA_CHROMIUM")
    browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 640}, color_scheme="light")
    page.goto((HERE / "social-card.html").as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1280, "height": 640})
    browser.close()
shutil.copyfile(out, HERE.parent / "site" / "social.png")
print(f"wrote {out} and site/social.png")
