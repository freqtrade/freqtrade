from playwright.sync_api import sync_playwright
import os

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the local file
        file_path = f"file://{os.path.abspath('freqtrade/rpc/api_server/ui/fallback_file.html')}"
        print(f"Loading {file_path}")
        page.goto(file_path)

        # Screenshot 1: Default
        page.screenshot(path="fallback_ui_default.png")
        print("Captured default state")

        # Toggle Theme
        toggle_btn = page.locator(".theme-toggle")
        if toggle_btn.is_visible():
            toggle_btn.click()
            page.wait_for_timeout(500) # Wait for transition
            page.screenshot(path="fallback_ui_toggled.png")
            print("Captured toggled state")
        else:
            print("Theme toggle not found!")

        # Refresh Button
        refresh_btn = page.locator(".refresh-btn")
        if refresh_btn.is_visible():
            refresh_btn.click()
            # The click triggers a reload after 500ms.
            # We want to capture the "Checking..." state before reload if possible.
            page.wait_for_timeout(100)
            page.screenshot(path="fallback_ui_loading.png")
            print("Captured loading state")
        else:
            print("Refresh button not found!")

        browser.close()

if __name__ == "__main__":
    run()
