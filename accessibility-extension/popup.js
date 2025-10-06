// popup.js

document.addEventListener('DOMContentLoaded', () => {
    // UI-Elemente abrufen
    const globalToggle = document.getElementById('global-toggle');
    const modelSelect = document.getElementById('model-select');
    const fontSizeSlider = document.getElementById('font-size-slider');
    const contrastSelect = document.getElementById('contrast-select');
    const customPrompt = document.getElementById('custom-prompt');
    const reloadButton = document.getElementById('reload-button');
    const whitelistButton = document.getElementById('whitelist-button');

    // 1. Gespeicherte Einstellungen laden und UI aktualisieren
    const loadSettings = () => {
        chrome.storage.local.get([
            'isGloballyEnabled', 'model', 'fontSize', 'contrastTheme', 'customPrompt'
        ], (result) => {
            globalToggle.checked = result.isGloballyEnabled !== false; // Default to true
            if (result.model) modelSelect.value = result.model;
            if (result.fontSize) fontSizeSlider.value = result.fontSize;
            if (result.contrastTheme) contrastSelect.value = result.contrastTheme;
            if (result.customPrompt) customPrompt.value = result.customPrompt;
        });
    };

    // 2. Event-Listener für alle Einstellungen hinzufügen
    globalToggle.addEventListener('change', (e) => {
        chrome.storage.local.set({ isGloballyEnabled: e.target.checked });
    });

    modelSelect.addEventListener('change', (e) => {
        chrome.storage.local.set({ model: e.target.value });
    });

    fontSizeSlider.addEventListener('input', (e) => {
        chrome.storage.local.set({ fontSize: parseInt(e.target.value, 10) });
    });

    contrastSelect.addEventListener('change', (e) => {
        chrome.storage.local.set({ contrastTheme: e.target.value });
    });

    customPrompt.addEventListener('input', (e) => {
        chrome.storage.local.set({ customPrompt: e.target.value });
    });

    // 3. Funktionalität für die Buttons

    // Ansicht neu generieren
    reloadButton.addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const currentTab = tabs[0];
            if (currentTab && currentTab.id && currentTab.url) {
                // Send a message to the background script to trigger the transformation directly.
                chrome.runtime.sendMessage({
                    action: "transformPage",
                    tabId: currentTab.id,
                    url: currentTab.url
                }, () => {
                    window.close(); // Close the popup after sending the message
                });
            }
        });
    });

    // Aktuelle Seite zur Whitelist hinzufügen
    whitelistButton.addEventListener('click', () => {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const currentTab = tabs[0];
            if (currentTab && currentTab.url) {
                chrome.storage.local.get(['whitelistedSites'], (result) => {
                    const updatedWhitelist = result.whitelistedSites || [];
                    const siteUrl = new URL(currentTab.url).hostname; // Nur den Hostnamen speichern
                    if (!updatedWhitelist.includes(siteUrl)) {
                        updatedWhitelist.push(siteUrl);
                        chrome.storage.local.set({ whitelistedSites: updatedWhitelist }, () => {
                            // Seite neu laden, um die originale Version wiederherzustellen
                            chrome.tabs.reload(currentTab.id);
                            window.close();
                        });
                    }
                });
            }
        });
    });

    // Initiales Laden der Einstellungen
    loadSettings();
});