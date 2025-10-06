// background.js

// 1. Initialisierung bei der Installation
chrome.runtime.onInstalled.addListener(() => {
  // Standardeinstellungen speichern
  chrome.storage.local.set({
    isGloballyEnabled: true,
    model: 'gemini-1.5-flash', // Update model name
    fontSize: 1, // 0: Klein, 1: Mittel, 2: Groß
    contrastTheme: 'yellow-on-black',
    customPrompt: '',
    apiKey: '',
    whitelistedSites: [],
    whitelistedImages: []
  });

  // Kontextmenü für Bilder erstellen
  chrome.contextMenus.create({
    id: "whitelistImage",
    title: "Dieses Bild immer anzeigen",
    contexts: ["image"]
  });

  console.log("Erweiterung installiert und Standardeinstellungen gesetzt.");
});

// 2. Listener für das Kontextmenü
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "whitelistImage") {
    const imageUrl = info.srcUrl;
    if (imageUrl) {
      chrome.storage.local.get(['whitelistedImages'], (result) => {
        const updatedWhitelist = result.whitelistedImages || [];
        if (!updatedWhitelist.includes(imageUrl)) {
          updatedWhitelist.push(imageUrl);
          chrome.storage.local.set({ whitelistedImages: updatedWhitelist }, () => {
            console.log("Bild zur Whitelist hinzugefügt:", imageUrl);
          });
        }
      });
    }
  }
});

// 3. Listener für manuelle Transformation und Seiten-Updates
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "transformPage") {
        triggerPageTransformation(request.tabId, request.url);
        return true; // Indicates an async response
    }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Sicherstellen, dass die Seite vollständig geladen ist und eine gültige URL hat
  if (changeInfo.status === 'complete' && tab.url && tab.url.startsWith('http')) {
    triggerPageTransformation(tabId, tab.url);
  }
});

async function triggerPageTransformation(tabId, url) {
  const settings = await chrome.storage.local.get([
    'isGloballyEnabled',
    'apiKey',
    'whitelistedSites'
  ]);

  if (!settings.isGloballyEnabled) {
    return; // Global deaktiviert
  }

  if (!settings.apiKey) {
    console.log("API-Schlüssel nicht gefunden.");
    return;
  }

  const isWhitelisted = settings.whitelistedSites.some(site => url.includes(site));
  if (isWhitelisted) {
    console.log("Seite ist auf der Whitelist.");
    return;
  }

  console.log("Starte Transformation für:", url);
  chrome.tabs.sendMessage(tabId, { action: "getHTML" }, (response) => {
    if (chrome.runtime.lastError) {
        console.error("Fehler beim Senden der Nachricht an content.js:", chrome.runtime.lastError.message);
        return;
    }
    if (response && response.html) {
      callGeminiAPI(response.html, tabId);
    } else {
      console.log("Kein HTML vom Content-Skript erhalten.");
    }
  });
}

// 4. Funktion zum Aufrufen der Gemini API
async function callGeminiAPI(html, tabId) {
  const settings = await chrome.storage.local.get([
    'apiKey', 'model', 'fontSize', 'contrastTheme', 'customPrompt', 'whitelistedImages'
  ]);

  const FONT_SIZES = ['16px', '20px', '24px'];
  const THEMES = {
    'yellow-on-black': { bg: '#000000', text: '#FFFF00', link: '#FFFF99' },
    'white-on-black': { bg: '#000000', text: '#FFFFFF', link: '#AAAAFF' },
    'black-on-white': { bg: '#FFFFFF', text: '#000000', link: '#0000FF' },
  };

  const selectedFontSize = FONT_SIZES[settings.fontSize];
  const selectedTheme = THEMES[settings.contrastTheme];

  const systemPrompt = `
    Du bist ein Experte für Web-Barrierefreiheit. Deine Aufgabe ist es, den folgenden HTML-Code so umzuwandeln, dass er extrem einfach zu lesen und zu navigieren ist.
    Die Zielgruppe sind ältere Menschen und Menschen mit Seheinschränkungen.

    Regeln für die Umwandlung:
    1.  **Struktur:** Behalte die semantische Grundstruktur der Seite bei (Überschriften, Absätze, Listen, Links). Entferne alles, was nicht zum Kerninhalt gehört (unnötige Divs, Sidebars, Werbebanner, komplexe Menüs). Vereinfache die Navigation auf das Nötigste.
    2.  **Styling:** Entferne ALLE vorhandenen Stylesheets und Inline-Styles. Erstelle ein EINZIGES `<style>`-Tag im `<head>`.
    3.  **CSS im Style-Tag:**
        *   `body`: Hintergrundfarbe: ${selectedTheme.bg}, Textfarbe: ${selectedTheme.text}, Schriftgröße: ${selectedFontSize}, Schriftart: 'Arial', sans-serif.
        *   `a` (Links): Farbe: ${selectedTheme.link}, unterstrichen.
        *   `h1, h2, h3`: Klare Hierarchie durch größere Schriftgrößen.
        *   Füge grundlegendes Padding und Margin für gute Lesbarkeit hinzu.
    4.  **Bilder:** Entferne alle \`<img>\`-Tags, außer denen, deren \`src\`-Attribut in der folgenden Whitelist enthalten ist: [${settings.whitelistedImages.join(', ')}]. Behalte auch Bilder bei, die für das Verständnis absolut notwendig sind (z.B. Produktbilder, Diagramme). Alle anderen Bilder (dekorative, Icons, etc.) müssen entfernt werden.
    5.  **Skripte:** Entferne alle \`<script>\`-Tags.
    6.  **Zusätzliche Anweisung:** Falls eine zusätzliche Anweisung folgt, befolge sie: "${settings.customPrompt}".
    7.  **Ausgabe:** Gib NUR den vollständigen, neuen HTML-Code zurück. Kein einleitender Text, keine Erklärungen, nur das HTML, beginnend mit \`<!DOCTYPE html>\`.

    Hier ist der zu konvertierende HTML-Code:
  `;

  const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${settings.model}:generateContent?key=${settings.apiKey}`;

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ parts: [{ text: systemPrompt + '\n\n' + html }] }],
        generationConfig: {
            response_mime_type: "text/plain",
        }
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API-Fehler: ${response.status} ${response.statusText} - ${errorData.error.message}`);
    }

    const data = await response.json();
    const newHtml = data.candidates[0].content.parts[0].text;

    // Neues HTML an das Content-Skript senden
    chrome.tabs.sendMessage(tabId, { action: "replaceHTML", html: newHtml });

  } catch (error) {
    console.error("Fehler beim Aufruf der Gemini API:", error);
    // Optional: Benutzer benachrichtigen, dass ein Fehler aufgetreten ist.
  }
}