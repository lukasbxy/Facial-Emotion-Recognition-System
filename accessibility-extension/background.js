// background.js

// 1. Initialisierung bei der Installation
chrome.runtime.onInstalled.addListener(() => {
  // Standardeinstellungen speichern
  chrome.storage.local.set({
    isGloballyEnabled: true,
    model: 'gemini-2.5-flash', // Update model name
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
  console.log("Background: Nachricht empfangen:", request);
  if (request.action === "transformPage") {
    console.log("Background: transformPage Aktion erkannt");
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
  console.log("Background: triggerPageTransformation aufgerufen für Tab:", tabId, "URL:", url);
  const settings = await chrome.storage.local.get([
    'isGloballyEnabled',
    'apiKey',
    'whitelistedSites'
  ]);

  console.log("Background: Einstellungen geladen:", {
    isGloballyEnabled: settings.isGloballyEnabled,
    hasApiKey: !!settings.apiKey,
    apiKeyLength: settings.apiKey ? settings.apiKey.length : 0,
    whitelistedSites: settings.whitelistedSites
  });

  if (!settings.isGloballyEnabled) {
    console.log("Background: Extension global deaktiviert");
    return; // Global deaktiviert
  }

  if (!settings.apiKey || settings.apiKey.trim() === '') {
    console.log("Background: API-Schlüssel nicht gefunden oder leer");
    // Benachrichtigung anzeigen
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: 'API-Schlüssel fehlt',
      message: 'Bitte geben Sie Ihren Google AI API-Schlüssel in den Einstellungen ein.'
    });
    return;
  }

  const isWhitelisted = settings.whitelistedSites.some(site => url.includes(site));
  if (isWhitelisted) {
    console.log("Background: Seite ist auf der Whitelist");
    return;
  }

  console.log("Background: Starte Transformation für:", url);

  // Überprüfe zuerst, ob der Tab noch existiert und zugänglich ist
  chrome.tabs.get(tabId, async (tab) => {
    if (chrome.runtime.lastError) {
      console.error("Background: Tab nicht gefunden:", chrome.runtime.lastError.message);
      return;
    }

    // Stelle sicher, dass das Content-Script injiziert ist
    try {
      console.log("Background: Injiziere Content-Script falls nötig");
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
        files: ['content.js']
      });
      console.log("Background: Content-Script erfolgreich injiziert");
    } catch (error) {
      console.log("Background: Content-Script bereits vorhanden oder Fehler beim Injizieren:", error.message);
    }

    console.log("Background: Sende Nachricht an content.js");
    chrome.tabs.sendMessage(tabId, { action: "getHTML" }, (response) => {
      if (chrome.runtime.lastError) {
        console.error("Background: Fehler beim Senden der Nachricht an content.js:", chrome.runtime.lastError.message);
        console.error("Background: Runtime-Fehler Details:", chrome.runtime.lastError);

        // Versuche alternative Methode mit scripting API
        console.log("Background: Versuche alternative Methode...");
        chrome.scripting.executeScript({
          target: { tabId: tabId },
          function: () => {
            return document.documentElement.outerHTML;
          }
        }, (results) => {
          if (chrome.runtime.lastError) {
            console.error("Background: Scripting API Fehler:", chrome.runtime.lastError.message);
            return;
          }
          if (results && results[0] && results[0].result) {
            console.log("Background: HTML via Scripting API erhalten, Länge:", results[0].result.length);
            callGeminiAPI(results[0].result, tabId);
          }
        });

        return;
      }
      console.log("Background: Antwort von content.js erhalten:", response ? "HTML vorhanden" : "Keine Antwort");
      if (response && response.html) {
        console.log("Background: HTML erhalten, Länge:", response.html.length);
        callGeminiAPI(response.html, tabId);
      } else {
        console.log("Background: Kein HTML vom Content-Skript erhalten");
      }
    });
  });
}

// 4. Funktion zum Aufrufen der Gemini API
async function callGeminiAPI(html, tabId) {
  console.log("Background: callGeminiAPI aufgerufen für Tab:", tabId);
  const settings = await chrome.storage.local.get([
    'apiKey', 'model', 'fontSize', 'contrastTheme', 'customPrompt', 'whitelistedImages'
  ]);

  console.log("Background: API-Einstellungen:", {
    model: settings.model,
    fontSize: settings.fontSize,
    contrastTheme: settings.contrastTheme,
    hasCustomPrompt: !!settings.customPrompt,
    whitelistedImagesCount: settings.whitelistedImages ? settings.whitelistedImages.length : 0
  });

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
    2.  **Styling:** Entferne ALLE vorhandenen Stylesheets und Inline-Styles. Erstelle ein EINZIGES \`<style>\`-Tag im \`<head>\`.
    3.  **CSS im Style-Tag:**
        *   \`body\`: Hintergrundfarbe: ${selectedTheme.bg}, Textfarbe: ${selectedTheme.text}, Schriftgröße: ${selectedFontSize}, Schriftart: 'Arial', sans-serif.
        *   \`a\` (Links): Farbe: ${selectedTheme.link}, unterstrichen.
        *   \`h1, h2, h3\`: Klare Hierarchie durch größere Schriftgrößen.
        *   Füge grundlegendes Padding und Margin für gute Lesbarkeit hinzu.
    4.  **Bilder:** Entferne alle \`<img>\`-Tags, außer denen, deren \`src\`-Attribut in der folgenden Whitelist enthalten ist: [${settings.whitelistedImages.join(', ')}]. Behalte auch Bilder bei, die für das Verständnis absolut notwendig sind (z.B. Produktbilder, Diagramme). Alle anderen Bilder (dekorative, Icons, etc.) müssen entfernt werden.
    5.  **Skripte:** Entferne alle \`<script>\`-Tags.
    6.  **Zusätzliche Anweisung:** Falls eine zusätzliche Anweisung folgt, befolge sie: "${settings.customPrompt}".
    7.  **Ausgabe:** Gib NUR den vollständigen, neuen HTML-Code zurück. Kein einleitender Text, keine Erklärungen, nur das HTML, beginnend mit \`<!DOCTYPE html>\`.

    Hier ist der zu konvertierende HTML-Code:
  `;

  const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${settings.model}:generateContent?key=${settings.apiKey}`;
  console.log("Background: API-URL erstellt (ohne Key):", API_URL.replace(settings.apiKey, '[API_KEY]'));

  try {
    console.log("Background: Sende API-Anfrage...");
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

    console.log("Background: API-Antwort erhalten, Status:", response.status);

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Background: API-Fehler Details:", errorData);
      throw new Error(`API-Fehler: ${response.status} ${response.statusText} - ${errorData.error.message}`);
    }

    const data = await response.json();
    console.log("Background: API-Antwort erfolgreich geparst");
    const newHtml = data.candidates[0].content.parts[0].text;
    console.log("Background: Neues HTML erhalten, Länge:", newHtml.length);

    // Neues HTML direkt über Scripting API einfügen
    console.log("Background: Ersetze HTML über Scripting API");
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tabId },
        func: (html) => {
          window.stop();
          document.open();
          document.write(html);
          document.close();
        },
        args: [newHtml]
      });
      console.log("Background: HTML erfolgreich ersetzt");
      // Erfolgsbenachrichtigung anzeigen
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon128.png',
        title: 'Barrierefreiheit aktiviert',
        message: 'Die Webseite wurde für bessere Lesbarkeit umgewandelt.'
      });
    } catch (scriptError) {
      console.error("Background: Fehler beim Ersetzen des HTML:", scriptError);
    }

  } catch (error) {
    console.error("Background: Fehler beim Aufruf der Gemini API:", error);
    // Benutzer benachrichtigen, dass ein Fehler aufgetreten ist
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon128.png',
      title: 'API-Fehler',
      message: 'Fehler bei der Verarbeitung der Webseite. Überprüfen Sie Ihren API-Schlüssel und die Internetverbindung.'
    });
  }
}