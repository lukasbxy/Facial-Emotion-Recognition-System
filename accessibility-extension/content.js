// content.js

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // Aktion: HTML abrufen und an background.js senden
  if (request.action === "getHTML") {
    console.log("Anfrage zum Abrufen von HTML erhalten.");
    // Senden Sie das gesamte HTML des Dokuments zurück.
    sendResponse({ html: document.documentElement.outerHTML });
    // Wichtig: 'true' zurückgeben, um anzuzeigen, dass die Antwort asynchron gesendet wird.
    return true;
  }

  // Aktion: HTML der Seite durch die neue Version ersetzen
  if (request.action === "replaceHTML") {
    console.log("Anfrage zum Ersetzen von HTML erhalten.");
    try {
      // Stoppt das Laden weiterer Ressourcen der alten Seite.
      window.stop();

      // Ersetzt den gesamten Dokumenteninhalt.
      // document.open/write/close ist eine robuste Methode für diesen Zweck.
      document.open();
      document.write(request.html);
      document.close();
    } catch (error) {
      console.error("Fehler beim Ersetzen des Seiteninhalts:", error);
    }
  }
});