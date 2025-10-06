// content.js
console.log("Content: Content-Script geladen");

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("Content: Nachricht empfangen:", request);

  // Aktion: HTML abrufen und an background.js senden
  if (request.action === "getHTML") {
    console.log("Content: Anfrage zum Abrufen von HTML erhalten.");
    const html = document.documentElement.outerHTML;
    console.log("Content: HTML abgerufen, Länge:", html.length);
    // Senden Sie das gesamte HTML des Dokuments zurück.
    sendResponse({ html: html });
    // Wichtig: 'true' zurückgeben, um anzuzeigen, dass die Antwort asynchron gesendet wird.
    return true;
  }

  // Aktion: HTML der Seite durch die neue Version ersetzen
  if (request.action === "replaceHTML") {
    console.log("Content: Anfrage zum Ersetzen von HTML erhalten.");
    try {
      console.log("Content: Stoppe Laden der Seite");
      // Stoppt das Laden weiterer Ressourcen der alten Seite.
      window.stop();

      console.log("Content: Ersetze Dokumenteninhalt");
      // Ersetzt den gesamten Dokumenteninhalt.
      // document.open/write/close ist eine robuste Methode für diesen Zweck.
      document.open();
      document.write(request.html);
      document.close();
      console.log("Content: HTML erfolgreich ersetzt");
    } catch (error) {
      console.error("Content: Fehler beim Ersetzen des Seiteninhalts:", error);
    }
  }
});