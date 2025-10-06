// options.js

document.addEventListener('DOMContentLoaded', () => {
    const apiKeyInput = document.getElementById('api-key-input');
    const saveApiKeyButton = document.getElementById('save-api-key-button');
    const saveStatus = document.getElementById('save-status');
    const siteWhitelistList = document.getElementById('whitelist-list');
    const imageWhitelistList = document.getElementById('image-whitelist-list');

    // 1. Gespeicherte Werte laden und anzeigen
    const loadSettings = () => {
        chrome.storage.local.get(['apiKey', 'whitelistedSites', 'whitelistedImages'], (result) => {
            if (result.apiKey) {
                apiKeyInput.value = result.apiKey;
            }
            renderSiteWhitelist(result.whitelistedSites || []);
            renderImageWhitelist(result.whitelistedImages || []);
        });
    };

    // 2. API-Schlüssel speichern
    saveApiKeyButton.addEventListener('click', () => {
        const apiKey = apiKeyInput.value.trim();
        chrome.storage.local.set({ apiKey: apiKey }, () => {
            saveStatus.textContent = 'API-Schlüssel gespeichert!';
            setTimeout(() => { saveStatus.textContent = ''; }, 3000);
        });
    });

    // 3. Site Whitelist rendern
    const renderSiteWhitelist = (sites) => {
        siteWhitelistList.innerHTML = '';
        if (sites.length > 0) {
            sites.forEach((site, index) => {
                const listItem = createListItem(site, index, removeSiteFromWhitelist);
                siteWhitelistList.appendChild(listItem);
            });
        } else {
            siteWhitelistList.innerHTML = '<li>Keine Webseiten auf der Whitelist.</li>';
        }
    };

    // 4. Image Whitelist rendern
    const renderImageWhitelist = (images) => {
        imageWhitelistList.innerHTML = '';
        if (images.length > 0) {
            images.forEach((imageUrl, index) => {
                const listItem = createListItem(imageUrl, index, removeImageFromWhitelist, true);
                imageWhitelistList.appendChild(listItem);
            });
        } else {
            imageWhitelistList.innerHTML = '<li>Keine Bilder auf der Whitelist.</li>';
        }
    };

    // Hilfsfunktion zum Erstellen von Listenelementen
    const createListItem = (content, index, removeCallback, isImage = false) => {
        const listItem = document.createElement('li');
        listItem.style.display = 'flex';
        listItem.style.justifyContent = 'space-between';
        listItem.style.alignItems = 'center';
        listItem.style.marginBottom = '10px';

        const contentElement = isImage ? document.createElement('img') : document.createElement('span');
        if (isImage) {
            contentElement.src = content;
            contentElement.style.maxWidth = '200px';
            contentElement.style.maxHeight = '50px';
            contentElement.style.marginRight = '10px';
        } else {
            contentElement.textContent = content;
        }

        const removeButton = document.createElement('button');
        removeButton.textContent = 'Entfernen';
        removeButton.style.width = 'auto';
        removeButton.addEventListener('click', () => removeCallback(index));

        listItem.appendChild(contentElement);
        listItem.appendChild(removeButton);
        return listItem;
    };

    // 5. Seite von der Whitelist entfernen
    const removeSiteFromWhitelist = (indexToRemove) => {
        chrome.storage.local.get(['whitelistedSites'], (result) => {
            const updatedList = result.whitelistedSites || [];
            updatedList.splice(indexToRemove, 1);
            chrome.storage.local.set({ whitelistedSites: updatedList }, () => {
                renderSiteWhitelist(updatedList);
            });
        });
    };

    // 6. Bild von der Whitelist entfernen
    const removeImageFromWhitelist = (indexToRemove) => {
        chrome.storage.local.get(['whitelistedImages'], (result) => {
            const updatedList = result.whitelistedImages || [];
            updatedList.splice(indexToRemove, 1);
            chrome.storage.local.set({ whitelistedImages: updatedList }, () => {
                renderImageWhitelist(updatedList);
            });
        });
    };

    // Initiales Laden der Einstellungen
    loadSettings();
});