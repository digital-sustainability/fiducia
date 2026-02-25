// i18n.js (or i18n.ts)
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

// This map will store promises for loading resources to prevent multiple fetches
const loadedResources = {};

const loadTranslationFile = async (lang) => {
    if (loadedResources[lang]) {
        return loadedResources[lang]; // Return existing promise if already loading/loaded
    }

    const fetchPromise = fetch(`/locales/${lang}/translation.json`) // Adjust path if needed
        .then(response => {
            if (!response.ok) {
                throw new Error(`Failed to load translations for ${lang}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            i18n.addResourceBundle(lang, 'translation', data, true, true); // Add resources to i18n
            return data;
        })
        .catch(error => {
            console.error(`Error loading translation file for ${lang}:`, error);
            // Fallback to empty object or default translations on error
            i18n.addResourceBundle(lang, 'translation', {}, true, true);
            return {};
        });

    loadedResources[lang] = fetchPromise;
    return fetchPromise;
};

// Initialize i18n once when this file is first imported
let i18nInitialized = false;

export const initI18n = async (lang) => {
    if (!i18nInitialized) {
        await i18n
            .use(initReactI18next)
            .init({
                lng: lang, // Initial language, will be updated by components
                fallbackLng: 'en',
                debug: false, // Set to true for debugging translation issues
                resources: {}, // Start with empty resources, load dynamically
                interpolation: {
                    escapeValue: false, // React already escapes by default
                },
            });
        i18nInitialized = true;
    }
    // Ensure the target language's resources are loaded and set the language
    await loadTranslationFile(lang);
    await i18n.changeLanguage(lang);
};

export default i18n; // Export the i18n instance