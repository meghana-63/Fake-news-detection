/**
 * TruthLens AI - Client-Side Controller
 * Handles real-time text stats, sample injection, progressive loading feedback,
 * and seamless smooth transitions.
 */

document.addEventListener('DOMContentLoaded', () => {
    const newsInput = document.getElementById('newsInput');
    const wordCountSpan = document.getElementById('wordCount');
    const charCountSpan = document.getElementById('charCount');
    const clearBtn = document.getElementById('clearBtn');
    const newsForm = document.getElementById('newsForm');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const samplePills = document.querySelectorAll('.sample-pill');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingStatusText = document.getElementById('loadingStatusText');
    const loadingStepDetail = document.getElementById('loadingStepDetail');
    const loadingProgressBar = document.getElementById('loadingProgressBar');
    const resultsSection = document.getElementById('resultsSection');

    // --------------------------------------------------
    // 1. Text Metrics Counter
    // --------------------------------------------------
    function updateTextMetrics() {
        if (!newsInput) return;
        const text = newsInput.value || '';
        const charCount = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;

        if (charCountSpan) {
            charCountSpan.textContent = `${charCount.toLocaleString()} ${charCount === 1 ? 'character' : 'characters'}`;
        }
        if (wordCountSpan) {
            wordCountSpan.textContent = `${words.toLocaleString()} ${words === 1 ? 'word' : 'words'}`;
        }
    }

    if (newsInput) {
        newsInput.addEventListener('input', updateTextMetrics);
        updateTextMetrics();
    }

    // --------------------------------------------------
    // 2. Clear Button Handler
    // --------------------------------------------------
    if (clearBtn && newsInput) {
        clearBtn.addEventListener('click', () => {
            newsInput.value = '';
            newsInput.focus();
            updateTextMetrics();
        });
    }

    // --------------------------------------------------
    // 3. Quick Sample Pill Ingestion
    // --------------------------------------------------
    samplePills.forEach(pill => {
        pill.addEventListener('click', () => {
            const sampleText = pill.getAttribute('data-text');
            if (sampleText && newsInput) {
                newsInput.value = sampleText;
                newsInput.focus();
                updateTextMetrics();

                // Subtle animation highlight
                newsInput.parentElement.style.borderColor = 'var(--accent-cyan)';
                setTimeout(() => {
                    newsInput.parentElement.style.borderColor = '';
                }, 700);
            }
        });
    });

    // --------------------------------------------------
    // 4. Progressive Multi-Step Loading State
    // --------------------------------------------------
    const loadingStages = [
        { progress: 20, title: "Tokenizing Input...", detail: "Normalizing text and stripping formatting markup" },
        { progress: 45, title: "Evaluating ML Patterns...", detail: "Transforming 50k TF-IDF n-grams & calculating probabilities" },
        { progress: 75, title: "Searching TheNewsAPI...", detail: "Querying live global news index for contemporary reporting" },
        { progress: 95, title: "Synthesizing Evidence...", detail: "Cross-referencing claims, computing similarity & generating dossier" }
    ];

    if (newsForm && loadingOverlay) {
        newsForm.addEventListener('submit', (e) => {
            const text = newsInput ? newsInput.value.trim() : '';
            if (!text) {
                e.preventDefault();
                alert('Please enter a news headline or article to analyze.');
                return;
            }

            // Reveal overlay
            loadingOverlay.style.display = 'flex';
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.style.opacity = '0.6';
            }

            let stageIdx = 0;
            const stepInterval = setInterval(() => {
                if (stageIdx < loadingStages.length) {
                    const stage = loadingStages[stageIdx];
                    if (loadingStatusText) loadingStatusText.textContent = stage.title;
                    if (loadingStepDetail) loadingStepDetail.textContent = stage.detail;
                    if (loadingProgressBar) loadingProgressBar.style.width = `${stage.progress}%`;
                    stageIdx++;
                } else {
                    clearInterval(stepInterval);
                }
            }, 650);
        });
    }

    // --------------------------------------------------
    // 5. Smooth Scroll to Results on Render
    // --------------------------------------------------
    if (resultsSection) {
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 150);
    }
});
