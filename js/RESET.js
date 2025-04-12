const typingArea = document.getElementById('typingArea');
const outputArea = document.getElementById('output');
const testCompleteIndicator = document.getElementById('testComplete');
const resetTestBtn = document.querySelector('.reset-test');
const testSentenceElement = document.getElementById('testSentence');

// Shared test configuration
let currentTestSentence = '';
let keystrokeLog = [];
let currentlyPressed = {};
let recordingStartTime = null;
let testActive = true;

// Initialize test
function initializeTest() {
    currentTestSentence = generateRandomSentence(3,1,1);
    testSentenceElement.textContent = `"${currentTestSentence}"`;
    resetTestState();
}

// Generate random sentence for both tests
/**
 * Generates random sentences for typing tests
 * @param {number} [complexity=1] - 1: Simple, 2: Medium, 3: Complex
 * @param {number} [minLength=5] - Minimum words in sentence
 * @param {number} [maxLength=15] - Maximum words in sentence
 * @returns {string} Randomly generated sentence
 */
function generateRandomSentence(complexity = 1, minLength = 5, maxLength = 15) {
    // Word banks by complexity
    const wordBanks = {
      simple: [
        'the', 'a', 'I', 'you', 'we', 'they', 'he', 'she', 'it',
        'is', 'are', 'was', 'were', 'have', 'has', 'had',
        'dog', 'cat', 'house', 'car', 'book', 'tree', 'sun',
        'runs', 'jumps', 'sleeps', 'eats', 'reads', 'writes',
        'quickly', 'slowly', 'happily', 'now', 'today', 'here'
      ],
      medium: [
        'although', 'because', 'before', 'while', 'after',
        'however', 'therefore', 'meanwhile', 'suddenly',
        'important', 'different', 'beautiful', 'interesting',
        'understand', 'recognize', 'appreciate', 'communicate',
        'morning', 'afternoon', 'evening', 'weekend', 'holiday',
        'frequently', 'occasionally', 'rarely', 'usually'
      ],
      complex: [
        'consequently', 'furthermore', 'nevertheless', 'nonetheless',
        'extraordinary', 'unbelievable', 'phenomenal', 'remarkable',
        'investigation', 'demonstration', 'application', 'evaluation',
        'metropolis', 'architecture', 'civilization', 'environment',
        'simultaneously', 'spontaneously', 'unexpectedly'
      ],
      nouns: [
        'doctor', 'mountain', 'ocean', 'computer', 'president',
        'discovery', 'education', 'knowledge', 'solution',
        'population', 'information', 'technology'
      ],
      verbs: [
        'constructed', 'developed', 'analyzed', 'discovered',
        'explained', 'predicted', 'transformed', 'implemented'
      ]
    };
  
    // Ensure we have enough words for each category
    const getWords = (category, count) => {
      const available = wordBanks[category] || [];
      return selectRandom(available, Math.min(count, available.length));
    };
  
    // Select words based on complexity
    const selectedWords = {
      simple: getWords('simple', 8),
      medium: complexity >= 2 ? getWords('medium', 8) : [],
      complex: complexity >= 3 ? getWords('complex', 8) : [],
      nouns: getWords('nouns', 5),
      verbs: getWords('verbs', 5)
    };
  
    // Simple sentence structures that are more robust
    const structures = [
      // Structure 1: Article + Noun + Verb
      (words) => {
        const article = selectFirst(words.simple.filter(w => ['the', 'a'].includes(w)), 'the');
        const noun = selectFirst(words.nouns, 'dog');
        const verb = selectFirst(words.verbs, 'runs');
        return `${capitalize(article)} ${noun} ${verb}.`;
      },
      
      // Structure 2: Pronoun + Verb + Article + Noun
      (words) => {
        const pronoun = selectFirst(words.simple.filter(w => ['I', 'you', 'he', 'she', 'it', 'we', 'they'].includes(w)), 'they');
        const verb = selectFirst(words.verbs, 'see');
        const article = selectFirst(words.simple.filter(w => ['the', 'a'].includes(w)), 'the');
        const noun = selectFirst(words.nouns, 'cat');
        return `${capitalize(pronoun)} ${verb} ${article} ${noun}.`;
      },
      
      // Structure 3: Article + Adjective + Noun + Verb + Adverb
      (words) => {
        const article = selectFirst(words.simple.filter(w => ['the', 'a'].includes(w)), 'the');
        const adjective = selectFirst(words.medium.filter(w => w.length > 6), 'beautiful');
        const noun = selectFirst(words.nouns, 'house');
        const verb = selectFirst(words.verbs, 'stands');
        const adverb = selectFirst(words.simple.filter(w => w.endsWith('ly')), 'quietly');
        return `${capitalize(article)} ${adjective} ${noun} ${verb} ${adverb}.`;
      }
    ];
  
    // Generate sentence parts
    const sentenceLength = randomInt(minLength, maxLength);
    let sentenceParts = [];
    
    for (let i = 0; i < sentenceLength; i++) {
      const structure = structures[randomInt(0, Math.min(structures.length-1, complexity))];
      try {
        const part = structure(selectedWords);
        if (part) sentenceParts.push(part);
      } catch (e) {
        console.warn('Error generating sentence part:', e);
      }
    }
  
    // Combine with proper punctuation
    let fullSentence = sentenceParts.join(' ');
    fullSentence = fullSentence.replace(/([^.])$/, '$1.'); // Ensure ends with period
    return fullSentence;
  
    // Helper functions with safety checks
    function capitalize(str) {
      return str && str.charAt ? str.charAt(0).toUpperCase() + str.slice(1) : 'The';
    }
  
    function selectRandom(array, count) {
      if (!array || array.length === 0) return [];
      const shuffled = [...array].sort(() => 0.5 - Math.random());
      return shuffled.slice(0, Math.min(count, shuffled.length));
    }
  
    function selectFirst(array, defaultValue) {
      if (!array || array.length === 0) return defaultValue;
      return array[0];
    }
  
    function randomInt(min, max) {
      return Math.floor(Math.random() * (max - min + 1)) + min;
    }
  }

function resetTestState() {
    testActive = true;
    keystrokeLog = [];
    currentlyPressed = {};
    recordingStartTime = null;
    typingArea.value = "";
    typingArea.readOnly = false;
    testCompleteIndicator.style.display = 'none';
    outputArea.textContent = "";
    typingArea.focus();
}

function handleKeyDown(event) {
    if (!testActive) {
        event.preventDefault();
        return;
    }

    if (recordingStartTime === null) {
        recordingStartTime = performance.now();
        outputArea.textContent = "";
    }

    if (event.key === 'Enter') {
        event.preventDefault();
        if (keystrokeLog.length > 0) endTest();
        return;
    }

    const key = event.key;
    if (!currentlyPressed.hasOwnProperty(key)) {
        currentlyPressed[key] = performance.now() - recordingStartTime;
    }
}

function handleKeyUp(event) {
    if (!testActive || recordingStartTime === null) return;

    const key = event.key;
    if (currentlyPressed.hasOwnProperty(key)) {
        const pressTime = currentlyPressed[key];
        const releaseTime = performance.now() - recordingStartTime;
        
        if (releaseTime > pressTime) {
            keystrokeLog.push({
                key: key,
                press_time: pressTime,
                release_time: releaseTime
            });
        }
        delete currentlyPressed[key];
    }
}

function endTest() {
    testActive = false;
    typingArea.blur();
    typingArea.readOnly = true;
    testCompleteIndicator.style.display = 'flex';
    displayResults();
}

function displayResults() {
    keystrokeLog.sort((a, b) => a.press_time - b.press_time);
    
    const csvData = keystrokeLog.map(entry => 
        `"${entry.key.replace(/"/g, '""')}",${entry.press_time.toFixed(3)},${entry.release_time.toFixed(3)}`
    ).join('\n');
    
    outputArea.textContent = csvData;
}

// Event listeners
typingArea.addEventListener('keydown', handleKeyDown);
typingArea.addEventListener('keyup', handleKeyUp);
resetTestBtn.addEventListener('click', () => {
    initializeTest();
    // Add voice test reset here if needed
});

// Initialize first test
initializeTest();