// Counter animation for results section
document.addEventListener('DOMContentLoaded', function() {
    const resultValues = document.querySelectorAll('.result-value');
    
    function animateCounters() {
        resultValues.forEach(value => {
            const target = parseInt(value.getAttribute('data-target'));
            const duration = 2000; // Animation duration in ms
            const start = 0;
            const increment = target / (duration / 16); // 60fps
            
            let current = start;
            const counter = setInterval(() => {
                current += increment;
                if (current >= target) {
                    clearInterval(counter);
                    current = target;
                }
                value.textContent = Math.floor(current);
            }, 16);
        });
    }
    
    // Only animate when results section is in view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounters();
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    const resultsSection = document.querySelector('.results-section');
    if (resultsSection) {
        observer.observe(resultsSection);
    }
});