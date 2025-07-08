// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded and ready!');
    
    // Get the CTA button
    const ctaButton = document.getElementById('cta-button');
    
    // Add click event listener
    ctaButton.addEventListener('click', function() {
        alert('Thanks for your interest! This is a demo website created with the AI Code Organizer.');
    });
    
    // Simple animation for the features
    const features = document.querySelectorAll('.feature');
    
    features.forEach(function(feature, index) {
        // Add a small delay for each feature
        setTimeout(function() {
            feature.style.opacity = '0';
            feature.style.transition = 'opacity 0.5s ease-in-out';
            
            setTimeout(function() {
                feature.style.opacity = '1';
            }, 100);
        }, index * 300);
    });
});