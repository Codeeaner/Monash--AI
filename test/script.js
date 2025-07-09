document.addEventListener('DOMContentLoaded', function() {
    fetch('analysis_6f33e0ac_1751648345.json')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('json-container');
            container.innerHTML = ''; // Clear previous content

            // Set timestamp
            const timestampEl = document.getElementById('timestamp');
            if (timestampEl) {
                timestampEl.textContent = new Date(data.timestamp).toLocaleString();
            }

            // Icon and class mapping
            const sectionStyles = {
                'AI Analysis': { icon: 'fa-solid fa-brain', class: 'card-analysis' },
                'Processing Recommendations': { icon: 'fa-solid fa-cogs', class: 'card-processing' },
                'Waste Prevention': { icon: 'fa-solid fa-recycle', class: 'card-waste' },
                'Economic Impact': { icon: 'fa-solid fa-dollar-sign', class: 'card-economic' },
                'Recommendations': { icon: 'fa-solid fa-clipboard-check', class: 'card-recommendations' }
            };

            // Function to format values
            function formatValue(value) {
                if (typeof value !== 'string') return value;
                let formatted = value.replace(/_/g, ' ');
                return formatted.charAt(0).toUpperCase() + formatted.slice(1);
            }

            // Function to create a card
            function createCard(title, content) {
                const style = sectionStyles[title] || { icon: 'fa-solid fa-question-circle', class: '' };
                const card = document.createElement('div');
                card.className = `card ${style.class}`;

                const cardTitle = document.createElement('h2');
                cardTitle.className = 'card-title';
                cardTitle.innerHTML = `<i class="${style.icon}"></i> ${title}`;
                card.appendChild(cardTitle);

                const cardContent = document.createElement('div');
                cardContent.className = 'card-content';
                cardContent.innerHTML = content;
                card.appendChild(cardContent);

                // Add copy to clipboard button
                const copyButton = document.createElement('button');
                copyButton.className = 'copy-btn';
                copyButton.innerHTML = '<i class="fas fa-copy"></i> Copy';
                copyButton.onclick = () => {
                    const textToCopy = cardContent.innerText;
                    navigator.clipboard.writeText(textToCopy).then(() => {
                        Toastify({
                            text: `Copied "${title}" to clipboard!`,
                            duration: 3000,
                            close: true,
                            gravity: "top",
                            position: "right",
                            backgroundColor: "linear-gradient(to right, #00b09b, #96c93d)",
                        }).showToast();
                    });
                };
                card.appendChild(copyButton);

                return card;
            }

            // Function to create a list
            function createList(items) {
                let listHtml = '<ul class="list">';
                items.forEach(item => {
                    listHtml += `<li class="list-item">${formatValue(item)}</li>`;
                });
                listHtml += '</ul>';
                return listHtml;
            }

            // Overall Assessment
            let overallContent = `<p><strong>Overall Assessment:</strong> ${data.ai_analysis.analysis.overall_assessment}</p>` +
                                 `<p><strong>Quality Grade:</strong> ${formatValue(data.ai_analysis.analysis.quality_grade)}</p>` +
                                 `<p><strong>Severity Level:</strong> ${formatValue(data.ai_analysis.analysis.severity_level)}</p>` +
                                 `<p><strong>Defect Types:</strong> ${data.ai_analysis.analysis.defect_types.join(', ')}</p>`;
            container.appendChild(createCard('AI Analysis', overallContent));

            // Processing Recommendations
            let processingContent = `<h3>Immediate Actions:</h3>${createList(data.ai_analysis.processing_recommendations.immediate_actions)}` +
                                    `<p><strong>Sorting Strategy:</strong> ${data.ai_analysis.processing_recommendations.sorting_strategy}</p>` +
                                    `<p><strong>Processing Method:</strong> ${data.ai_analysis.processing_recommendations.processing_method}</p>` +
                                    `<h3>Quality Preservation:</h3>${createList(data.ai_analysis.processing_recommendations.quality_preservation)}`;
            container.appendChild(createCard('Processing Recommendations', processingContent));

            // Waste Prevention
            let wasteContent = `<p><strong>Salvageable Portions:</strong> ${data.ai_analysis.waste_prevention.salvageable_portions}</p>` +
                               `<h3>Alternative Uses:</h3>${createList(data.ai_analysis.waste_prevention.alternative_uses)}` +
                               `<p><strong>Composting Guidelines:</strong> ${data.ai_analysis.waste_prevention.composting_guidelines}</p>` +
                               `<h3>Prevention Measures:</h3>${createList(data.ai_analysis.waste_prevention.prevention_measures)}`;
            container.appendChild(createCard('Waste Prevention', wasteContent));

            // Economic Impact
            let economicContent = `<p><strong>Estimated Loss Percentage:</strong> ${data.ai_analysis.economic_impact.estimated_loss_percentage}</p>` +
                                  `<h3>Cost Saving Opportunities:</h3>${createList(data.ai_analysis.economic_impact.cost_saving_opportunities)}` +
                                  `<h3>Value Recovery Methods:</h3>${createList(data.ai_analysis.economic_impact.value_recovery_methods)}`;
            container.appendChild(createCard('Economic Impact', economicContent));

            // Recommendations
            let recommendationsContent = `<h3>Priority Actions:</h3>${createList(data.ai_analysis.recommendations.priority_actions)}` +
                                         `<p><strong>Timeline:</strong> ${data.ai_analysis.recommendations.timeline}</p>` +
                                         `<h3>Monitoring Points:</h3>${createList(data.ai_analysis.recommendations.monitoring_points)}`;
            container.appendChild(createCard('Recommendations', recommendationsContent));

            Toastify({
                text: "Successfully loaded analysis data!",
                duration: 3000,
                close: true,
                gravity: "top", 
                position: "right", 
                backgroundColor: "linear-gradient(to right, #00b09b, #96c93d)",
            }).showToast();

        })
        .catch(error => {
            console.error('Error fetching or parsing JSON:', error);
            Toastify({
                text: "Failed to load analysis data. Please check the console.",
                duration: 5000,
                close: true,
                gravity: "top",
                position: "right",
                backgroundColor: "linear-gradient(to right, #ff5f6d, #ffc371)",
            }).showToast();
        });
});
