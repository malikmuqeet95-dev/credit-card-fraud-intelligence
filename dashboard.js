document.getElementById('fraudForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const inspectBtn = document.getElementById('inspectBtn');
    const probVal = document.getElementById('probVal');
    const statusBadge = document.getElementById('statusBadge');
    const tableBody = document.getElementById('shapTableBody');

    inspectBtn.textContent = "Analyzing Payload & Computing SHAP...";
    inspectBtn.disabled = true;

    // Gather form inputs
    const payload = {
        features: {
            Time: parseFloat(document.getElementById('time').value),
            Amount: parseFloat(document.getElementById('amount').value),
            V14: parseFloat(document.getElementById('v14').value),
            V12: parseFloat(document.getElementById('v12').value),
            V17: parseFloat(document.getElementById('v17').value),
            V4: parseFloat(document.getElementById('v4').value),
            // Default remaining PCA features to 0.0 if omitted
            ...Array.from({length: 28}, (_, i) => i + 1).reduce((acc, i) => {
                const name = `V${i}`;
                if (!['V14', 'V12', 'V17', 'V4'].includes(name)) {
                    acc[name] = 0.0;
                }
                return acc;
            }, {})
        }
    };

    try {
        const response = await fetch('/api/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        if (result.status === 'success') {
            probVal.textContent = result.fraud_percentage;
            
            // Set Risk Tier Styling
            statusBadge.textContent = result.risk_tier;
            statusBadge.className = "status-badge";
            
            if (result.fraud_probability >= 0.75) {
                statusBadge.classList.add('status-critical');
                probVal.style.color = 'var(--accent-crimson)';
            } else if (result.fraud_probability >= 0.35) {
                statusBadge.classList.add('status-med');
                probVal.style.color = '#fbbf24';
            } else {
                statusBadge.classList.add('status-low');
                probVal.style.color = 'var(--accent-teal)';
            }

            // Populate SHAP Table
            tableBody.innerHTML = '';
            result.top_risk_drivers.forEach(driver => {
                const row = document.createElement('tr');
                const isPos = driver.shap_impact > 0;
                row.innerHTML = `
                    <td><strong>${driver.feature}</strong></td>
                    <td>${driver.actual_value}</td>
                    <td class="${isPos ? 'impact-pos' : 'impact-neg'}">${driver.shap_impact > 0 ? '+' : ''}${driver.shap_impact}</td>
                    <td>${driver.risk_direction}</td>
                `;
                tableBody.appendChild(row);
            });

        } else {
            alert("Error: " + result.message);
        }

    } catch (err) {
        console.error(err);
        alert("Failed to connect to Flask REST backend.");
    } finally {
        inspectBtn.textContent = "🔍 Evaluate Fraud Risk & Compute SHAP";
        inspectBtn.disabled = false;
    }
});