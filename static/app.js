// Bitcoin Vulnerability Scanner - Frontend JavaScript

class VulnerabilityScanner {
    constructor() {
        this.autopilotRunning = false;
        this.statusUpdateInterval = null;
        this.dashboardUpdateInterval = null;
        this.config = {
            explorers: {},
            defaultExplorer: 'blockstream.info'
        };
        this.init();
    }

    init() {
        this.loadConfiguration();
        this.bindEvents();
        this.startDashboardUpdates();
        this.checkAutopilotStatus();
    }

    async loadConfiguration() {
        try {
            const response = await fetch('/config');
            if (response.ok) {
                const config = await response.json();
                this.config = config;
                console.log('Configuration loaded:', config);
            } else {
                console.error('Failed to load configuration');
            }
        } catch (error) {
            console.error('Error loading configuration:', error);
        }
    }

    bindEvents() {
        // Autopilot controls
        document.getElementById('start-autopilot')?.addEventListener('click', () => this.startAutopilot());
        document.getElementById('stop-autopilot')?.addEventListener('click', () => this.stopAutopilot());
        document.getElementById('change-direction')?.addEventListener('click', () => this.changeDirection());

        // Vulnerability card clicks
        document.querySelectorAll('.vuln-card.clickable').forEach(card => {
            card.addEventListener('click', (e) => {
                const vulnType = e.currentTarget.dataset.vuln;
                this.viewVulnerabilityType(vulnType);
            });
        });

        // Copy functionality for hashes
        document.querySelectorAll('.clickable-hash').forEach(element => {
            element.addEventListener('click', (e) => {
                this.copyToClipboard(e.target.textContent);
            });
        });
    }

    async startAutopilot() {
        const startBlock = document.getElementById('start-block').value;
        const endBlock = document.getElementById('end-block').value;
        const direction = document.getElementById('direction').value;

        if (!startBlock || !endBlock) {
            this.showAlert('Please enter valid start and end block numbers', 'danger');
            return;
        }

        const formData = new FormData();
        formData.append('start_block', startBlock);
        formData.append('end_block', endBlock);
        formData.append('direction', direction);

        try {
            const response = await fetch('/autopilot/start', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.autopilotRunning = true;
                this.updateAutopilotUI();
                this.startStatusUpdates();
                this.showAlert('Autopilot started successfully', 'success');
            } else {
                this.showAlert(result.message, 'danger');
            }
        } catch (error) {
            console.error('Error starting autopilot:', error);
            this.showAlert('Failed to start autopilot', 'danger');
        }
    }

    async stopAutopilot() {
        try {
            const response = await fetch('/autopilot/stop', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                this.autopilotRunning = false;
                this.updateAutopilotUI();
                this.stopStatusUpdates();
                this.showAlert('Autopilot stopped', 'warning');
            } else {
                this.showAlert(result.message, 'danger');
            }
        } catch (error) {
            console.error('Error stopping autopilot:', error);
            this.showAlert('Failed to stop autopilot', 'danger');
        }
    }

    async changeDirection() {
        const currentDirection = document.getElementById('direction').value;
        const newDirection = currentDirection === 'forward' ? 'backward' : 'forward';

        document.getElementById('direction').value = newDirection;

        const formData = new FormData();
        formData.append('direction', newDirection);

        try {
            const response = await fetch('/autopilot/change_direction', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.showAlert(`Direction changed to ${newDirection}`, 'info');
            } else {
                this.showAlert(result.message, 'danger');
            }
        } catch (error) {
            console.error('Error changing direction:', error);
            this.showAlert('Failed to change direction', 'danger');
        }
    }

    async checkAutopilotStatus() {
        try {
            const response = await fetch('/autopilot/status');
            const status = await response.json();

            if (status.running) {
                this.autopilotRunning = true;
                this.updateAutopilotUI();
                this.startStatusUpdates();
                this.updateAutopilotStatus(status);
            }
        } catch (error) {
            console.error('Error checking autopilot status:', error);
        }
    }

    startStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
        }

        this.statusUpdateInterval = setInterval(async () => {
            if (!this.autopilotRunning) {
                this.stopStatusUpdates();
                return;
            }

            try {
                const response = await fetch('/autopilot/status');
                const status = await response.json();

                this.updateAutopilotStatus(status);

                if (!status.running) {
                    this.autopilotRunning = false;
                    this.updateAutopilotUI();
                    this.stopStatusUpdates();
                    this.showAlert('Autopilot completed', 'success');
                }
            } catch (error) {
                console.error('Error updating status:', error);
            }
        }, 3000); // Update every 3 seconds
    }

    stopStatusUpdates() {
        if (this.statusUpdateInterval) {
            clearInterval(this.statusUpdateInterval);
            this.statusUpdateInterval = null;
        }
    }

    startDashboardUpdates() {
        if (this.dashboardUpdateInterval) {
            clearInterval(this.dashboardUpdateInterval);
        }

        this.dashboardUpdateInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/dashboard_stats');
                const stats = await response.json();
                this.updateDashboardStats(stats);
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }, 30000); // Update every 30 seconds
    }

    updateAutopilotUI() {
        const startBtn = document.getElementById('start-autopilot');
        const stopBtn = document.getElementById('stop-autopilot');
        const changeBtn = document.getElementById('change-direction');
        const statusDiv = document.getElementById('autopilot-status');
        const runningSpan = document.getElementById('autopilot-running');

        if (this.autopilotRunning) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
            changeBtn.disabled = false;
            statusDiv.style.display = 'block';
            runningSpan.textContent = 'RUNNING';
            runningSpan.className = 'text-success';
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
            changeBtn.disabled = true;
            statusDiv.style.display = 'none';
            runningSpan.textContent = 'STOPPED';
            runningSpan.className = 'text-danger';
        }
    }

    updateAutopilotStatus(status) {
        const elements = {
            'current-block': status.current_block,
            'blocks-analyzed': status.blocks_analyzed,
            'total-blocks': status.total_blocks,
            'vulnerabilities-found': status.vulnerabilities_found,
            'keys-recovered': status.private_keys_recovered,
            'last-update': new Date(status.last_update).toLocaleTimeString()
        };

        for (const [id, value] of Object.entries(elements)) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        }

        // Update progress bar
        const progressBar = document.getElementById('progress-bar');
        if (progressBar && status.total_blocks > 0) {
            const percentage = (status.blocks_analyzed / status.total_blocks) * 100;
            progressBar.style.width = `${percentage}%`;
            progressBar.textContent = `${percentage.toFixed(1)}%`;
        }
    }

    updateDashboardStats(stats) {
        // Update vulnerability counts
        for (const [vulnType, count] of Object.entries(stats.vulnerability_counts)) {
            const countElement = document.querySelector(`[data-vuln="${vulnType}"] .vuln-count`);
            if (countElement) {
                countElement.textContent = count;

                // Add animation for new vulnerabilities
                if (parseInt(countElement.textContent) < count) {
                    countElement.classList.add('animate__pulse');
                    setTimeout(() => {
                        countElement.classList.remove('animate__pulse');
                    }, 1000);
                }
            }
        }
    }

    viewVulnerabilityType(vulnType) {
        window.location.href = `/vulnerabilities/${vulnType}`;
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showToast('Copied to clipboard!', 'success');
        }).catch(err => {
            console.error('Failed to copy:', err);
            this.showToast('Failed to copy to clipboard', 'error');
        });
    }

    openTransactionExplorer(txid, explorer = null) {
        if (txid && txid.length === 64) {
            // Get the explorer URL from configuration
            let explorerUrl;
            if (explorer && this.config.explorers && this.config.explorers[explorer]) {
                explorerUrl = this.config.explorers[explorer].tx + txid;
            } else if (this.config.explorers && this.config.explorers[this.config.defaultExplorer]) {
                explorerUrl = this.config.explorers[this.config.defaultExplorer].tx + txid;
            } else {
                // Fallback to blockstream.info
                explorerUrl = 'https://blockstream.info/tx/' + txid;
            }
            window.open(explorerUrl, '_blank');
        }
    }

    showAlert(message, type) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'danger' ? 'danger' : type === 'success' ? 'success' : 'info'} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        const container = document.querySelector('.container-fluid');
        container.insertBefore(alertDiv, container.firstChild);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    showToast(message, type) {
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'success' ? 'success' : 'danger'} position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 250px;';
        toast.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : 'exclamation-triangle'}"></i> ${message}
        `;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Add from_json filter for Jinja2
if (typeof window !== 'undefined') {
    window.fromJson = function(jsonString) {
        try {
            return JSON.parse(jsonString);
        } catch (e) {
            return {};
        }
    };
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.vulnerabilityScanner = new VulnerabilityScanner();
});

// Global functions for inline event handlers
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showCopyFeedback();
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

function openTransactionExplorer(txid, explorer = null) {
    if (txid && txid.length === 64) {
        // Get the scanner instance to access config
        const scanner = window.vulnerabilityScanner;
        if (scanner && scanner.config) {
            let explorerUrl;
            if (explorer && scanner.config.explorers && scanner.config.explorers[explorer]) {
                explorerUrl = scanner.config.explorers[explorer].tx + txid;
            } else if (scanner.config.explorers && scanner.config.explorers[scanner.config.defaultExplorer]) {
                explorerUrl = scanner.config.explorers[scanner.config.defaultExplorer].tx + txid;
            } else {
                // Fallback to blockstream.info
                explorerUrl = 'https://blockstream.info/tx/' + txid;
            }
            window.open(explorerUrl, '_blank');
        } else {
            // Fallback if scanner not available
            const explorerUrl = 'https://blockstream.info/tx/' + txid;
            window.open(explorerUrl, '_blank');
        }
    }
}

function showCopyFeedback(message = 'Copied to clipboard!') {
    const toast = document.createElement('div');
    toast.className = 'alert alert-success position-fixed';
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 200px;';
    toast.innerHTML = `<i class="fas fa-check"></i> ${message}`;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.remove();
    }, 2000);
}

function toggleDetails(detailsId) {
    const detailsRow = document.getElementById(detailsId);
    if (detailsRow) {
        detailsRow.style.display = detailsRow.style.display === 'none' ? 'table-row' : 'none';
    }
}

// Add any additional JavaScript functions here

function testManualRecovery() {
    // Show loading message
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'alert alert-info position-fixed';
    loadingMsg.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    loadingMsg.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing K-reuse recovery...';
    document.body.appendChild(loadingMsg);
    
    // Test the manual K-reuse recovery
    fetch('/test_manual_recovery')
        .then(response => response.json())
        .then(data => {
            console.log('Manual recovery result:', data);
            loadingMsg.remove();

            const hasSuccessfulAttempt = Boolean(data?.successful_recovery);

            if (hasSuccessfulAttempt) {
                showRecoveryModal(data, true);
                showCopyFeedback('🔓 PRIVATE KEY RECOVERED! Check modal for details.');
            } else if (data.success) {
                showRecoveryModal(data, false);
            } else {
                showRecoveryModal(data, false);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingMsg.remove();
            showErrorModal('Network Error', `Network error during recovery test: ${error.message}`);
        });
}

function showRecoveryModal(data, isSuccess) {
    // Create modal backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop fade show';
    backdrop.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1040;';
    
    // Create modal dialog
    const modal = document.createElement('div');
    modal.className = 'modal fade show';
    modal.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 1050; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto;';
    
    const title = isSuccess ? '🔓 PRIVATE KEY RECOVERED! 🔓' : '❌ Recovery Failed';
    const modalClass = isSuccess ? 'border-success' : 'border-danger';

    let content = '';
    if (isSuccess) {
        const successData = data?.successful_recovery || {};
        const privateKeyWif = successData.private_key_wif || 'Unavailable';
        const privateKeyHex = successData.private_key_hex || successData.validation_result?.private_key_hex || 'Unavailable';
        const attackMethod = successData.attack_method || 'ECDSA nonce reuse';
        const recoveryPair = successData.recovery_pair?.transactions || successData.transaction_ids || [];
        const recoveryPairLabel = recoveryPair.length ? recoveryPair.join(' ↔ ') : 'Unknown pair';
        const kValue = successData.k_value || 'Unknown';
        const transactionIds = successData.transaction_ids || data?.test_data?.transaction_ids || [];
        const primaryTxId = successData.primary_transaction_id || transactionIds[0] || null;
        const hasValidPrimaryTx = typeof primaryTxId === 'string' && primaryTxId.length === 64;
        const blockHash = successData.block_hash || data?.test_data?.block_hash || 'Unknown';
        const blockHeight = successData.block_height || data?.test_data?.block_height || 'Unknown';
        const rValue = successData.r_value || data?.test_data?.r || 'Unknown';
        const sValues = successData.s_values || [];
        const addresses = successData.addresses || {};
        const addressEntries = Object.entries(addresses);
        const addressesHtml = addressEntries.length
            ? addressEntries.map(([type, addr]) => `<div><strong>${type}:</strong> <code>${addr}</code></div>`).join('')
            : '<em>No derived addresses available.</em>';
        const copyTarget = privateKeyWif && typeof privateKeyWif === 'string' ? privateKeyWif : '';
        const copyTargetEscaped = copyTarget.replace(/'/g, "\\'");

        content = `
            <div class="mb-3">
                <strong>Private Key (WIF):</strong><br>
                <code class="text-break">${privateKeyWif}</code>
            </div>
            <div class="mb-3">
                <strong>Private Key (HEX):</strong><br>
                <code class="text-break">${privateKeyHex}</code>
            </div>
            <div class="mb-3">
                <strong>Recovery Method:</strong> ${attackMethod}
            </div>
            <div class="mb-3">
                <strong>Recovery Pair:</strong> ${recoveryPairLabel}
            </div>
            <div class="mb-3">
                <strong>K-value:</strong> ${kValue}
            </div>
            <div class="mb-3">
                <strong>Transaction:</strong> ${hasValidPrimaryTx
                    ? `<a href="#" onclick="openTransactionExplorer('${primaryTxId}'); return false;">${primaryTxId}</a>`
                    : '<span>Unavailable</span>'}
            </div>
            <div class="mb-3">
                <strong>Block:</strong> ${blockHash} (Height: ${blockHeight})
            </div>
            <div class="mb-3">
                <strong>R-value:</strong> ${rValue}
            </div>
            <div class="mb-3">
                <strong>S-values:</strong> ${sValues.length ? sValues.join(', ') : 'Unknown'}
            </div>
            <div class="mb-3">
                <strong>Addresses Generated:</strong><br>
                ${addressesHtml}
            </div>
            <div class="alert alert-success">
                <i class="fas fa-check-circle"></i> This demonstrates successful exploitation of the nonce reuse vulnerability!
            </div>
        `;
    } else {
        const totalAttempts = data?.attempt_pairs_evaluated || 0;
        const successfulAttempts = data?.successful_recoveries || 0;
        const firstTxId = data?.test_data?.transaction_ids?.[0] || data?.transaction_id || null;
        const hasValidFirstTx = typeof firstTxId === 'string' && firstTxId.length === 64;

        content = `
            <div class="mb-3">
                <strong>Reason:</strong> ${data.message || data.error || 'No successful recoveries were produced.'}
            </div>
            <div class="mb-3">
                <strong>Transaction:</strong> ${hasValidFirstTx
                    ? `<a href="#" onclick="openTransactionExplorer('${firstTxId}'); return false;">${firstTxId}</a>`
                    : '<span>Unavailable</span>'}
            </div>
            <div class="mb-3">
                <strong>Attempted Pairs:</strong> ${totalAttempts}
            </div>
            <div class="mb-3">
                <strong>Successful Recoveries:</strong> ${successfulAttempts}
            </div>
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i> The k-reuse attack was not successful with the current signature data.
            </div>
        `;
    }
    
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content border ${modalClass}">
                <div class="modal-header bg-${isSuccess ? 'success' : 'danger'} text-white">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close btn-close-white" onclick="closeRecoveryModal()"></button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="closeRecoveryModal()">Close</button>
                    ${isSuccess && copyTarget ? `<button type="button" class="btn btn-primary" onclick="copyPrivateKey('${copyTargetEscaped}')">Copy Private Key</button>` : ''}
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(backdrop);
    document.body.appendChild(modal);
    
    // Close on backdrop click
    backdrop.addEventListener('click', closeRecoveryModal);
}

function showErrorModal(title, message) {
    const backdrop = document.createElement('div');
    backdrop.className = 'modal-backdrop fade show';
    backdrop.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1040;';
    
    const modal = document.createElement('div');
    modal.className = 'modal fade show';
    modal.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 1050; max-width: 500px; width: 90%;';
    
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content border-danger">
                <div class="modal-header bg-danger text-white">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="btn-close btn-close-white" onclick="closeRecoveryModal()"></button>
                </div>
                <div class="modal-body">
                    <p>${message}</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="closeRecoveryModal()">Close</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(backdrop);
    document.body.appendChild(modal);
    
    backdrop.addEventListener('click', closeRecoveryModal);
}

function closeRecoveryModal() {
    const backdrop = document.querySelector('.modal-backdrop');
    const modal = document.querySelector('.modal');
    if (backdrop) backdrop.remove();
    if (modal) modal.remove();
}

function copyPrivateKey(privateKey) {
    navigator.clipboard.writeText(privateKey).then(() => {
        showCopyFeedback('Private key copied to clipboard!');
        closeRecoveryModal();
    }).catch(err => {
        console.error('Failed to copy private key:', err);
        showCopyFeedback('Failed to copy private key');
    });
}