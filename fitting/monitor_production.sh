#!/bin/bash

# Monitor production run progress
echo "Monitoring production run for J0021+0052..."
echo "Started at: $(date)"
echo "----------------------------------------"

while true; do
    # Check if the process is still running
    if ps aux | grep -v grep | grep "run_J0021_shorter_production.py" > /dev/null; then
        # Get current time
        echo -n "$(date '+%H:%M:%S') - "
        
        # Check for latest step number in stderr
        if [ -f production_results_shorter/chains/J0021+0052_mcmc_results.npy ]; then
            echo "Chain file created - run may be finishing"
        else
            echo "Still running..."
        fi
        
        # Wait 60 seconds before next check
        sleep 60
    else
        echo "Production run completed!"
        break
    fi
done

echo "----------------------------------------"
echo "Finished at: $(date)"

# Check if results exist
if [ -f production_results_shorter/chains/J0021+0052_mcmc_results.npy ]; then
    echo "✓ Results saved successfully"
    ls -la production_results_shorter/chains/
fi

if [ -d production_results_shorter/plots ]; then
    echo "✓ Plots created successfully"
    ls -la production_results_shorter/plots/
fi