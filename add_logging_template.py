"""
Template Generator for Enhanced Exercise Logging
Run this to add comprehensive logging to any exercise file.
"""

LOGGING_TEMPLATE = '''
# Logging Module Template
import csv
from datetime import datetime

experiment_log = []

def log_message(message, level="INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    experiment_log.append(log_entry)

def save_results_to_csv(results, filename):
    """Save numerical results to CSV."""
    log_message(f"Saving results to {filename}...", "INFO")
    # Implement based on your results structure
    log_message(f"✓ CSV saved successfully", "SUCCESS")

def save_insights_to_md(results, filename):
    """Generate comprehensive MD report with insights."""
    log_message(f"Generating insights report to {filename}...", "INFO")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Exercise Analysis Report\\n\\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        # Add your analysis sections
    log_message(f"✓ Insights report saved successfully", "SUCCESS")
'''

print("="*70)
print("ENHANCED LOGGING TEMPLATE GENERATOR")
print("="*70)
print("\nTo add comprehensive logging to an exercise:")
print("\n1. Copy the logging module template above")
print("2. Add to your exercise file")
print("3. Replace print() with log_message()")
print("4. Add CSV/MD export functions")
print("5. Call save functions at end")
print("\nExample usage:")
print("  log_message('Starting experiment...', 'INFO')")
print("  log_message('✓ Training complete', 'SUCCESS')")
print("  save_results_to_csv(results, 'results/data.csv')")
print("  save_insights_to_md(results, 'results/insights.md')")
print("\n" + "="*70)
print("TEMPLATE READY - See LOGGING_TEMPLATE variable")
print("="*70)
