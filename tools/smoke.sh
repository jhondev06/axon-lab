#!/bin/bash
# AXON Smoke Test - Complete Pipeline Validation

set -e  # Exit on any error

echo "🚀 Starting AXON smoke test..."

# Run complete intelligence pipeline
echo "📊 Running triage..."
python -m src.brains.triage

echo "📈 Preparing dataset..."
python -m src.dataset

echo "🤖 Training models..."
python -m src.train

echo "💰 Running backtest..."
python -m src.backtest

echo "🔍 Analyzing errors..."
python -m src.brains.error_lens

echo "⚖️ Making decision..."
python -m src.brains.decision

echo "📋 Generating report..."
python -m src.report

echo "🧠 Updating memory..."
python -m src.brains.memory

echo "✅ Validating outputs..."

# Check required outputs exist
required_files=(
    "outputs/metrics/TRIAGE.json"
    "outputs/metrics/DECISION.json"
    "outputs/artifacts"
    "outputs/reports"
    "outputs/figures"
    "knowledge/findings.jsonl"
)

for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        echo "❌ Missing required output: $file"
        exit 1
    fi
done

# Check for at least one model artifact
if [ -z "$(ls -A outputs/artifacts/ 2>/dev/null)" ]; then
    echo "❌ No model artifacts found in outputs/artifacts/"
    exit 1
fi

# Check for at least one validation metric
if [ -z "$(ls outputs/metrics/VAL_*.json 2>/dev/null)" ]; then
    echo "❌ No validation metrics found (VAL_*.json)"
    exit 1
fi

# Check for at least one backtest metric
if [ -z "$(ls outputs/metrics/BT_*.json 2>/dev/null)" ]; then
    echo "❌ No backtest metrics found (BT_*.json)"
    exit 1
fi

# Check for at least one report
if [ -z "$(ls outputs/reports/*.md 2>/dev/null)" ]; then
    echo "❌ No reports found in outputs/reports/"
    exit 1
fi

# Check for at least one figure
if [ -z "$(ls outputs/figures/*.png 2>/dev/null)" ]; then
    echo "❌ No figures found in outputs/figures/"
    exit 1
fi

# Check findings.jsonl has content
if [ ! -s "knowledge/findings.jsonl" ]; then
    echo "❌ knowledge/findings.jsonl is empty"
    exit 1
fi

echo "🎉 All smoke tests passed! AXON pipeline is working correctly."
echo "📊 Check outputs/ directory for results"
echo "🧠 Check knowledge/ directory for accumulated learnings"