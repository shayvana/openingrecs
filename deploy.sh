#!/bin/bash

# Quick deployment script for Vercel
# Usage: ./deploy.sh [preview|production]

set -e

echo "========================================="
echo "Chess Opening Recommendations Deployment"
echo "========================================="
echo ""

MODE=${1:-preview}

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found"
    echo ""
    echo "Install with: npm i -g vercel"
    exit 1
fi

# Check if network files exist
if [ ! -f "data/relatedness_network.pkl" ]; then
    echo "❌ Network files not found!"
    echo ""
    echo "Please ensure data/relatedness_network.pkl exists"
    exit 1
fi

# Show file sizes
echo "📊 Deployment files:"
du -h data/*.pkl 2>/dev/null | grep -v bipartite || true
echo ""

# Check total size (excluding bipartite network)
TOTAL_SIZE=$(du -sh . | awk '{print $1}')
echo "Total project size: $TOTAL_SIZE"
echo ""

# Confirm deployment
if [ "$MODE" == "production" ]; then
    echo "🚀 Deploying to PRODUCTION"
    echo ""
    read -p "Are you sure? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 1
    fi

    vercel --prod
else
    echo "🔍 Deploying to PREVIEW"
    echo ""
    vercel
fi

echo ""
echo "========================================="
echo "✅ Deployment complete!"
echo "========================================="
echo ""
echo "Test your deployment:"
echo "  curl https://YOUR_URL/health"
echo "  curl -X POST https://YOUR_URL/recommend -d 'username=DrNykterstein'"
echo ""
