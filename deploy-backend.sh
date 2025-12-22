#!/bin/bash
# Railway Deployment Script
# Run this after you've logged in with: railway login

set -e

echo "🚂 Setting up Railway Backend Deployment..."

# Link to your Railway project
railway link -p 87781121-e4c2-424b-a5a4-78944ceffa9e

# Set environment variables
railway variables set ALLOWED_MODELS="tinyllama-1.1b,qwen-0.5b"
railway variables set PORT=8000
railway variables set ALLOWED_ORIGINS="http://localhost:3000"

echo "✅ Environment variables set!"
echo "📝 Note: You'll need to update ALLOWED_ORIGINS after deploying to Vercel"

# Deploy
echo "🚀 Deploying backend..."
railway up

echo ""
echo "✅ Backend deployed to Railway!"
echo "🔗 Get your Railway URL with: railway status"
railway status

