#!/bin/bash
# Vercel Frontend Deployment Script
# Run this after you've logged in with: vercel login

set -e

echo "▲ Setting up Vercel Frontend Deployment..."

# Navigate to frontend directory
cd frontend

# Interactive deployment (first time)
# This will ask you to link/create a project
echo "🚀 Deploying frontend to Vercel..."
echo "⚠️  When prompted:"
echo "   - Set up and deploy: Yes"
echo "   - Which scope: Select your account"
echo "   - Link to existing project: No (if first time)"
echo "   - Project name: platonic-ideal (or your choice)"
echo "   - Directory: ./ (current directory, since we're already in frontend/)"
echo ""
read -p "Press Enter to continue with deployment..."

# You'll need to provide the backend URL when Vercel asks for env vars
# Or set it manually after deployment
vercel --prod

echo ""
echo "✅ Frontend deployed to Vercel!"
echo "🔗 Your Vercel URL is shown above"
echo ""
echo "📝 IMPORTANT NEXT STEPS:"
echo "1. Copy your Vercel URL (e.g., https://platonic-ideal.vercel.app)"
echo "2. Go to Railway dashboard and update ALLOWED_ORIGINS with this URL"
echo "3. Set the backend URL in Vercel:"
echo "   vercel env add NEXT_PUBLIC_API_BASE_URL production"
echo "   Then enter your Railway URL when prompted"
echo "4. Redeploy Vercel to apply env var: vercel --prod"

