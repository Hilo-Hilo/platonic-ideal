#!/bin/bash
# Complete Railway + Vercel Setup Guide
# Run each section when prompted by the AI assistant

echo "═══════════════════════════════════════════════════════════════"
echo "🚀 PLATONIC IDEAL - Complete Deployment Setup"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Section 1: Railway Authentication
echo "📋 SECTION 1: RAILWAY AUTHENTICATION"
echo "────────────────────────────────────────────────────────────────"
echo "Run this command in your terminal:"
echo ""
echo "    railway login"
echo ""
echo "This will open your browser. Log in with your Railway account."
echo "After successful login, come back here and press Enter."
echo "────────────────────────────────────────────────────────────────"
read -p "Press Enter after you've logged in to Railway..."
echo ""

# Section 2: Link Railway Project
echo "📋 SECTION 2: LINK TO RAILWAY PROJECT"
echo "────────────────────────────────────────────────────────────────"
echo "Linking to your Railway project..."
railway link -p 87781121-e4c2-424b-a5a4-78944ceffa9e

if [ $? -eq 0 ]; then
    echo "✅ Project linked successfully!"
else
    echo "❌ Failed to link project. Make sure you're logged in."
    exit 1
fi
echo ""

# Section 3: Configure Railway Environment
echo "📋 SECTION 3: CONFIGURE RAILWAY ENVIRONMENT"
echo "────────────────────────────────────────────────────────────────"
echo "Setting environment variables..."

railway variables set ALLOWED_MODELS="tinyllama-1.1b,qwen-0.5b"
railway variables set PORT=8000
railway variables set ALLOWED_ORIGINS="http://localhost:3000"

echo "✅ Environment variables set!"
echo ""
echo "📝 Current Railway variables:"
railway variables
echo ""

# Section 4: Deploy to Railway
echo "📋 SECTION 4: DEPLOY BACKEND TO RAILWAY"
echo "────────────────────────────────────────────────────────────────"
echo "🚀 Deploying backend (this will take 5-10 minutes on first deploy)..."
echo ""

railway up

echo ""
echo "✅ Backend deployed to Railway!"
echo ""
echo "🔗 Your Railway URL:"
railway status | grep "railway.app"
echo ""

# Get Railway URL for next steps
RAILWAY_URL=$(railway status 2>/dev/null | grep -oE 'https://[a-zA-Z0-9-]+\.up\.railway\.app' | head -1)

if [ -z "$RAILWAY_URL" ]; then
    echo "⚠️  Could not automatically detect Railway URL."
    echo "Run 'railway status' to get your backend URL manually."
    read -p "Please paste your Railway URL here: " RAILWAY_URL
fi

echo ""
echo "Your Railway Backend URL: $RAILWAY_URL"
echo ""

# Section 5: Vercel Authentication
echo "═══════════════════════════════════════════════════════════════"
echo "📋 SECTION 5: VERCEL AUTHENTICATION"
echo "────────────────────────────────────────────────────────────────"
echo "Now let's set up the frontend on Vercel."
echo ""
read -p "Press Enter to continue with Vercel login..."
echo ""
echo "Run this command:"
echo ""
echo "    vercel login"
echo ""
echo "Follow the email verification process."
echo "────────────────────────────────────────────────────────────────"
read -p "Press Enter after you've logged in to Vercel..."
echo ""

# Section 6: Deploy to Vercel
echo "📋 SECTION 6: DEPLOY FRONTEND TO VERCEL"
echo "────────────────────────────────────────────────────────────────"
echo "Navigating to frontend directory..."
cd frontend

echo ""
echo "⚠️  IMPORTANT: When Vercel asks for environment variables,"
echo "    provide this value:"
echo ""
echo "    NEXT_PUBLIC_API_BASE_URL=$RAILWAY_URL"
echo ""
read -p "Press Enter to start Vercel deployment..."
echo ""

# Deploy to Vercel
vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Frontend deployed to Vercel!"
else
    echo ""
    echo "❌ Deployment failed. Check the error above."
    exit 1
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
echo "📝 Copy your Vercel URL from above (e.g., https://xxx.vercel.app)"
read -p "Paste your Vercel URL here: " VERCEL_URL
echo ""

# Section 7: Final CORS Update
echo "═══════════════════════════════════════════════════════════════"
echo "📋 SECTION 7: UPDATE RAILWAY CORS"
echo "────────────────────────────────────────────────────────────────"
echo "Updating Railway to allow requests from Vercel..."
cd ..

railway variables set ALLOWED_ORIGINS="$VERCEL_URL,http://localhost:3000,http://127.0.0.1:3000"

echo ""
echo "✅ CORS updated! Railway will redeploy automatically (~2 min)."
echo ""

# Final Summary
echo "═══════════════════════════════════════════════════════════════"
echo "🎉 DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "✅ Backend (Railway):  $RAILWAY_URL"
echo "✅ Frontend (Vercel):  $VERCEL_URL"
echo ""
echo "📝 What happens next:"
echo "   1. Railway is redeploying with updated CORS (~2 min)"
echo "   2. Once done, visit your Vercel URL and test the app"
echo "   3. Every git push will auto-deploy to both platforms"
echo ""
echo "🐛 If something doesn't work:"
echo "   - Check Railway logs: railway logs"
echo "   - Check Vercel logs: vercel logs"
echo "   - See CLI-DEPLOYMENT.md for troubleshooting"
echo ""
echo "═══════════════════════════════════════════════════════════════"

