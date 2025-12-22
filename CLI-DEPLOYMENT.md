# CLI Deployment Guide

I've installed both CLIs (`railway` and `vercel`) and created automated deployment scripts for you.

## 🚂 Step 1: Deploy Backend on Railway

### 1.1 Login to Railway
```bash
railway login
```
This will open your browser for authentication.

### 1.2 Run the Deployment Script
```bash
./deploy-backend.sh
```

This script will:
- Link to your Railway project (ID: `87781121-e4c2-424b-a5a4-78944ceffa9e`)
- Set environment variables (`ALLOWED_MODELS`, `PORT`, `ALLOWED_ORIGINS`)
- Deploy your backend using the `Dockerfile`

### 1.3 Get Your Railway URL
```bash
railway status
```

Copy your backend URL (e.g., `https://platonic-ideal-production.up.railway.app`)

**Save this URL** - you'll need it for the frontend!

---

## ▲ Step 2: Deploy Frontend on Vercel

### 2.1 Login to Vercel
```bash
vercel login
```
This will open your browser or ask for an email confirmation.

### 2.2 Set the Backend URL (Environment Variable)
Before deploying, set the backend URL:

```bash
cd frontend
vercel env add NEXT_PUBLIC_API_BASE_URL production
```

When prompted, paste your Railway URL from Step 1.3 (e.g., `https://platonic-ideal-production.up.railway.app`)

### 2.3 Deploy to Vercel
```bash
vercel --prod
```

Follow the prompts:
- **Set up and deploy**: Yes
- **Which scope**: Select your account
- **Link to existing project**: No (first time)
- **Project name**: `platonic-ideal` (or your choice)
- **Directory**: `./` (current, since you're in `frontend/`)

Wait ~1 minute for the build to complete.

Copy your Vercel URL (e.g., `https://platonic-ideal.vercel.app`)

---

## 🔄 Step 3: Update Railway CORS (CRITICAL!)

Now that you have your Vercel URL, update Railway to allow requests from it:

```bash
railway variables set ALLOWED_ORIGINS="https://platonic-ideal.vercel.app"
```

Replace `platonic-ideal.vercel.app` with your actual Vercel domain.

Railway will automatically redeploy with the new CORS settings (~2 minutes).

---

## ✅ Step 4: Test Your Deployment

1. Visit your Vercel URL: `https://platonic-ideal.vercel.app`
2. Open the **Configuration** dropdown
3. You should see only **TinyLlama 1.1B** and **Qwen 0.5B**
4. Add some words and click **Analyze**
5. Wait 30-60 seconds (first run downloads models)
6. Results should appear!

---

## 📊 Monitoring & Logs

### Railway Logs
```bash
railway logs
```

### Vercel Logs
```bash
vercel logs
```

Or visit the dashboards:
- Railway: https://railway.app/dashboard
- Vercel: https://vercel.com/dashboard

---

## 🔧 Future Updates

Every time you push to GitHub:
- **Railway**: Auto-deploys automatically ✅
- **Vercel**: Auto-deploys automatically ✅
- **GitHub Actions**: Runs tests first to catch errors before deployment

To manually redeploy:
```bash
# Backend
railway up

# Frontend
cd frontend && vercel --prod
```

---

## 🐛 Troubleshooting

### "Cannot connect to backend"
- Check `NEXT_PUBLIC_API_BASE_URL` in Vercel: `vercel env ls`
- Verify Railway backend is running: `railway logs`

### CORS errors
- Verify `ALLOWED_ORIGINS` in Railway: `railway variables`
- Must match your Vercel domain exactly (with `https://`)

### "Model not available"
- Check `ALLOWED_MODELS` is set in Railway: `railway variables`
- Default is `tinyllama-1.1b,qwen-0.5b`

### Railway memory errors
- TinyLlama + Qwen 0.5B need ~800MB RAM
- Upgrade to Railway Starter plan ($5/mo, 1GB RAM)

