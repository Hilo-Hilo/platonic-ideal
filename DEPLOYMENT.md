# Deployment Guide: Vercel + Railway

## Overview
- **Frontend** → Deploy on Vercel
- **Backend** → Deploy on Railway

---

## Part 1: Deploy Backend on Railway

### Step 1: Connect GitHub Repository
1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your `platonic-ideal` repository
4. Railway will auto-detect the `Dockerfile` ✅

### Step 2: Configure Environment Variables
In Railway's **Variables** tab, add:

```env
ALLOWED_MODELS=tinyllama-1.1b,qwen-0.5b
PORT=8000
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

**Important**: Replace `your-frontend.vercel.app` with your actual Vercel domain (you'll get this in Part 2).

### Step 3: Deploy
- Railway will automatically build and deploy from the `Dockerfile`
- Wait 3-5 minutes for the first build (downloads models and NLTK data)
- Copy the **Public Domain** (e.g., `https://platonic-ideal-production.up.railway.app`)
- Save this URL for Part 2!

### Optional: Add Redis for Session Locking
If you want distributed session locking (recommended for production):
1. In Railway, click **"New"** → **"Database"** → **"Add Redis"**
2. Railway will automatically create a `REDIS_URL` environment variable
3. Your backend will use it automatically (already coded in `session_lock.py`)

---

## Part 2: Deploy Frontend on Vercel

### Step 1: Connect GitHub Repository
1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"Add New..."** → **"Project"**
3. Import your `platonic-ideal` repository
4. Configure the project:
   - **Framework Preset**: Next.js ✅ (auto-detected)
   - **Root Directory**: `frontend` ⚠️ (IMPORTANT!)

### Step 2: Configure Environment Variables
In the **Environment Variables** section, add:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-backend.up.railway.app
```

Replace `your-backend.up.railway.app` with the Railway URL from Part 1.

### Step 3: Deploy
- Click **"Deploy"**
- Vercel will build and deploy in ~1 minute
- Copy your **Production Domain** (e.g., `https://platonic-ideal.vercel.app`)

### Step 4: Update Backend CORS (CRITICAL!)
Go back to **Railway** → **Variables** and update:

```env
ALLOWED_ORIGINS=https://platonic-ideal.vercel.app,https://your-frontend.vercel.app
```

Use your actual Vercel domain(s). Railway will automatically redeploy with the new CORS settings.

---

## Testing Your Deployment

1. Visit your Vercel URL: `https://platonic-ideal.vercel.app`
2. Open the Configuration dropdown
3. You should see **only** TinyLlama 1.1B and Qwen 0.5B (as configured)
4. Add words to a group and click **Analyze**
5. Results should appear within 30-60 seconds

---

## Troubleshooting

### Frontend shows "Cannot connect to backend"
- Check that `NEXT_PUBLIC_API_BASE_URL` in Vercel matches your Railway URL exactly
- Railway URLs should start with `https://` (not `http://`)

### Backend returns CORS errors
- Ensure `ALLOWED_ORIGINS` in Railway includes your Vercel domain
- Check Railway logs: **Deployments** → Click latest → **View Logs**

### "Model not available" errors
- Verify `ALLOWED_MODELS` is set in Railway
- Default is `tinyllama-1.1b,qwen-0.5b` if not set

### Backend crashes with memory errors
- TinyLlama + Qwen 0.5B need ~800MB RAM
- Railway's free tier has 512MB; upgrade to **Starter Plan** ($5/mo, 1GB RAM)

---

## Cost Estimate

- **Vercel**: Free tier (plenty for this app)
- **Railway**: $5-10/mo (Starter plan recommended, 1GB RAM)
- **Total**: ~$5-10/mo

---

## Alternative: Single Railway Deployment (Advanced)

If you want to deploy both frontend and backend on Railway (costs more but simpler):

1. Use the `docker-compose.yml` approach (requires Railway Pro)
2. Or create two separate Railway services:
   - Service 1: Backend (use root `Dockerfile`)
   - Service 2: Frontend (use `frontend/Dockerfile`)
   - Link them via Railway's internal networking

This is more expensive but gives you full control.

