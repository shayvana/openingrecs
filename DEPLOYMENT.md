# Deploying to Vercel

## Quick Deployment (5 minutes)

Since your main site is already on Vercel, deploying this app is straightforward.

### Prerequisites

- Vercel account (you already have one)
- Vercel CLI installed: `npm i -g vercel`
- Git repository (recommended)

### Option 1: Deploy via Vercel CLI (Fastest)

```bash
cd /Users/shayvana/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Code/Projects/chessopeningrecs

# Login to Vercel (if not already)
vercel login

# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

That's it! Vercel will:
- Detect the `vercel.json` configuration
- Install Python dependencies from `requirements.txt`
- Build and deploy your app
- Give you a URL like `https://chessopeningrecs.vercel.app`

### Option 2: Deploy via GitHub (Recommended for Production)

1. **Push to GitHub:**
   ```bash
   cd /Users/shayvana/Library/Mobile\ Documents/com~apple~CloudDocs/Documents/Code/Projects/chessopeningrecs

   git init
   git add .
   git commit -m "Initial commit - Chess opening recommendations v2.0"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/chessopeningrecs.git
   git push -u origin main
   ```

2. **Connect to Vercel:**
   - Go to https://vercel.com/new
   - Import your GitHub repository
   - Vercel auto-detects Python project
   - Click "Deploy"

3. **Configure Domain (Optional):**
   - In Vercel dashboard, go to Settings → Domains
   - Add custom domain: `openings.serialexperiment.ing`
   - Add DNS records (Vercel provides them)

## Configuration Details

### vercel.json

Already configured in your project:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app/app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app/app.py"
    }
  ],
  "env": {
    "FLASK_ENV": "production"
  }
}
```

### Large Files Issue

Your network files are large:
- `data/relatedness_network.pkl` - ~82 KB
- `data/relatedness_network_metadata.pkl` - ~11 MB
- `data/bipartite_network.pkl` - ~85 MB (if included)

**Vercel Limits:**
- Deployment size: 250 MB (you're fine)
- Serverless function: 50 MB (you might hit this)

**Solutions:**

#### Solution A: Use Git LFS (Recommended)

```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track large files
git lfs track "data/*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"

# Add and commit files
git add data/
git commit -m "Add network data files"
git push
```

Vercel supports Git LFS automatically.

#### Solution B: Download on Build

Create `build.sh`:

```bash
#!/bin/bash
# Download pre-built network files from cloud storage
curl -o data/relatedness_network.pkl https://YOUR_STORAGE_URL/relatedness_network.pkl
curl -o data/relatedness_network_metadata.pkl https://YOUR_STORAGE_URL/metadata.pkl
```

Update `vercel.json`:
```json
{
  "builds": [
    {
      "src": "app/app.py",
      "use": "@vercel/python"
    }
  ],
  "buildCommand": "bash build.sh"
}
```

#### Solution C: Exclude bipartite network

The bipartite network (85 MB) is only used for rebuilding. If you don't need it in production:

Create `.vercelignore`:
```
data/bipartite_network.pkl
data/bipartite_network_june2024.pkl
data/*_fresh.pkl
*.pgn
tests/
scripts/
*.md
.git/
__pycache__/
*.pyc
```

This keeps deployment small while including the essential files.

## Environment Variables

No secrets needed! Your app uses public Lichess API.

If you want to add analytics or other services later:

```bash
vercel env add ANALYTICS_KEY
```

## Performance Optimization

### 1. Add Caching Headers

Already implemented in `app/app.py`:
```python
@app.after_request
def add_header(response):
    response.cache_control.max_age = 3600  # 1 hour
    return response
```

### 2. Enable Serverless Function Caching

Vercel automatically caches your serverless functions. The recommendation engine caches loaded networks in memory (already implemented).

### 3. Cold Start Optimization

Your app loads network files on first request, which can be slow. Consider:

```python
# In app/app.py, force load on import (not just on request)
_recommendation_engine = None

def init_app():
    global _recommendation_engine
    with open('data/relatedness_network.pkl', 'rb') as f:
        network = pickle.load(f)
    _recommendation_engine = RecommendationEngine(network)

# Call at module level
init_app()
```

## Monitoring

### View Logs

```bash
vercel logs YOUR_DEPLOYMENT_URL
```

### Monitor Performance

Vercel dashboard shows:
- Response times
- Error rates
- Bandwidth usage
- Function duration

## Custom Domain Setup

### Subdomain of serialexperiment.ing

1. **In Vercel Dashboard:**
   - Go to your deployment → Settings → Domains
   - Add: `openings.serialexperiment.ing`

2. **In Your DNS Provider (wherever serialexperiment.ing is hosted):**
   - Add CNAME record:
     ```
     Type: CNAME
     Name: openings
     Value: cname.vercel-dns.com
     ```

3. **Wait for DNS propagation** (5-10 minutes)

4. **Vercel auto-issues SSL certificate**

### Alternative Domain Names

Good options:
- `chess.serialexperiment.ing`
- `openings.serialexperiment.ing`
- `theory.serialexperiment.ing`
- `repertoire.serialexperiment.ing`

## Deployment Checklist

- [ ] Network files present in `data/`
- [ ] `requirements.txt` up to date
- [ ] `vercel.json` configured
- [ ] `.vercelignore` created (optional, for size optimization)
- [ ] Git LFS configured (if using large files)
- [ ] Repository pushed to GitHub (if using GitHub integration)
- [ ] Vercel CLI installed or GitHub connected
- [ ] Deploy command run: `vercel --prod`
- [ ] Custom domain configured (optional)
- [ ] SSL certificate active
- [ ] Test deployment with real username

## Testing Deployment

Once deployed, test with:

```bash
# Test health endpoint
curl https://YOUR_URL/health

# Test recommendation endpoint
curl -X POST https://YOUR_URL/recommend \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=DrNykterstein"
```

Or visit in browser:
- Main page: `https://YOUR_URL/`
- Try username: DrNykterstein, penguingm1, or GMWSO

## Troubleshooting

### "Application Error" on Vercel

Check logs: `vercel logs`

Common issues:
1. **Module not found**: Missing in `requirements.txt`
2. **File not found**: Network files missing (use Git LFS)
3. **Memory limit**: Reduce network file size or upgrade plan

### Slow Cold Starts

First request after inactivity can be slow (5-10 seconds) as Vercel spins up serverless function and loads network files.

Solutions:
- Upgrade to Pro plan (keeps functions warm)
- Use Vercel Edge Functions (faster cold starts)
- Add health check ping every 5 minutes

### Network File Size Issues

If deployment fails due to size:
1. Use Git LFS (recommended)
2. Host files on S3/Cloud Storage and download at runtime
3. Reduce network size (fewer openings, higher z-score threshold)

## Cost

**Free Tier (Hobby):**
- 100 GB bandwidth/month
- Unlimited requests
- Good for personal projects

**Your usage estimate:**
- Network files: ~100 MB
- Each request: ~10-50 KB response
- Expected: Well within free tier

## Updating Deployment

After making changes:

```bash
# If using Vercel CLI
git add .
git commit -m "Update recommendations"
git push
vercel --prod

# If using GitHub integration
git add .
git commit -m "Update recommendations"
git push
# Vercel auto-deploys!
```

## Recommended Approach

For your use case (personal site already on Vercel):

1. **Create `.vercelignore`** to exclude large unnecessary files
2. **Push to GitHub** with Git LFS for network files
3. **Connect GitHub repo to Vercel** (auto-deploy on push)
4. **Add custom domain**: `openings.serialexperiment.ing`
5. **Monitor in Vercel dashboard**

This gives you:
- Automatic deployments on git push
- SSL certificate
- Global CDN
- Integration with your existing Vercel setup

## Next Steps

```bash
# 1. Create .vercelignore
echo "data/bipartite_network*.pkl
*.pgn
tests/
scripts/
*.md
__pycache__/" > .vercelignore

# 2. Initialize Git (if not already)
git init
git add .
git commit -m "Initial deployment - Chess opening recommendations"

# 3. Create GitHub repo and push
# (Use GitHub web UI to create repo)
git remote add origin https://github.com/YOUR_USERNAME/chess-openings.git
git push -u origin main

# 4. Deploy via Vercel
vercel --prod
```

Your app will be live in ~2 minutes!
