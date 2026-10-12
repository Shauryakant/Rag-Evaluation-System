# Deployment Guide: rag-eval-lab

This guide covers production deployment of the **FastAPI backend** (Hugging Face Spaces or Render) and the **Next.js frontend** (Vercel).

---

## 1. Environment Architecture

Connecting the two independent tiers requires exactly two environment variables:
1. `NEXT_PUBLIC_API_URL` (in Frontend on Vercel) -> points to your live backend domain (e.g. `https://your-app.hf.space` or `https://rag-eval-lab.onrender.com`).
2. `ALLOWED_ORIGINS` (in Backend) -> contains your frontend Vercel domain (e.g. `https://rag-eval-lab.vercel.app`).
3. `GROQ_API_KEY` (Server-side ONLY) -> exists strictly in backend environment variables, NEVER prefixed with `NEXT_PUBLIC_`.

---

## 2. Backend Deployment Option A: Hugging Face Spaces (Docker SDK)

Hugging Face Spaces provides 16GB RAM and 2 vCPUs on the free tier, with fast container startup.

### Steps:
1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space).
2. Space SDK: Select **Docker** (Blank).
3. Set your Space to **Public**.
4. In Space Settings -> Variables and Secrets:
   - Secret: `GROQ_API_KEY` = your Groq API key
   - Variable: `ALLOWED_ORIGINS` = `http://localhost:3000,https://<your-vercel-app>.vercel.app`
5. Push the repository to the Space remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<username>/<space-name>
   git push space main
   ```
6. The container builds using `Dockerfile`, pre-caches `bge-small` and `ms-marco-MiniLM` weights during the Docker image build, and serves on port 8000.

---

## 3. Backend Deployment Option B: Render Free Tier (Docker)

Render provides free Docker web services using the included `render.yaml` or direct Git integration.

### Steps:
1. Create a new Web Service on [render.com](https://dashboard.render.com).
2. Connect your GitHub repository `Shauryakant/Rag-Evaluation-System`.
3. Select **Docker** runtime (using `./Dockerfile`).
4. Set Environment Variables:
   - `PORT` = `8000`
   - `ALLOWED_ORIGINS` = `https://<your-vercel-app>.vercel.app`
   - `GROQ_API_KEY` = `<your-groq-key>`
5. Click **Create Web Service**.

> **Cold-Start Trade-off:** Render free tier spins down containers after 15 minutes of inactivity. Because model weights and pre-computed benchmark artifacts are baked directly into the Docker image, initial cold start takes ~30-40s rather than having to re-download 500MB from Hugging Face Hub at runtime.

---

## 4. Frontend Deployment: Vercel

The frontend is built with Next.js 15 (App Router).

### Steps:
1. Import your GitHub repository on [vercel.com](https://vercel.com/new).
2. Set **Root Directory** to `frontend`.
3. In Environment Variables, configure:
   - `NEXT_PUBLIC_API_URL` = `https://<your-backend-space-or-render-url>`
4. Click **Deploy**.
5. Once deployed, verify that `/api/health` connectivity shows "Connected" in the navigation bar.
