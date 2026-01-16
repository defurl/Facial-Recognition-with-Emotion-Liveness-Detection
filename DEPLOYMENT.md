# Deployment Guide: HuggingFace Spaces + Vercel

## Overview

- **Backend:** HuggingFace Spaces (Docker SDK)
- **Frontend:** Vercel (React)

---

## Step 1: Deploy Backend to HuggingFace Spaces

### 1.1 Create HuggingFace Account

1. Go to [huggingface.co](https://huggingface.co)
2. Create an account or sign in

### 1.2 Create New Space

1. Click your profile → **New Space**
2. Fill in:
   - **Space name:** `face-recognition-api`
   - **License:** Choose appropriate
   - **SDK:** Select **Docker**
   - **Visibility:** Public (or Private with Pro account)

### 1.3 Push Code to Space

```bash
# Clone your HuggingFace Space (empty)
git clone https://huggingface.co/spaces/YOUR_USERNAME/face-recognition-api hf-space
cd hf-space

# Copy required files from your local project
cp ../Facial-Recognition-with-Emotion-Liveness-Detection/Dockerfile .
cp ../Facial-Recognition-with-Emotion-Liveness-Detection/HF_README.md ./README.md
cp ../Facial-Recognition-with-Emotion-Liveness-Detection/requirements.txt .
cp ../Facial-Recognition-with-Emotion-Liveness-Detection/best_face_embedding_model.pth .
cp -r ../Facial-Recognition-with-Emotion-Liveness-Detection/src .
cp -r ../Facial-Recognition-with-Emotion-Liveness-Detection/backend .
cp -r ../Facial-Recognition-with-Emotion-Liveness-Detection/artifacts .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

### 1.4 Wait for Build

- HuggingFace will automatically build the Docker image
- Monitor build logs in the Space page
- Once running, test: `https://YOUR_USERNAME-face-recognition-api.hf.space/health`

---

## Step 2: Deploy Frontend to Vercel

### 2.1 Push Frontend to GitHub

```bash
cd frontend
# Create a new repo on GitHub: face-recognition-frontend
git init
git add .
git commit -m "Initial frontend"
git remote add origin https://github.com/YOUR_GITHUB/face-recognition-frontend.git
git push -u origin main
```

### 2.2 Connect Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **Add New Project**
3. Import your GitHub repo
4. Configure:
   - **Framework Preset:** Create React App
   - **Build Command:** `npm run build`
   - **Output Directory:** `build`

### 2.3 Set Environment Variables

In Vercel Dashboard → Settings → Environment Variables:

| Key                      | Value                                                 |
| ------------------------ | ----------------------------------------------------- |
| `REACT_APP_API_BASE_URL` | `https://YOUR_USERNAME-face-recognition-api.hf.space` |

### 2.4 Deploy

Click **Deploy** and wait for build to complete.

---

## Step 3: Test End-to-End

1. Open your Vercel URL
2. Camera should activate
3. Register a face
4. The registration and verification calls go to HuggingFace backend
5. Verify blink detection works

---

## Troubleshooting

### CORS Errors

Backend already has CORS configured for all origins (`allow_origins=["*"]`).

### Cold Starts

HuggingFace Spaces with Docker don't have sleep issues like Render.

### Build Failures

Check HuggingFace build logs for missing dependencies.

### Large Model Files

If model is too large for Git:

1. Use HuggingFace Hub to host the model
2. Download in Dockerfile with `huggingface_hub` library

---

## File Checklist for HuggingFace Space

```
face-recognition-api/
├── README.md           # HF_README.md content
├── Dockerfile          # Docker build config
├── requirements.txt    # Python dependencies
├── best_face_embedding_model.pth
├── src/                # Python modules
├── backend/            # FastAPI server
└── artifacts/          # Pre-trained data
    └── outputs/
        ├── employee_db.pt
        └── gui_threshold.json
```
