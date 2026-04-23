# AMSI-Net LULC Dashboard - Frontend

This is the Next.js frontend for the Land Use & Land Cover Recognition system.

## 🚀 Getting Started

### 1. Prerequisites
- Node.js 18+ 
- Local backend running (port 8000) or HF Space URL

### 2. Installation
```bash
npm install
```

### 3. Setup Environment
Create a `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```
*Note: Replace with your Hugging Face Space URL when live.*

### 4. Run Development Server
```bash
npm run dev
```
Access the dashboard at `http://localhost:3000`.

## 🛠 Tech Stack
- **Framework**: Next.js 16 (App Router)
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Utilities**: clsx, tailwind-merge
