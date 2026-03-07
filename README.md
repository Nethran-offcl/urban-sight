# 🏙️ Urban Sight

<div align="center">
  <img alt="Urban Sight Banner" src="frontend/public/vite.svg" width="120" />
</div>

<div align="center">
  <strong>Intelligent Urban Navigation and Safety Prediction System</strong>
</div>
<br />

<div align="center">
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-19.2-61DAFB?logo=react&logoColor=black" alt="React" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-1.0-009688?logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-Model-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-Learn" /></a>
  <a href="https://tailwindcss.com/"><img src="https://img.shields.io/badge/Tailwind_CSS-4.2-38B2AC?logo=tailwind-css&logoColor=white" alt="TailwindCSS" /></a>
</div>

---

## 📖 Overview

**Urban Sight** is a comprehensive, full-stack AI system designed to ensure safe urban navigation. By leveraging environmental factors, temporal data, and dynamic spatial markers, Urban Sight empowers users to make informed, safety-first routing decisions—whether they are driving, taking public transit, or walking. 

Under the hood, Urban Sight employs a robust Machine Learning model (`RandomForestRegressor`) trained on comprehensive synthesized urban datasets. The platform provides localized safety scores, generates comparative route profiles, and highlights risk zones via an interactive heat map.

## ✨ Key Features

- **Personalized Safety Scoring**: Real-time risk prediction factoring in hour of day, crowd density, local lighting, and transit proximity. 
- **Explainable AI (XAI)**: Understand *why* an area received its score natively through integrated SHAP value explanations.
- **Dynamic Routing Profiles**: Get automated comparative routing—compare the **Safest**, **Fastest**, and **Comfortable** paths based on intelligent geographic coordinate interpolation strings.
- **Live Risk Heatmaps**: Visualize broader area safety to aid in macro-level geographic planning.
- **Mode-Specific Adjustments**: Scoring adjusts intrinsically depending on whether the user is commuting via walking, transit, or driving.

---

## 🛠️ Technology Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) for rapid API generation and strict endpoints typing using Pydantic.
- **Machine Learning**: `scikit-learn` & `shap` for model training (`RandomForestRegressor`) and inference explanations.
- **Data Engineering**: `pandas` and `numpy` handling tabular transformations and geometric vector algebra.
- **Hosting**: Pre-configured for automatic deployments on [Render](https://render.com) using Uvicorn.

### Frontend
- **Interface**: [React 19](https://react.dev/) integrated with Vite for optimal HMR and build performance.
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/) for highly responsive, utility-first aesthetics.
- **Mapping Engine**: `leaflet` & `react-leaflet` to render routes, maps, and geographical UI components.
- **Networking**: `axios` for fast backend-frontend communications.

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python 3.9+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Nethran-offcl/urban-sight.git
cd urban-sight
```

### 2. Backend Setup
The backend contains a data factory and a model training module out of the box.

```bash
cd backend
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the ML model. 
# This generates `urban_safety.csv`, `urban_sight_model.pkl`, and `scaler.pkl`
python data_factory.py
python train_v1.py

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.*

### 3. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
*The UI will be available at `http://localhost:5173` (or the port specified by Vite).*

---

## 🔌 Core API Endpoints

- `POST /predict-risk`: Calculate a detailed safety score and SHAP explanation for a single geographic coordinate and user profile.
- `POST /route`: Generate pathfinding profiles (Safest, Fastest, Comfortable) between origin and destination waypoints.
- `GET /heatmap`: Access gridded spatial safety data across bounding box coordinates (`min_lat`, `max_lat`, `min_lng`, `max_lng`) to render visual heat maps.
- `GET /health`: Standard health check ping.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check the [issues page](https://github.com/Nethran-offcl/urban-sight/issues) if you have any ideas.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
