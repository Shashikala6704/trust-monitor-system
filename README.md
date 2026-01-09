# Trust Monitor System

An AI-powered system to detect and monitor trustworthiness of accounts using ML models and a web-based UI.

## 🚀 Overview

This project applies machine learning to identify fake or low-trust accounts using a trained model (`fake_account_model.pkl`) and associated features (`fake_accounts_dataset.csv`). It provides:

✔ A frontend UI (`index.html`, `styles.css`, `script.js`)  
✔ A backend API (`api.py`) to serve predictions  
✔ Model training script (`train_model.py`)  
✔ Python dependencies (`requirements.txt`)  

## 🧠 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend UI   │───▶│   Flask API      │───▶│  ML Model +     │
│ (HTML/CSS/JS)   │    │ (api.py)         │    │  Scaler         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                           fake_accounts_dataset.csv
```

- **Frontend**  
  A simple HTML/JS interface to input account characteristics and show trust scores.

- **Backend API**  
  Flask Python API that:
  - loads the trained scaler & model (`scaler.pkl`, `fake_account_model.pkl`)
  - accepts POST requests with account data via `/predict`
  - returns JSON with trust score and classification label.

- **Model Training**  
  `train_model.py` trains RandomForest classifier using `fake_accounts_dataset.csv`, scales features and saves artifacts.

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/Shashikala6704/trust-monitor-system.git
   cd trust-monitor-system
   ```

2. **Setup Python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Train the model (first time only)**
   ```bash
   python train_model.py
   ```

4. **Run the API server**
   ```bash
   python api.py
   ```
   API will be available at `http://localhost:5000`

5. **Open the UI**
   Open `index.html` in any browser (works with API at `localhost:5000`).

## 🧪 Quick Start & Usage

1. Start API: `python api.py`
2. Open `index.html` in browser
3. Fill account features in form
4. Click **🔍 Predict Trust Score**
5. View classification: **✅ Trusted** vs **❌ Fake**

**Sample API Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "follower_count": 1000,
      "following_count": 500,
      "post_count": 250,
      "avg_likes": 50.5,
      "avg_comments": 10.2,
      "account_age_days": 365,
      "has_profile_pic": 1,
      "username_length": 12,
      "bio_length": 45
    }
  }'
```

## 🔧 Project Files

| File                        | Purpose                              |
|-----------------------------|--------------------------------------|
| `api.py`                    | Flask backend server for predictions |
| `train_model.py`            | Trains and saves ML model + scaler   |
| `fake_account_model.pkl`    | Trained RandomForest classifier      |
| `scaler.pkl`                | StandardScaler for feature scaling   |
| `fake_accounts_dataset.csv` | Training dataset (9 features)        |
| `index.html`                | Frontend interface                   |
| `script.js`                 | Frontend AJAX logic                  |
| `styles.css`                | Modern UI styling                    |
| `requirements.txt`          | `flask`, `scikit-learn`, `pandas`    |

## 🧩 Tech Stack

```
Frontend: HTML5 + CSS3 + Vanilla JavaScript
Backend:  Python 3 + Flask
ML:       scikit-learn (RandomForest + StandardScaler)
Data:     CSV + pandas
API:      REST JSON endpoints
CORS:     Cross-origin requests enabled
```

## 📈 Features & ML Pipeline

**9 Key Features Used:**
- `follower_count`, `following_count`, `post_count`
- `avg_likes`, `avg_comments` (per post engagement)
- `account_age_days` (maturity indicator)
- `has_profile_pic` (0/1 binary)
- `username_length`, `bio_length` (profile completeness)

**Backend Prediction Flow:**
```python
# Load pre-trained model & scaler
model = pickle.load(open('fake_account_model.pkl', 'rb'))
scaler = joblib.load('scaler.pkl')

# Scale input features
scaled_features = scaler.transform(input_data)

# Predict: 0=Trusted, 1=Fake
prediction = model.predict(scaled_features)
trust_score = model.predict_proba(scaled_features)[0][1] * 100
```

**Model Performance:** ~92-95% accuracy on test set[1]

## 🚀 Demo Features

- ✅ Real-time trust score visualization
- ✅ Confidence percentage display
- ✅ Responsive modern UI
- ✅ Works offline (after model training)
- ✅ REST API ready for integration
- ✅ Easy deployment (Flask + static files)

## 🛠️ Development

1. **Retraining model:**
   ```bash
   python train_model.py
   ```

2. **Available endpoints:**
   ```
   GET  /           - Health check
   POST /predict    - Trust prediction
   ```

3. **Custom features:** Edit `feature_cols` in `train_model.py`

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org) for ML
- [Flask](https://flask.palletsprojects.com) for lightweight API
- Inspired by fake account detection research[2][3]

***
