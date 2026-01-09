

````markdown
# Trust Monitor System

An AI-powered system to detect and monitor trustworthiness of accounts using ML models and a web-based UI.

## 🚀 Overview

This project applies machine learning to identify fake or low-trust accounts using a trained model (`fake_account_model.pkl`) and associated features (`fake_accounts_dataset.csv`). It provides:

✔ A frontend UI (`index.html`, `styles.css`, `script.js`)  
✔ A backend API (`api.py`) to serve predictions  
✔ Model training script (`train_model.py`)  
✔ Python dependencies (`requirements.txt`)  

## 🧠 Architecture

- **Frontend**  
  A simple HTML/JS interface to input account characteristics and show trust scores.

- **Backend API**  
  Flask (or similar) Python API that:
  - loads the trained scaler & model (`scaler.pkl`, `fake_account_model.pkl`)
  - accepts requests with account data
  - returns trust score or label.

- **Model Training**  
  `train_model.py` trains the classifier using `fake_accounts_dataset.csv`, scales features and saves artifacts.

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/Shashikala6704/trust-monitor-system.git
   cd trust-monitor-system
````

2. **Setup Python env**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the API**

   ```bash
   python api.py
   ```

4. **Open the UI**
   Open `index.html` in any browser and connect to the API.

## 🧪 Usage

* Enter account features in the web form
* Click **Predict Trust Score**
* Get classification: *Trusted* vs *Fake*

## 🔧 Project Files

| File                        | Purpose                        |
| --------------------------- | ------------------------------ |
| `api.py`                    | Backend server for predictions |
| `train_model.py`            | Script to train and save model |
| `fake_account_model.pkl`    | Trained ML model               |
| `scaler.pkl`                | Feature scaler                 |
| `fake_accounts_dataset.csv` | Training data                  |
| `index.html`                | Frontend interface             |
| `script.js`                 | Frontend logic                 |
| `styles.css`                | UI styling                     |
| `requirements.txt`          | Python dependencies            |

## 🧩 Tech Stack

* Python (Flask / API)
* scikit-learn (ML model + scaler)
* HTML/CSS/JavaScript (UI)
* CSV dataset for training

## 📈 How It Works

The backend loads the scaler and ML model, then:

```python
scaled = scaler.transform(input_features)
prediction = model.predict(scaled)
```

Frontend sends data via AJAX and renders results.

## 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first.

## 📜 License

Specify your license here (MIT / Apache / etc.).

---

If you want, I can generate a **live HTML+JS UI update**, a **refactored FastAPI backend version**, or a **deployment plan** (e.g., container + serverless) to make this production-ready — just say the word.
