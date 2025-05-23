## 💡 Chronic Kidney Disease Prediction using Transformers

This project leverages **Transformer-based models** (BERT) to predict the presence of Chronic Kidney Disease (CKD) based on medical parameters such as age, blood pressure, albumin, sugar levels, and other key health indicators. The goal is to assist early diagnosis and raise awareness using AI-powered prediction systems.

### 🔍 Features

* ✅ Clean user interface built with HTML, CSS, and JavaScript
* 🧠 CKD prediction using pretrained **BERT model**
* 📊 Input parameters: age, BP, albumin, sugar, etc.
* 📹 Embedded awareness video on CKD
* 🌐 Responsive navigation: Home | Symptoms | Treat & Prevent | Prediction
* 🔒 Secure backend integration using Python and Flask/FastAPI

### ⚙️ Technologies Used

* Python
* BERT (Transformer-based NLP model)
* Flask / FastAPI (API backend)
* HTML, CSS, JavaScript (Frontend)
* Pandas, NumPy, Scikit-learn (Data processing)
* Matplotlib / Seaborn (Optional: Visualization)

### 🚀 How It Works

1. **User Interface** collects health metrics.
2. Data is sent to the **backend API**.
3. The **BERT model** processes inputs and returns prediction: *CKD Positive* or *Negative*.
4. Interface displays results and provides awareness resources.

### 📁 Project Structure

```
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── bert_ckd_model.pkl
├── static/
│   └── video.mp4
├── README.md
```

### 🙌 Future Enhancements

* Model optimization with larger datasets
* Doctor’s dashboard for bulk predictions
* Graphical visualizations of results

### 🧠 Note

This project is for **educational and awareness purposes only** and not intended for clinical diagnosis.
