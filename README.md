# 🧠 Parkinson's Detector

**AI-powered early detection of Parkinson's Disease using Voice Biomarkers and Spiral Drawing Analysis.**

© 2026 Eiman Ajmal. All Rights Reserved. See [LICENSE](./LICENSE) for details.

---

## 💡 About the Project

Parkinson's Disease is often diagnosed only after visible motor symptoms appear — by which point significant neurological changes have already taken place. Early indicators, however, can show up much sooner in subtle ways: small changes in voice steadiness, and irregularities in fine motor control like spiral drawing patterns.

This project was built to make those early signs easier to catch. It combines two independent, non-invasive detection methods into a single accessible tool:

- 🎙️ **Voice Biomarker Analysis** — analyzes vocal recordings for acoustic features linked to Parkinson's (jitter, shimmer, pitch variation, and other clinically-referenced markers).
- ✍️ **Spiral Drawing Analysis** — evaluates hand-drawn spirals for tremor patterns and motor irregularities commonly associated with early-stage Parkinson's.

This tool is intended as a **screening aid, not a medical diagnosis**. It's meant to help flag potential early signs so people can seek professional evaluation sooner — not to replace one.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | JavaScript, HTML, CSS |
| Backend | Python |
| Machine Learning | Trained models for voice + spiral feature evaluation |

**Project structure:**
```
parkinsons-detector/
├── backend/      # Python API and model inference logic
├── frontend/     # Web interface (React-based)
├── models/       # Trained ML models, feature importance & evaluation data
└── LICENSE
```

---

## 🚀 Features

- Upload or record a voice sample for instant biomarker analysis
- Upload a spiral drawing for tremor pattern evaluation
- Combined risk assessment based on both inputs
- Clean, simple web interface for non-technical users

---

## ⚙️ Getting Started

```bash
# Clone the repository
git clone https://github.com/EimanAjmal/parkinsons-detector.git
cd parkinsons-detector

# Backend setup
cd backend
pip install -r requirements.txt
python app.py

# Frontend setup
cd ../frontend
npm install
npm start
```


## ❤️ Why I Built This

This project is personal. Someone close to my family was diagnosed with Parkinson's, and by the time we recognized the signs, valuable early time had already passed. That experience is what pushed me to learn how AI could help catch these signs sooner — and to actually build something, instead of just wishing something like this existed.

---

## 📜 License

This project is protected under an **All Rights Reserved** license. No part of this code, including models or documentation, may be copied, modified, redistributed, or used commercially without written permission from the author. See the [LICENSE](./LICENSE) file for full terms.

---

## 📬 Contact

Built by **Eiman Ajmal**
🔗 [github.com/EimanAjmal](https://github.com/EimanAjmal)
