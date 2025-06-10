# 🌿 GreenGuard - AI-Powered Plant Disease Detection System

## 🌟 Overview

GreenGuard is an intelligent web application that helps farmers, gardeners, and plant enthusiasts quickly identify diseases in their plants using artificial intelligence. Simply upload a photo of a plant leaf or use your device's camera, and our system will analyze it to detect potential diseases with treatment recommendations.

**Key Features:**
- 🚀 Instant disease detection using advanced machine learning
- 📸 Two convenient detection methods: image upload or live camera
- 🌱 Supports 11 plant types with 38 different disease classifications
- 💡 Detailed treatment and prevention advice for each diagnosis
- 📱 Fully responsive design works on all devices

## 🛠️ How It Works

Our system uses a trained MobileNetV2 deep learning model to analyze plant images:

1. **Capture** - Take a clear photo of a plant leaf or use your webcam
2. **Upload** - Submit the image to our analysis system
3. **Analyze** - Our AI model processes the image in seconds
4. **Results** - Get the diagnosis with confidence level and treatment advice

## 🌿 Supported Plants

GreenGuard can detect diseases in these common plants:
- 🍎 Apple
- 🔵 Blueberry
- 🍒 Cherry
- 🌽 Corn
- 🍇 Grape
- 🍊 Orange
- 🍑 Peach
- 🌶️ Pepper
- 🥔 Potato
- 🍓 Strawberry
- 🍅 Tomato

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Flask
- Pillow (PIL)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hamzanawazsangha/Plants-Disease-Detection-System.git
   cd greenguard
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have:
   - `mobilenetv2_plant_model_final.keras` (your trained model)
   - `class_names.json` (list of class names)

### Running the Application

Start the Flask development server:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🏗️ Project Structure

```
greenguard/
├── app.py                # Flask application
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Main application interface
├── mobilenetv2_plant_model_final.keras  # Trained model
└── class_names.json      # Class names for predictions
```

## 🌐 Deployment Options

For production deployment, consider:

1. **Docker**:
   ```bash
   docker build -t greenguard .
   docker run -p 5000:5000 greenguard
   ```

2. **Cloud Platforms**:
   - AWS Elastic Beanstalk
   - Google App Engine
   - Azure App Service
   - Heroku

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📬 Contact

For questions or support, please contact:

Project Lead: Muhammad Hamza Nawaz  
Email: iamhamzanawaz14@gmail.com 
Project Link: [https://github.com/hamzanawazsangha/Plants-Disease-Detection-System](https://github.com/hamzanawazsangha/Plants-Disease-Detection-System)

---

**🌱 Happy Planting!** GreenGuard helps you keep your plants healthy and thriving with AI-powered disease detection. Try it today and give your plants the care they deserve!
