from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load model and scalers
model = pickle.load(open('model/model.pkl', 'rb'))
scaler = pickle.load(open('model/minmaxscaler.pkl', 'rb'))  # Use only one scaler
le = pickle.load(open('model/labelencoder.pkl', 'rb'))

# Crop rules for realism check
CROP_RULES = {
    "banana":   {"temp": (24, 35), "rainfall": (1500, 2500), "humidity": (70, 100), "ph": (5.5, 7)},
    "sugarcane": {"temp": (25, 35), "rainfall": (200, 500), "humidity": (60, 90), "ph": (6, 7.5)},
    "cotton":   {"temp": (21, 35), "rainfall": (50, 120), "humidity": (30, 80), "ph": (6, 8)},
    "rice":     {"temp": (20, 38), "rainfall": (150, 300), "humidity": (60, 100), "ph": (5, 6.5)},
    "maize":    {"temp": (18, 35), "rainfall": (100, 150), "humidity": (50, 80), "ph": (5.5, 7)},
    "wheat":    {"temp": (12, 25), "rainfall": (75, 120), "humidity": (40, 80), "ph": (6, 7)},
    "apple":    {"temp": (10, 24), "rainfall": (1000, 1500), "humidity": (50, 80), "ph": (6, 7)},
    "coffee":   {"temp": (15, 30), "rainfall": (1000, 2000), "humidity": (60, 90), "ph": (5, 6.5)},
    "mungbean": {"temp": (25, 35), "rainfall": (400, 700), "humidity": (50, 80), "ph": (6, 7.5)}
}

def rule_based_crop(N, P, K, temp, humidity, ph, rainfall):
    if temp < 10 or temp > 40 or ph < 4.5 or ph > 9:
        return None

    # Rice check early due to overlap with banana
    if 20 <= temp <= 38 and rainfall >= 150 and humidity >= 60 and ph <= 6.5:
        return "rice"

    if temp >= 24:
        if humidity >= 70 and rainfall >= 1500:
            return "banana"
        elif 200 <= rainfall <= 500:
            return "sugarcane"
        elif rainfall <= 120:
            return "cotton"

    if 18 <= temp < 24:
        if 100 <= rainfall <= 150:
            return "maize"
        elif 75 <= rainfall <= 120:
            return "wheat"

    if temp <= 18:
        if rainfall >= 1000 and humidity >= 50:
            return "apple"
        elif 400 <= rainfall <= 700:
            return "mungbean"

    return None

def is_crop_possible(crop, temp, rainfall, ph, humidity):
    constraints = CROP_RULES.get(crop)
    if not constraints:
        return False
    return all([
        constraints["temp"][0] <= temp <= constraints["temp"][1],
        constraints["rainfall"][0] <= rainfall <= constraints["rainfall"][1],
        constraints["ph"][0] <= ph <= constraints["ph"][1],
        constraints["humidity"][0] <= humidity <= constraints["humidity"][1]
    ])

@app.route('/')
def index():
    return render_template("web.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        if not (0 <= ph <= 14):
            return render_template('web.html', prediction_text="❌ pH must be 0-14")
        if not (0 <= humidity <= 100):
            return render_template('web.html', prediction_text="❌ Humidity must be 0-100%")

        rule_crop = rule_based_crop(N, P, K, temp, humidity, ph, rainfall)
        if rule_crop:
            return render_template('web.html',
                prediction_text=f'✅ Best Crop: {rule_crop.upper()} (based on agri rules)')

        input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled_input = scaler.transform(input_data)
        prediction = le.inverse_transform(model.predict(scaled_input))[0]

        if not is_crop_possible(prediction, temp, rainfall, ph, humidity):
            return render_template('web.html',
                prediction_text="⚠️ No suitable crop (unrealistic prediction)")

        return render_template('web.html',
            prediction_text=f'🌿 Suggested Crop: {prediction.upper()} ✅ [ML prediction]')

    except Exception as e:
        return render_template('web.html',
            prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
