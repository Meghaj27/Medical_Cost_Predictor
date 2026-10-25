
"""InsuraAI - Flask insurance cost prediction application."""

import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

MODEL_PATH = Path(__file__).resolve().parent / "xg_model.pkl"

FEATURES = [
    "Age", "Sex", "BMI", "Children", "Smoker", "Region"
]

DEFAULTS = {
    "Age": "30",
    "Sex": "0",
    "BMI": "25.0",
    "Children": "0",
    "Smoker": "0",
    "Region": "0"
}

# Load the trusted model once at application startup.
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

if not callable(getattr(model, "predict", None)):
    raise RuntimeError("Model does not support prediction")

if getattr(model, "n_features_in_", 6) != 6:
    raise RuntimeError("Model requires a different input shape")

model_features = list(
    getattr(model, "feature_names_in_", FEATURES)
)

if model_features != FEATURES:
    raise RuntimeError(
        f"Incorrect model features: {model_features}"
    )


def prepare_inputs(form):
    values = {
        name: str(form.get(name, "")).strip()
        for name in FEATURES
    }

    if any(not value for value in values.values()):
        raise ValueError("Please complete all six fields.")

    parsed = {}

    limits = {
        "Age": (1, 120),
        "Sex": (0, 1),
        "Children": (0, 20),
        "Smoker": (0, 1),
        "Region": (0, 3)
    }

    for name, (low, high) in limits.items():
        try:
            parsed[name] = int(values[name])
        except ValueError:
            raise ValueError(
                f"{name} must be a whole number."
            ) from None

        if not low <= parsed[name] <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}."
            )

    try:
        parsed["BMI"] = float(values["BMI"])
    except ValueError:
        raise ValueError("BMI must be a valid number.") from None

    if (
        not math.isfinite(parsed["BMI"])
        or not 10 <= parsed["BMI"] <= 80
    ):
        raise ValueError("BMI must be between 10 and 80.")

    X = pd.DataFrame(
        [{name: parsed[name] for name in FEATURES}],
        columns=FEATURES
    )

    return values, X


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#0a1124">

<title>InsuraAI | Insurance Cost Predictor</title>

<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0a1124;
    --panel: #111e35;
    --panel2: #15243d;
    --line: rgba(198,218,255,.13);
    --text: #eff5ff;
    --muted: #9aabc7;
    --aqua: #5ee7d5;
    --violet: #8888ff;
    --blue: #61b5ff;
}

* {
    box-sizing: border-box;
}

html {
    scroll-behavior: smooth;
}

body {
    margin: 0;
    min-height: 100vh;
    color: var(--text);
    font-family: 'DM Sans', system-ui, sans-serif;
    background:
        radial-gradient(
            ellipse 65% 45% at 90% 0%,
            #28215885,
            transparent 75%
        ),
        radial-gradient(
            ellipse 38% 34% at 0% 85%,
            #07566955,
            transparent 75%
        ),
        var(--bg);
}

.wrap {
    max-width: 1180px;
    margin: auto;
    padding: 0 26px 36px;
}

header {
    height: 83px;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 15px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font: 800 23px Outfit, sans-serif;
    letter-spacing: -.8px;
}

.logo {
    height: 43px;
    width: 43px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    color: #0b2435;
    background: linear-gradient(
        125deg,
        var(--aqua),
        var(--blue),
        var(--violet)
    );
    font-size: 24px;
    box-shadow: 0 8px 26px #5cd1e52c;
}

.brand em {
    font-style: normal;
    color: var(--aqua);
}

.status {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 15px;
    border: 1px solid var(--line);
    background: #12243cbb;
    border-radius: 100px;
    color: #c5d5ec;
    font-size: 12px;
    font-weight: 700;
}

.dot {
    height: 8px;
    width: 8px;
    border-radius: 50%;
    background: #67edb4;
    box-shadow: 0 0 15px #67edb4;
}

.hero {
    padding: 56px 0 35px;
}

.eyebrow {
    color: #a5a6ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 800;
}

h1, h2, h3 {
    font-family: Outfit, system-ui, sans-serif;
}

h1 {
    font-size: clamp(37px, 5.3vw, 65px);
    line-height: 1.11;
    letter-spacing: -2px;
    margin: 13px 0 15px;
    max-width: 780px;
}

h1 span {
    background: linear-gradient(100deg,#8e93ff,#5ee7d5);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero p {
    color: var(--muted);
    font-size: 15px;
    line-height: 1.8;
    max-width: 670px;
    margin: 0;
}

.tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 23px;
}

.tags span {
    border: 1px solid var(--line);
    border-radius: 9px;
    padding: 8px 13px;
    background: #12213a;
    font-size: 12px;
    color: #b4c6df;
}

.tags b {
    color: var(--aqua);
}

.layout {
    display: grid;
    grid-template-columns: minmax(0,1.1fr) minmax(320px,.9fr);
    gap: 24px;
    align-items: stretch;
}

.panel {
    border: 1px solid #91aeeb2c;
    background: linear-gradient(150deg,#172640,#101b30);
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 25px 60px #03091655;
}

.form-panel {
    padding: 30px;
}

.panel-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
}

h2 {
    font-size: 24px;
    letter-spacing: -.5px;
    margin: 0;
}

.step {
    padding: 8px 11px;
    color: #bcb7ff;
    font-size: 11px;
    font-weight: 800;
    background: #8276eb20;
    border-radius: 8px;
}

.sub {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
    margin: 9px 0 27px;
}

.fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px 17px;
}

.field {
    min-width: 0;
}

label {
    display: block;
    color: #dce8f9;
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 9px;
}

label small {
    float: right;
    color: #8094b1;
    font-size: 11px;
    font-weight: 500;
}

input, select {
    width: 100%;
    height: 50px;
    padding: 0 15px;
    border: 1px solid #a4bdd63b;
    background: #0b172b;
    color: #f1f7ff;
    border-radius: 12px;
    font: 700 14px 'DM Sans', sans-serif;
    outline: none;
}

input:focus, select:focus {
    border-color: var(--aqua);
    box-shadow: 0 0 0 3px #5ee7d526;
}

input::placeholder {
    color: #7589a7;
}

.pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}

.pair input {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
}

.pair label {
    display: grid;
    place-items: center;
    height: 50px;
    margin: 0;
    border-radius: 12px;
    border: 1px solid #a4bdd63b;
    background: #0b172b;
    color: #9cacc5;
    cursor: pointer;
    transition: .15s;
}

.pair input:checked + label {
    border-color: var(--aqua);
    background: #15505a77;
    color: #e8fffc;
}

.pair input:focus-visible + label {
    outline: 2px solid var(--aqua);
    outline-offset: 3px;
}

.help {
    padding: 14px 16px;
    border: 1px solid #62bbe630;
    border-radius: 12px;
    background: #54bce711;
    color: #b5c8e0;
    font-size: 12px;
    line-height: 1.65;
    margin: 25px 0 19px;
}

.help strong {
    color: #86e8e3;
}

.error {
    padding: 13px 16px;
    margin-bottom: 20px;
    border-radius: 12px;
    border: 1px solid #ff898966;
    color: #ffd5db;
    background: #a8364433;
    font-size: 13px;
}

.buttons {
    display: flex;
    gap: 10px;
}

button {
    cursor: pointer;
    font: 800 14px 'DM Sans', sans-serif;
    border-radius: 12px;
    height: 52px;
}

.submit {
    flex: 1;
    color: #081e2a;
    border: 0;
    background: linear-gradient(
        105deg,
        #868aff,
        #66c5fa,
        #5ee7d5
    );
    box-shadow: 0 10px 32px #42c4da26;
    transition: transform .15s;
}

.submit:hover {
    transform: translateY(-2px);
}

.submit:disabled {
    opacity: .6;
    cursor: wait;
}

.reset {
    padding: 0 20px;
    color: #bed0e8;
    background: transparent;
    border: 1px solid var(--line);
}

.result {
    display: flex;
    flex-direction: column;
}

.stripe {
    height: 4px;
    background: linear-gradient(
        90deg,
        var(--violet),
        var(--blue),
        var(--aqua)
    );
}

.result-main {
    flex: 1;
    padding: 34px 30px 30px;
    position: relative;
    background: radial-gradient(
        circle at 95% 0%,
        #236d7350,
        transparent 55%
    );
}

.result h2 {
    font-size: 26px;
    line-height: 1.2;
    margin: 12px 0;
}

.result p {
    font-size: 13px;
    line-height: 1.7;
    color: var(--muted);
    margin: 0;
    max-width: 310px;
}

.estimate {
    font: 800 clamp(37px,4.5vw,56px) Outfit,sans-serif;
    letter-spacing: -1.5px;
    overflow-wrap: anywhere;
    color: #a4ffe8;
    margin: 50px 0 10px;
}

.unit {
    font-size: 12px;
    color: #a6b6ce;
}

.empty {
    margin-top: 55px;
    border: 1px dashed #b6d0ee3b;
    border-radius: 18px;
    min-height: 215px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 20px;
    padding: 20px;
}

.circle {
    width: 82px;
    height: 82px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-size: 36px;
    border: 1px solid #75e7df44;
    background: #45d2c51a;
    color: #90f6df;
}

.result-foot {
    padding: 24px 30px 27px;
    border-top: 1px solid var(--line);
    background: #0b172b66;
}

.info-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    color: #92a6bf;
    font-size: 12px;
    margin-top: 17px;
}

.info-row strong {
    color: #e1edfc;
    text-align: right;
}

.fine {
    border-top: 1px solid var(--line);
    margin-top: 33px;
    padding-top: 22px;
    display: flex;
    justify-content: space-between;
    gap: 15px;
    flex-wrap: wrap;
    color: #7e92b1;
    font-size: 12px;
}

@media(max-width: 850px) {
    .layout {
        grid-template-columns: 1fr;
    }
    .result {
        min-height: 380px;
    }
    .estimate {
        margin-top: 30px;
    }
    .empty {
        margin-top: 28px;
    }
}

@media(max-width: 520px) {
    .wrap {
        padding: 0 15px 30px;
    }
    header {
        height: 72px;
    }
    .hero {
        padding: 39px 0 26px;
    }
    h1 {
        letter-spacing: -1.3px;
    }
    .form-panel, .result-main, .result-foot {
        padding: 23px 19px;
    }
    .fields {
        grid-template-columns: 1fr;
        gap: 18px;
    }
    .buttons {
        flex-wrap: wrap;
    }
    .submit {
        min-width: 100%;
    }
    .reset {
        flex: 1;
    }
}

</style>
</head>

<body>

<div class="wrap">

<header>
    <div class="brand">
        <div class="logo">✦</div>
        <div>Insura<em>AI</em></div>
    </div>

    <div class="status">
        <span class="dot"></span>
        XGBoost Model Ready
    </div>
</header>

<main>

<section class="hero">
    <div class="eyebrow">
        Smart insurance analytics · ML demo
    </div>

    <h1>
        Understand your costs.<br>
        <span>Predict with confidence.</span>
    </h1>

    <p>
        Get an instant insurance charge estimate using
        your trained XGBoost model. Enter six details
        below to see the predicted result.
    </p>

    <div class="tags">
        <span><b>06</b> Input features</span>
        <span>XGBoost regression</span>
        <span>Instant prediction</span>
    </div>
</section>

<section class="layout">

<div class="panel form-panel">

    <div class="panel-heading">
        <h2>Your details</h2>
        <span class="step">STEP 01 / 01</span>
    </div>

    <p class="sub">
        Fill in the information below to calculate
        the model's predicted charge.
    </p>

    {% if error %}
    <div class="error" role="alert">
        {{ error }}
    </div>
    {% endif %}

    <form method="POST" id="prediction-form">

    <div class="fields">

        <div class="field">
            <label for="age">
                Age <small>Years</small>
            </label>
            <input
                id="age"
                name="Age"
                type="number"
                min="1"
                max="120"
                step="1"
                required
                value="{{ values['Age'] }}"
            >
        </div>

        <div class="field">
            <label for="bmi">
                Body Mass Index <small>BMI</small>
            </label>
            <input
                id="bmi"
                name="BMI"
                type="number"
                min="10"
                max="80"
                step="any"
                required
                value="{{ values['BMI'] }}"
            >
        </div>

        <div class="field">
            <label>
                Sex <small>Training code</small>
            </label>

            <div class="pair">
                <input
                    id="sex0"
                    type="radio"
                    name="Sex"
                    value="0"
                    required
                    {% if values['Sex']=='0' %}checked{% endif %}
                >
                <label for="sex0">Code 0</label>

                <input
                    id="sex1"
                    type="radio"
                    name="Sex"
                    value="1"
                    {% if values['Sex']=='1' %}checked{% endif %}
                >
                <label for="sex1">Code 1</label>
            </div>
        </div>

        <div class="field">
            <label for="children">
                Children <small>Number</small>
            </label>

            <input
                id="children"
                name="Children"
                type="number"
                min="0"
                max="20"
                step="1"
                required
                value="{{ values['Children'] }}"
            >
        </div>

        <div class="field">
            <label>
                Smoker <small>Training code</small>
            </label>

            <div class="pair">
                <input
                    id="smoker0"
                    type="radio"
                    name="Smoker"
                    value="0"
                    required
                    {% if values['Smoker']=='0' %}checked{% endif %}
                >
                <label for="smoker0">Code 0</label>

                <input
                    id="smoker1"
                    type="radio"
                    name="Smoker"
                    value="1"
                    {% if values['Smoker']=='1' %}checked{% endif %}
                >
                <label for="smoker1">Code 1</label>
            </div>
        </div>

        <div class="field">
            <label for="region">
                Region <small>Training code</small>
            </label>

            <select
                id="region"
                name="Region"
                required
            >
                {% for n in range(4) %}
                <option
                    value="{{ n }}"
                    {% if values['Region']==(n|string) %}selected{% endif %}
                >
                    Region code {{ n }}
                </option>
                {% endfor %}
            </select>
        </div>

    </div>

    <div class="help">
        <strong>Important:</strong>
        Sex, Smoker and Region use the numerical
        categories from your training dataset.
        Verify which category each number represents
        in your notebook before using these predictions.
    </div>

    <div class="buttons">
        <button
            class="submit"
            id="submit"
            type="submit"
        >
            ✦ &nbsp; Predict insurance cost
        </button>

        <button
            class="reset"
            type="button"
            onclick="window.location.href='/'"
        >
            Reset
        </button>
    </div>

    </form>
</div>

<aside class="panel result" aria-live="polite">

    <div class="stripe"></div>

    <div class="result-main">

        <div class="eyebrow">
            Machine learning result
        </div>

        <h2>Your estimated charge</h2>

        <p>
            Prediction based on the six details
            entered and the saved XGBoost
            regression model.
        </p>

        {% if prediction is not none %}

            <div class="estimate">
                {{ prediction }}
            </div>

            <div class="unit">
                Model's original charge units;
                currency not specified by the model
            </div>

        {% else %}

            <div class="empty">
                <div class="circle">⌁</div>
                <p>
                    Your prediction will appear
                    here after you submit the form.
                </p>
            </div>

        {% endif %}

    </div>

    <div class="result-foot">

        <div class="eyebrow">
            Prediction details
        </div>

        <div class="info-row">
            <span>Algorithm</span>
            <strong>XGBoost Regressor</strong>
        </div>

        <div class="info-row">
            <span>Features used</span>
            <strong>6</strong>
        </div>

        <div class="info-row">
            <span>Output</span>
            <strong>
                Estimated insurance charges
            </strong>
        </div>

    </div>
</aside>

</section>
</main>

<footer class="fine">
    <span>
        © InsuraAI · Machine learning project
    </span>
    <span>
        Educational estimate, not a real insurance quote.
    </span>
</footer>

</div>

<script>
document.getElementById('prediction-form')
    .addEventListener('submit', function () {
        const btn = document.getElementById('submit');
        btn.disabled = true;
        btn.textContent = 'Calculating estimate…';
    });
</script>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    values = DEFAULTS.copy()
    prediction = None
    error = None

    if request.method == "POST":
        values = {
            name: str(request.form.get(name, "")).strip()
            for name in FEATURES
        }

        try:
            values, X = prepare_inputs(request.form)

            number = float(
                np.asarray(model.predict(X)).reshape(-1)[0]
            )

            if not math.isfinite(number):
                raise ValueError(
                    "The model returned an invalid value."
                )

            prediction = f"{number:,.2f}"

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Please check "
                "the Render application logs."
            )

    return render_template_string(
        PAGE,
        values=values,
        prediction=prediction,
        error=error
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "XGBRegressor",
        "features": FEATURES
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
