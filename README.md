# Renewable Energy Monitoring System for Microgrids

An IoT-based monitoring system for rural solar/wind microgrids. It watches generation, storage, and consumption in real time, flags faults using both hard-coded rules and an on-device machine learning model, sheds load automatically when supply falls short, and pushes everything to a live web dashboard.

---

## 1. What This Project Actually Does

A microgrid can fail in ways a simple threshold check won't catch — a partially shaded panel, a degrading connection between two modules, one string quietly disconnecting. So this system runs two layers of fault detection side by side instead of relying on just one:

1. **Rule-based checks** — fast, deterministic, catches the obvious stuff (overvoltage, overcurrent, low battery, overheating).
2. **An ML fault classifier running directly on the ESP32** — a small neural network trained to recognize five specific fault signatures from electrical + environmental readings, with no cloud round-trip needed for inference.

On top of that, the firmware does **priority-based load shedding**: if available power drops below what's needed, it disconnects non-critical loads first, then essential, and keeps critical loads alive as long as possible.

Everything — live readings, historical data, and alerts — is pushed to **Firebase Realtime Database** and rendered on a **React + TypeScript + Tailwind CSS** dashboard.

Two separate hardware simulations were built (in Wokwi) to develop and test this, each pairing different sensors with a different anomaly-detection approach.

---

## 2. System Architecture

```
Sensors (pots/DHT22/DS18B20/photoresistor)
        │
        ▼
  Engineering-unit conversion (voltage, current, temp, irradiance, SOC)
        │
        ▼
  Rule-based fault checks  ──────┐   (hysteresis + 60s cooldown per fault)
        │                        │
        ▼                        ▼
  ML fault classifier      Priority load shedding
  (predict_fault())        (critical → essential → non-critical relays)
        │                        │
        └───────────┬────────────┘
                     ▼
        WiFi → Firebase Realtime DB
        (live.json every 5s, history.json every 10s, alerts.json on fault)
                     │
                     ▼
        React + TypeScript + Tailwind dashboard
```

---

## 3. The Two Simulations

### Simulation 1 — Full Pipeline with the Neural-Network Classifier

This is the main build: two simulated PV strings, a battery, and an environmental sensor suite, all feeding a **6-input → 32 → 16 → 5-output MLP** that was trained offline and hand-exported to a C header (no TensorFlow Lite Micro runtime needed on the ESP32).

**Sensors / actuators**

| Component | Purpose |
|---|---|
| Potentiometer × 3 | Simulate PV string 1 voltage, PV string 1 current, PV string 2 voltage |
| Potentiometer (VN pin) | Simulates PV string 2 current |
| Battery SOC potentiometer | Simulates battery state of charge |
| Photoresistor (LDR) | Irradiance proxy |
| DHT22 | Ambient temperature + humidity |
| DS18B20 × 2 | Battery temperature, panel temperature (shared 1-Wire bus, distinct ROM IDs) |
| Relay × 3 | Critical / essential / non-critical load switching |
| LED × 3 | Visual indicator per relay |

**Pin map (ESP32 DevKit-C, all ADC1 — safe to read while WiFi is active)**

| Signal | GPIO | Notes |
|---|---|---|
| PV string 1 voltage (vdc1) | 34 | 0–400 V simulated range |
| PV string 1 current (idc1) | 35 | 0–10 A simulated range |
| PV string 2 voltage (vdc2) | 32 | Reused from an old load-demand pot — see note below |
| Battery SOC | 33 | 0–100 % |
| Irradiance (LDR) | 36 (VP) | Input-only ADC1 pin |
| PV string 2 current (idc2) | 39 (VN) | Input-only ADC1 pin |
| DHT22 data | 16 | — |
| DS18B20 bus | 4 | Shared by both temp sensors |
| Critical / Essential / Non-critical relays | 25 / 26 / 27 | — |

> A standard 30-pin ESP32 DevKit only exposes 6 ADC1 pins (32, 33, 34, 35, 36, 39), and this build uses all six. **GPIO37/38 don't physically exist on this board** and shouldn't show up anywhere in future revisions of the wiring or code.

The **load demand** input isn't read from a potentiometer in this version — it's generated in software as a triangle wave (`computeSyntheticLoad()`, 0–500 W, one ramp up-and-down per minute). That freed up GPIO32, which now reads PV string 2's voltage instead.

**The ML model (`fault_detector_model.h`)**

- Architecture: 6 → 32 (ReLU) → 16 (ReLU) → 5 (softmax), fully hand-rolled — no external ML library, just nested loops and `expf()`.
- Inputs, in this exact order: `idc1, idc2, vdc1, vdc2, irradiance, panel_temperature`.
- Inputs are standardized first (`(x - mean) / scale`) using parameters baked in from the training set.
- Output classes:
  1. Normal operation
  2. Short-circuit between two modules of a string
  3. Degradation (resistance fault between two modules)
  4. Open circuit (one string disconnected from the inverter)
  5. Shadowing (one or more modules shaded)
- `predict_fault()` returns the class index and (optionally) the full probability vector, so the firmware can log a confidence score alongside the fault name.

### Simulation 2 — Lighter Setup with an Isolation-Forest-Style Anomaly Scorer

A second, simpler Wokwi build swaps the neural net for a hand-rolled **isolation-forest-style anomaly scorer** — really a bounds-and-consistency check dressed up in the isolation-forest API (`decisionFunction()` / `predict()`), rather than a trained ensemble of isolation trees.

**Sensors / actuators**

| Component | Purpose |
|---|---|
| Potentiometer × 3 | Simulated voltage, current, and a third analog input |
| DS18B20 | Temperature |
| DHT22 | Ambient temperature + humidity |
| LED × 2 + resistors | Status indicators (e.g. normal / anomaly) |
| Buzzer | Audible alert on anomaly |
| Clock generator (50 Hz) | Simulates a frequency signal, e.g. grid/AC line frequency |

**How the scorer works**

`IsolationForest::decisionFunction()` checks each of six input values against an expected range and adds a "violation" for each one out of bounds:

| Index | Expected range | Likely represents |
|---|---|---|
| 0 | 0–22 | Voltage |
| 1 | 0–3 | Current |
| 3 | 0–100 | Battery SOC (%) |
| 4 | 1000–65535 | Frequency-related signal |
| 5 | −40 to 80 | Temperature (°C) |

It also checks that measured power (index 2) is within 25% of `voltage × current` — catching sensor drift or wiring faults that individual range checks would miss. The score is `0.5 - violations/6`; anything ≥ 0 is classified normal (`+1`), anything below is anomalous (`-1`).

> Worth double-checking before you present this: index 4's range (1000–65535) doesn't look like a typical scaled representation of a 50 Hz mains signal — if that's meant to track the clock generator's frequency, it's worth confirming the units/scaling actually line up before relying on it live.

---

## 4. Firmware Logic (Simulation 1 — main build)

1. **`readElectricalSensors()`** — reads all six analog channels, converts raw ADC counts to engineering units (V, A, W, %, W/m²), and computes total available power as `PV string 1 + PV string 2 + battery contribution` (a fix was needed here — string 2's power used to be computed only for the ML model's input and silently dropped from the total everywhere else).
2. **`readEnvironmentalSensors()`** — DHT22 + both DS18B20 probes, with safe fallback values if a sensor read fails.
3. **`runRuleChecks()`** — evaluates six deterministic fault conditions (PV overvoltage/overcurrent, PV under-generation despite daylight, battery over-temperature, battery undervoltage, low SOC). Each fault uses a small hysteresis state machine (`FaultState`) requiring 5 consecutive bad readings before it fires, plus a 60-second cooldown before it can re-alert.
4. **`runAnomalyDetection()`** — feeds the six ML input features to `predict_fault()`, applies the same hysteresis/cooldown logic to the "not normal" prediction, and updates the dashboard status string with the predicted fault name and confidence.
5. **`manageLoadShedding()`** — compares available power to total required load and sheds non-critical → essential loads as supply drops, restoring them in the same order as supply recovers.
6. **Cloud upload** — `live.json` is overwritten (PUT) every 5 seconds; `history.json` gets a new entry (POST) every 10 seconds; `alerts.json` gets a new entry whenever any fault crosses its hysteresis threshold.

---

## 5. Frontend Dashboard

Built with **React, TypeScript, and Tailwind CSS**, reading from the same Firebase Realtime Database the firmware writes to. It's the single pane of glass for both simulations — live readings, historical trends, active alerts, and load-shedding state, all updating from the same `live` / `history` / `alerts` JSON structure the firmware produces.

---

## 6. Data Flow Summary

Both simulations independently push structured JSON to the backend — one over Firebase's REST-style HTTPS PUT/POST calls, matching the payload shape the dashboard expects. This means the frontend doesn't need to know which simulation (or which fault-detection method) produced a given reading — it just renders whatever lands in `live.json` / `history.json` / `alerts.json`.

---

## 7. Getting It Running

1. **Hardware simulation** — open either Wokwi diagram (`diagram.json` for the full pipeline, or the isolation-forest variant) in the Wokwi simulator or VS Code Wokwi extension.
2. **Firmware** — flash the corresponding `.ino` via Arduino IDE, with the ESP32 board package and the `DHT`, `OneWire`, and `DallasTemperature` libraries installed. Update `WIFI_SSID` / `WIFI_PASS` and `FIREBASE_BASE` for your own network and Firebase project (the current build uses `Wokwi-GUEST`, which only works inside the simulator).
3. **Firebase** — create a Realtime Database instance and confirm the rules allow read/write for `/microgrid/*` (or lock it down with proper auth before using this outside a demo).
4. **Frontend** — install dependencies and run the dev server; point its Firebase config at the same project the firmware writes to.

---

## 8. Known Limitations / Things Worth Fixing Before This Goes Beyond a Demo

- `client.setInsecure()` skips TLS certificate verification — fine for a Wokwi demo, but Firebase's root CA should be pinned before this touches real hardware or real data.
- With the current `PV_VOLTAGE_MAX` / `PV_CURRENT_MAX` (400 V / 10 A), simulated available power can reach ~4000 W per string against a ~400 W max load — you'll need to turn the potentiometers down to actually see load-shedding trigger in a demo.
- Alerts currently only reach a Firebase log and the dashboard — there's no external phone/SMS/push notification path in this build, despite the architecture being described that way in places.
- The isolation-forest-style scorer isn't a trained model — it's a fixed rule set wearing the isolation-forest interface. Worth being upfront about that distinction if this is presented as "two ML models."

---

## 9. Possible Next Steps

- Replace the hand-rolled isolation-forest stand-in with an actually trained isolation forest (or another lightweight unsupervised model) for a fairer comparison against the MLP.
- Add an external notification channel (SMS/push) for sustained faults instead of dashboard-only alerts.
- Pin Firebase's TLS certificate instead of using `setInsecure()`.
- Extend the fault classifier with wind-generation fault classes if a wind input is added later.

## 9. Contributors 

  Lakshya -: https://github.com/LAKSHYA-2005   
  hardware ARCHETECTURE , Design And Development 

  Aditya -: https://github.com/Adityaupadhyay013
  ML models Training , dataset collection 
