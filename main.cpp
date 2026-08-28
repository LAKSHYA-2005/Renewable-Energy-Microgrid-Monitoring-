/*
 * Renewable Energy Monitoring System for Microgrids
 * GGSIPU2647 - Final firmware (rule-based checks + ML fault classifier)
 *
 * Pipeline:
 *   Potentiometers/DHT22/DS18B20/photoresistor
 *     -> engineering units -> MicrogridData
 *     -> rule-based fault checks (with hysteresis + rate limiting)
 *     -> ML fault classifier (predict_fault(), hand-exported 6->32->16->5 MLP)
 *     -> priority load shedding (critical/essential/non-critical relays)
 *     -> WiFi + Firebase upload (live + history)
 *     -> Firebase alert log on sustained faults (dashboard-only, no
 *        external phone-alert service in this build)
 *
 * ML MODEL: integrated in runAnomalyDetection() via fault_detector_model.h.
 * Feature order is idc1, idc2, vdc1, vdc2, irr, pvt - verified against the
 * training dataset's own column statistics. vdc2/idc2 use GPIO32/39 -
 * GPIO32 was freed by replacing the load-demand potentiometer with a
 * synthetic triangle-wave load profile (computeSyntheticLoad()), since this
 * board (standard 30-pin ESP32 DevKit) has only 6 ADC1 pins total
 * (32,33,34,35,36,39) and all 6 are now spoken for - GPIO37/38 do not exist
 * on this board and must not be used.
 *
 * FIX (this version): availablePower previously only summed string 1's
 * power + battery contribution - string 2's generation was computed for the
 * ML model's input but never added to the total or sent to the dashboard.
 * Added pvPower2, folded into availablePower, and included in the payload.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include "fault_detector_model.h"   // hand-exported 6->32->16->5 MLP, no TFLite Micro needed

// ---------------- Pin map (all ADC1 - safe to read while WiFi is active) ----------------
#define PV_VOLTAGE_PIN     34   // PV string 1 voltage (vdc1)  (0-400 V)
#define PV_CURRENT_PIN     35   // PV string 1 current (idc1)  (0-10 A)
#define PV_VOLTAGE2_PIN    32   // PV string 2 voltage (vdc2)  (0-400 V) - reuses the pin
                                 // that used to be LOAD_DEMAND_PIN; load demand is now
                                 // synthetic (see computeSyntheticLoad()) so this pin was free
#define BATTERY_SOC_PIN    33   // Battery SOC      (0-100 %)
#define LIGHT_PIN          36   // Photoresistor: irradiance proxy (VP / input-only ADC1)
#define PV_CURRENT2_PIN    39   // PV string 2 current (idc2)  (0-10 A) (VN / input-only ADC1)
// Confirmed against your board's actual pinout: GPIO32,33,34,35,36,39 are the ONLY ADC1
// pins broken out on this devkit. GPIO37/38 do not exist on this board at all - do not
// reference them anywhere in the diagram or code.

#define DHT_PIN            16
#define DHT_TYPE           DHT22

#define ONE_WIRE_BUS       4    // DS18B20 x2 share this bus (unique ROM IDs)

#define RELAY_CRITICAL     25
#define RELAY_ESSENTIAL    26
#define RELAY_NONCRITICAL  27

// ---------------- Thresholds ----------------
// PV_VOLTAGE_MAX/PV_CURRENT_MAX match the ML model's training dataset
// (dataset_elec_1_.mat: vdc1/vdc2 range ~0.4-364V, idc1/idc2 range ~0.04-9.65A).
const float PV_VOLTAGE_MAX        = 400.0f;  // dataset max ~364V, rounded up with headroom
const float PV_CURRENT_MAX        = 10.0f;   // dataset max ~9.65A, rounded up with headroom
const float PV_UNDERGEN_LIGHT_MIN = 40.0f;   // % - "there should be sunlight"
const float PV_UNDERGEN_POWER_MAX = 5.0f;    // W  - but almost no power out = fault (string 1 only)
const float BATTERY_TEMP_MAX      = 45.0f;
const float BATTERY_LOW_VOLTAGE   = 11.0f;
const float BATTERY_SOC_LOW       = 15.0f;

const float LOAD_CRITICAL_W    = 100.0f;
const float LOAD_ESSENTIAL_W   = 100.0f;
const float LOAD_NONCRITICAL_W = 200.0f;
// Note for the demo: with PV_VOLTAGE_MAX/PV_CURRENT_MAX now at 400V/10A,
// availablePower can reach ~4000W per string at full pot rotation - far
// above the ~400W max load. To actually demonstrate load shedding live,
// turn pot1/pot2 (and/or pot3/pot5) DOWN toward zero first.

// ---------------- Alert hysteresis / rate limiting ----------------
const uint8_t CONSECUTIVE_THRESHOLD = 5;        // ~5 loop iterations before an alert fires
const unsigned long ALERT_COOLDOWN_MS = 60000UL; // don't re-alert same fault within 60s

struct FaultState {
  uint8_t consecutiveCount = 0;
  unsigned long lastAlertMs = 0;
  bool everAlerted = false;
};

FaultState fsPvOverV, fsPvOverI, fsPvUnderGen, fsBattOverTemp, fsBattLowV, fsBattLowSOC;
FaultState fsMlFault; // hysteresis for the ML model's non-Normal predictions

bool shouldAlert(FaultState &fs, bool conditionNow) {
  if (!conditionNow) {
    fs.consecutiveCount = 0;
    return false;
  }
  if (fs.consecutiveCount < 255) fs.consecutiveCount++;
  bool cooldownOk = !fs.everAlerted || (millis() - fs.lastAlertMs > ALERT_COOLDOWN_MS);
  if (fs.consecutiveCount >= CONSECUTIVE_THRESHOLD && cooldownOk) {
    fs.lastAlertMs = millis();
    fs.everAlerted = true;
    return true;
  }
  return false;
}

// ---------------- WiFi / Firebase ----------------
const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

const char* FIREBASE_BASE = "https://microgrid-monitoring-f49e3-default-rtdb.asia-southeast1.firebasedatabase.app";

unsigned long lastLiveUpload = 0;
const unsigned long LIVE_UPLOAD_INTERVAL_MS = 5000;

unsigned long lastHistoryUpload = 0;
const unsigned long HISTORY_UPLOAD_INTERVAL_MS = 10000; // 1 reading/10s -> chart fills in fast for a demo

// ---------------- Sensor objects ----------------
DHT dht(DHT_PIN, DHT_TYPE);
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds18b20(&oneWire);
DeviceAddress batteryTempAddr, panelTempAddr;
bool dsAddressesFound = false;

// ---------------- Shared data structure ----------------
struct MicrogridData {
  float pvVoltage;
  float pvCurrent;
  float pvPower;

  float batteryVoltage;
  float batterySOC;
  float batteryTemperature;

  float panelTemperature;
  float ambientTemperature;
  float humidity;
  float lightLevel;

  float requiredPower;
  float availableBatteryPower;
  float availablePower;   // FIX: now string1 + string2 + battery, was string1-only

  float pvVoltage2;      // vdc2
  float pvCurrent2;      // idc2
  float pvPower2;        // NEW - was computed nowhere before, only fed to the ML model
  float irradianceWm2;   // irr, physical W/m^2 (0-1100), separate from lightLevel %
  int   mlFaultClass;    // last predicted class index (0-4)
  float mlFaultConfidence;

  bool criticalOn;
  bool essentialOn;
  bool nonCriticalOn;

  String status;
};

MicrogridData data;

// ---------------- HTTPS helpers ----------------
int httpsRequest(const String &url, const String &jsonPayload, const char *method) {
  if (WiFi.status() != WL_CONNECTED) return -1;

  WiFiClientSecure client;
  client.setInsecure(); // fine for a demo; pin Firebase's root CA for production
  HTTPClient http;
  if (!http.begin(client, url)) return -2;

  int code;
  if (strcmp(method, "PUT") == 0) {
    http.addHeader("Content-Type", "application/json");
    code = http.PUT(jsonPayload);
  } else if (strcmp(method, "POST") == 0) {
    http.addHeader("Content-Type", "application/json");
    code = http.POST(jsonPayload);
  } else {
    code = http.GET();
  }
  http.end();
  return code;
}

// ---------------- Alerting ----------------
void sendAlert(const String &message) {
  Serial.println("[ALERT] " + message);

  String alertsUrl = String(FIREBASE_BASE) + "/microgrid/alerts.json";
  String payload = "{\"message\":\"" + message + "\",\"uptimeMs\":" + String(millis()) + "}";
  httpsRequest(alertsUrl, payload, "POST");
}

// ---------------- Setup helpers ----------------
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected: " + WiFi.localIP().toString());
  } else {
    Serial.println("WiFi NOT connected - continuing offline");
  }
}

void setupRelays() {
  pinMode(RELAY_CRITICAL, OUTPUT);
  pinMode(RELAY_ESSENTIAL, OUTPUT);
  pinMode(RELAY_NONCRITICAL, OUTPUT);
  digitalWrite(RELAY_CRITICAL, HIGH);
  digitalWrite(RELAY_ESSENTIAL, HIGH);
  digitalWrite(RELAY_NONCRITICAL, HIGH);
  data.criticalOn = data.essentialOn = data.nonCriticalOn = true;
}

void printDeviceAddress(DeviceAddress addr) {
  for (uint8_t i = 0; i < 8; i++) {
    if (addr[i] < 16) Serial.print("0");
    Serial.print(addr[i], HEX);
    if (i < 7) Serial.print(":");
  }
}

void setupTempSensors() {
  ds18b20.begin();
  int count = ds18b20.getDeviceCount();
  Serial.printf("DS18B20 devices found: %d\n", count);
  if (count >= 2) {
    ds18b20.getAddress(batteryTempAddr, 0);
    ds18b20.getAddress(panelTempAddr, 1);
    dsAddressesFound = true;

    Serial.print("  index 0 (battery) address: ");
    printDeviceAddress(batteryTempAddr);
    Serial.println();
    Serial.print("  index 1 (panel)   address: ");
    printDeviceAddress(panelTempAddr);
    Serial.println();
  }
}

// ---------------- Synthetic load profile (replaces the old load-demand pot) ----------------
const unsigned long LOAD_CYCLE_PERIOD_MS = 60000UL; // one full ramp up+down per minute
float computeSyntheticLoad() {
  float phase = fmod((float)millis(), (float)LOAD_CYCLE_PERIOD_MS) / (float)LOAD_CYCLE_PERIOD_MS;
  float triangle = (phase < 0.5f) ? (phase * 2.0f) : (2.0f - phase * 2.0f); // 0 -> 1 -> 0
  return triangle * 500.0f; // 0-500 W, matches the old pot's range
}

// ---------------- Reading sensors ----------------
void readElectricalSensors() {
  int vRaw    = analogRead(PV_VOLTAGE_PIN);
  int iRaw    = analogRead(PV_CURRENT_PIN);
  int v2Raw   = analogRead(PV_VOLTAGE2_PIN);
  int i2Raw   = analogRead(PV_CURRENT2_PIN);
  int socRaw  = analogRead(BATTERY_SOC_PIN);
  int lightRaw= analogRead(LIGHT_PIN);

  data.pvVoltage = (vRaw / 4095.0f) * PV_VOLTAGE_MAX;
  data.pvCurrent = (iRaw / 4095.0f) * PV_CURRENT_MAX;
  data.pvPower   = data.pvVoltage * data.pvCurrent;

  data.pvVoltage2 = (v2Raw / 4095.0f) * PV_VOLTAGE_MAX;
  data.pvCurrent2 = (i2Raw / 4095.0f) * PV_CURRENT_MAX;
  data.pvPower2   = data.pvVoltage2 * data.pvCurrent2;   // FIX: this was never computed before

  data.irradianceWm2 = (lightRaw / 4095.0f) * 1100.0f;

  data.requiredPower = computeSyntheticLoad();

  data.batterySOC = (socRaw / 4095.0f) * 100.0f;
  data.batteryVoltage = BATTERY_LOW_VOLTAGE + (data.batterySOC / 100.0f) * 1.6f;

  data.lightLevel = (lightRaw / 4095.0f) * 100.0f;

  data.availableBatteryPower = (data.batterySOC > 20.0f) ? 150.0f : 0.0f;
  // FIX: was data.pvPower + data.availableBatteryPower - string 2 was silently dropped.
  data.availablePower = data.pvPower + data.pvPower2 + data.availableBatteryPower;
}

void readEnvironmentalSensors() {
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  data.ambientTemperature = isnan(t) ? 25.0f : t;
  data.humidity            = isnan(h) ? 50.0f : h;

  ds18b20.requestTemperatures();
  if (dsAddressesFound) {
    float bt = ds18b20.getTempC(batteryTempAddr);
    float pt = ds18b20.getTempC(panelTempAddr);
    data.batteryTemperature = (bt == DEVICE_DISCONNECTED_C) ? 30.0f : bt;
    data.panelTemperature   = (pt == DEVICE_DISCONNECTED_C) ? 35.0f : pt;
  } else {
    data.batteryTemperature = 30.0f;
    data.panelTemperature   = 35.0f;
  }
}

// ---------------- Rule-based fault detection (with hysteresis) ----------------
void runRuleChecks() {
  data.status = "NORMAL";

  bool pvOverVNow      = data.pvVoltage > PV_VOLTAGE_MAX;
  bool pvOverINow       = data.pvCurrent > PV_CURRENT_MAX;
  bool pvUnderGenNow    = (data.lightLevel > PV_UNDERGEN_LIGHT_MIN) && (data.pvPower < PV_UNDERGEN_POWER_MAX);
  bool battOverTempNow  = data.batteryTemperature > BATTERY_TEMP_MAX;
  bool battLowVNow      = data.batteryVoltage < BATTERY_LOW_VOLTAGE;
  bool battLowSOCNow    = data.batterySOC < BATTERY_SOC_LOW;

  if (pvOverVNow)      data.status = "FAULT: PV OVERVOLTAGE";
  if (pvOverINow)       data.status = "FAULT: PV OVERCURRENT";
  if (pvUnderGenNow)    data.status = "FAULT: PV UNDER-GENERATION";
  if (battOverTempNow)  data.status = "FAULT: BATTERY OVER TEMPERATURE";
  if (battLowVNow)      data.status = "FAULT: BATTERY UNDERVOLTAGE";
  if (battLowSOCNow)    data.status = "WARNING: LOW BATTERY SOC";

  if (shouldAlert(fsPvOverV, pvOverVNow))
    sendAlert("PV overvoltage: " + String(data.pvVoltage, 1) + " V");
  if (shouldAlert(fsPvOverI, pvOverINow))
    sendAlert("PV overcurrent: " + String(data.pvCurrent, 1) + " A");
  if (shouldAlert(fsPvUnderGen, pvUnderGenNow))
    sendAlert("PV under-generation despite light level " + String(data.lightLevel, 0) +
              "%: only " + String(data.pvPower, 1) + " W");
  if (shouldAlert(fsBattOverTemp, battOverTempNow))
    sendAlert("Battery over-temperature: " + String(data.batteryTemperature, 1) + " C");
  if (shouldAlert(fsBattLowV, battLowVNow))
    sendAlert("Battery undervoltage: " + String(data.batteryVoltage, 2) + " V");
  if (shouldAlert(fsBattLowSOC, battLowSOCNow))
    sendAlert("Battery SOC critically low: " + String(data.batterySOC, 1) + " %");
}

// ---------------- Load requirement / shedding ----------------
void manageLoadShedding() {
  float totalRequired = 0;
  if (data.criticalOn)    totalRequired += LOAD_CRITICAL_W;
  if (data.essentialOn)   totalRequired += LOAD_ESSENTIAL_W;
  if (data.nonCriticalOn) totalRequired += LOAD_NONCRITICAL_W;

  if (data.availablePower >= totalRequired) {
    if (!data.essentialOn && data.availablePower >= LOAD_CRITICAL_W + LOAD_ESSENTIAL_W) {
      data.essentialOn = true;
    }
    if (!data.nonCriticalOn &&
        data.availablePower >= LOAD_CRITICAL_W + LOAD_ESSENTIAL_W + LOAD_NONCRITICAL_W) {
      data.nonCriticalOn = true;
    }
  } else {
    if (data.nonCriticalOn) {
      data.nonCriticalOn = false;
      sendAlert("Supply deficit - shedding NON-CRITICAL load");
    } else if (data.essentialOn) {
      data.essentialOn = false;
      sendAlert("Supply deficit - shedding ESSENTIAL load");
    } else {
      sendAlert("CRITICAL: supply cannot meet even critical load demand");
    }
  }

  digitalWrite(RELAY_CRITICAL, data.criticalOn ? HIGH : LOW);
  digitalWrite(RELAY_ESSENTIAL, data.essentialOn ? HIGH : LOW);
  digitalWrite(RELAY_NONCRITICAL, data.nonCriticalOn ? HIGH : LOW);
}

// ---------------- ML fault classifier ----------------
void runAnomalyDetection() {
  float features[FAULT_MODEL_INPUT_SIZE] = {
    data.pvCurrent,       // idc1
    data.pvCurrent2,      // idc2
    data.pvVoltage,       // vdc1
    data.pvVoltage2,      // vdc2
    data.irradianceWm2,   // irr
    data.panelTemperature // pvt
  };

  float probs[FAULT_MODEL_NUM_CLASSES];
  int predicted = predict_fault(features, probs);

  data.mlFaultClass = predicted;
  data.mlFaultConfidence = probs[predicted];

  bool isFaultNow = (predicted != 0);
  if (shouldAlert(fsMlFault, isFaultNow)) {
    data.status = String("ML FAULT: ") + FAULT_CLASS_NAMES[predicted];
    sendAlert(String("ML model detected: ") + FAULT_CLASS_NAMES[predicted] +
              " (confidence " + String(data.mlFaultConfidence * 100.0f, 0) + "%)");
  }
}

// ---------------- Cloud upload ----------------
String buildReadingPayload() {
  String payload = "{";
  payload += "\"pvVoltage\":" + String(data.pvVoltage, 2) + ",";
  payload += "\"pvCurrent\":" + String(data.pvCurrent, 2) + ",";
  payload += "\"pvPower\":" + String(data.pvPower, 2) + ",";
  payload += "\"pvVoltage2\":" + String(data.pvVoltage2, 2) + ",";
  payload += "\"pvCurrent2\":" + String(data.pvCurrent2, 2) + ",";
  payload += "\"pvPower2\":" + String(data.pvPower2, 2) + ",";
  payload += "\"irradianceWm2\":" + String(data.irradianceWm2, 0) + ",";
  payload += "\"batteryVoltage\":" + String(data.batteryVoltage, 2) + ",";
  payload += "\"batterySOC\":" + String(data.batterySOC, 1) + ",";
  payload += "\"batteryTemperature\":" + String(data.batteryTemperature, 1) + ",";
  payload += "\"panelTemperature\":" + String(data.panelTemperature, 1) + ",";
  payload += "\"requiredPower\":" + String(data.requiredPower, 1) + ",";
  payload += "\"availablePower\":" + String(data.availablePower, 1) + ",";
  payload += "\"criticalOn\":" + String(data.criticalOn ? "true" : "false") + ",";
  payload += "\"essentialOn\":" + String(data.essentialOn ? "true" : "false") + ",";
  payload += "\"nonCriticalOn\":" + String(data.nonCriticalOn ? "true" : "false") + ",";
  payload += "\"status\":\"" + data.status + "\",";
  payload += "\"mlFaultClass\":\"" + String(FAULT_CLASS_NAMES[data.mlFaultClass]) + "\",";
  payload += "\"mlFaultConfidence\":" + String(data.mlFaultConfidence, 2) + ",";
  payload += "\"uptimeMs\":" + String(millis());
  payload += "}";
  return payload;
}

void uploadLive() {
  String url = String(FIREBASE_BASE) + "/microgrid/live.json";
  int code = httpsRequest(url, buildReadingPayload(), "PUT");
  Serial.printf("Firebase live upload HTTP code: %d\n", code);
}

void uploadHistory() {
  String url = String(FIREBASE_BASE) + "/microgrid/history.json";
  int code = httpsRequest(url, buildReadingPayload(), "POST");
  Serial.printf("Firebase history upload HTTP code: %d (uptime %lus)\n", code, millis() / 1000);
}

// ---------------- Serial dashboard ----------------
void printStatus() {
  Serial.println("==========================================");
  Serial.printf("PV str1   : %.2f V  %.2f A  %.2f W\n", data.pvVoltage, data.pvCurrent, data.pvPower);
  Serial.printf("PV str2   : %.2f V  %.2f A  %.2f W\n", data.pvVoltage2, data.pvCurrent2, data.pvPower2);
  Serial.printf("Total gen : %.2f W (str1+str2), + battery %.0f W available = %.2f W\n",
                data.pvPower + data.pvPower2, data.availableBatteryPower, data.availablePower);
  Serial.printf("Irradiance: %.0f W/m^2\n", data.irradianceWm2);
  Serial.printf("Battery   : SOC %.1f %%  %.2f V  Temp %.1f C\n",
                data.batterySOC, data.batteryVoltage, data.batteryTemperature);
  Serial.printf("Env       : Panel %.1f C  Ambient %.1f C  Humidity %.1f %%  Light %.1f %%\n",
                data.panelTemperature, data.ambientTemperature, data.humidity, data.lightLevel);
  Serial.printf("Power     : Required %.1f W  Available %.1f W\n",
                data.requiredPower, data.availablePower);
  Serial.printf("Loads     : Critical=%s Essential=%s NonCritical=%s\n",
                data.criticalOn ? "ON" : "OFF",
                data.essentialOn ? "ON" : "OFF",
                data.nonCriticalOn ? "ON" : "OFF");
  Serial.printf("ML        : %s (confidence %.0f%%)\n",
                FAULT_CLASS_NAMES[data.mlFaultClass], data.mlFaultConfidence * 100.0f);
  Serial.println("Status    : " + data.status);
  Serial.println("==========================================");
}

// ---------------- Arduino entry points ----------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("Renewable Microgrid Monitor - starting");

  pinMode(PV_VOLTAGE_PIN, INPUT);
  pinMode(PV_CURRENT_PIN, INPUT);
  pinMode(PV_VOLTAGE2_PIN, INPUT);
  pinMode(PV_CURRENT2_PIN, INPUT);
  pinMode(BATTERY_SOC_PIN, INPUT);
  pinMode(LIGHT_PIN, INPUT);

  dht.begin();
  setupTempSensors();
  setupRelays();
  connectWiFi();
}

void loop() {
  readElectricalSensors();
  readEnvironmentalSensors();
  runRuleChecks();
  manageLoadShedding();
  runAnomalyDetection();
  printStatus();

  unsigned long now = millis();
  if (now - lastLiveUpload > LIVE_UPLOAD_INTERVAL_MS) {
    uploadLive();
    lastLiveUpload = now;
  }
  if (now - lastHistoryUpload > HISTORY_UPLOAD_INTERVAL_MS) {
    uploadHistory();
    lastHistoryUpload = now;
  }

  delay(1000);
}