/*
 * Solar PV Anomaly Detection - ESP32 / Wokwi
 *
 * Model:
 *   sklearn IsolationForest exported to solar_PV_anomaly_detector.h
 *
 * Model input order MUST remain:
 *   0 voltage_v
 *   1 current_a
 *   2 power_w
 *   3 panel_temperature_c
 *   4 light_lux
 *   5 ambient_temperature_c
 *
 * Hardware:
 *   POT1  -> GPIO34 : PV voltage simulation
 *   POT2  -> GPIO35 : PV current simulation
 *   DS18B20 -> GPIO32: panel temperature
 *   LDR -> GPIO33    : light_lux proxy
 *   DHT22 -> GPIO4   : ambient temperature + humidity
 *   Green LED -> GPIO25 : NORMAL
 *   Red LED -> GPIO26   : ANOMALY
 *   Buzzer -> GPIO27    : ANOMALY alarm
 *
 * Put solar_PV_anomaly_detector.h in the same folder.
 */

#include <Arduino.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <WiFi.h>
#include <HTTPClient.h>

#include "solar_PV_anomaly_detector.h"

// ---------------- Pins ----------------
#define PV_VOLTAGE_PIN 34
#define PV_CURRENT_PIN 35
#define PANEL_TEMP_PIN 32
#define LIGHT_PIN      33
#define DHT_PIN        4
#define DHT_TYPE       DHT22

#define NORMAL_LED_PIN 25
#define ANOMALY_LED_PIN 26
#define BUZZER_PIN 27

// ---------------- Simulation ranges ----------------
// These are deliberately close to the uploaded training dataset.
// Do NOT use 230 V / 5 A here: that is outside the dataset range.
const float SIM_VOLTAGE_MAX = 22.0f;
const float SIM_CURRENT_MAX = 3.0f;

// Uploaded dataset light_lux range is approximately 1,001 to 65,535 lux.
const float SIM_LUX_MIN = 1000.0f;
const float SIM_LUX_MAX = 65535.0f;

// ---------------- Sensors ----------------
DHT dht(DHT_PIN, DHT_TYPE);
OneWire oneWire(PANEL_TEMP_PIN);
DallasTemperature ds18b20(&oneWire);

// ---------------- Firebase ----------------
// Set USE_FIREBASE to 0 if you only want local Wokwi testing.
#define USE_FIREBASE 1

const char* WIFI_SSID = "Wokwi-GUEST";
const char* WIFI_PASS = "";

const char* FIREBASE_URL =
  "https://microgrid-monitoring-f49e3-default-rtdb.asia-southeast1.firebasedatabase.app/"
  "microgrid/anomalySimulation/live.json";

unsigned long lastUpload = 0;
const unsigned long UPLOAD_INTERVAL_MS = 5000;

// ============================================================
// Isolation Forest inference
// Reconstructs sklearn IsolationForest from the generated header.
// ============================================================

static float average_path_length(int n) {
  if (n <= 1) return 0.0f;
  if (n == 2) return 1.0f;

  // c(n) = 2*H(n-1) - 2*(n-1)/n
  float harmonic = 0.0f;
  for (int i = 1; i <= n - 1; ++i) {
    harmonic += 1.0f / (float)i;
  }
  return 2.0f * harmonic - 2.0f * ((float)(n - 1) / (float)n);
}

static float tree_path_length(
    const float x[IF_NUM_FEATURES],
    const int16_t* feature,
    const float* threshold,
    const int32_t* left,
    const int32_t* right,
    const int32_t* samples,
    int nodes)
{
  int node = 0;
  int depth = 0;

  while (node >= 0 && node < nodes) {
    int16_t f = feature[node];

    // -2 is used by the exporter for leaf nodes.
    if (f < 0) {
      return (float)depth + average_path_length(samples[node]);
    }

    if (x[f] <= threshold[node]) {
      node = left[node];
    } else {
      node = right[node];
    }

    depth++;

    // Safety guard against malformed/corrupt model data.
    if (depth > 1024) {
      return (float)depth;
    }
  }

  return (float)depth;
}

static float tree_path_length_by_id(int tree, const float x[IF_NUM_FEATURES]) {
  switch (tree) {
    case 0:
      return tree_path_length(x, IF_TREE_0_FEATURE, IF_TREE_0_THRESHOLD,
                              IF_TREE_0_LEFT, IF_TREE_0_RIGHT,
                              IF_TREE_0_SAMPLES, IF_TREE_0_NODES);
    case 1:
      return tree_path_length(x, IF_TREE_1_FEATURE, IF_TREE_1_THRESHOLD,
                              IF_TREE_1_LEFT, IF_TREE_1_RIGHT,
                              IF_TREE_1_SAMPLES, IF_TREE_1_NODES);
    case 2:
      return tree_path_length(x, IF_TREE_2_FEATURE, IF_TREE_2_THRESHOLD,
                              IF_TREE_2_LEFT, IF_TREE_2_RIGHT,
                              IF_TREE_2_SAMPLES, IF_TREE_2_NODES);
    case 3:
      return tree_path_length(x, IF_TREE_3_FEATURE, IF_TREE_3_THRESHOLD,
                              IF_TREE_3_LEFT, IF_TREE_3_RIGHT,
                              IF_TREE_3_SAMPLES, IF_TREE_3_NODES);
    case 4:
      return tree_path_length(x, IF_TREE_4_FEATURE, IF_TREE_4_THRESHOLD,
                              IF_TREE_4_LEFT, IF_TREE_4_RIGHT,
                              IF_TREE_4_SAMPLES, IF_TREE_4_NODES);
    case 5:
      return tree_path_length(x, IF_TREE_5_FEATURE, IF_TREE_5_THRESHOLD,
                              IF_TREE_5_LEFT, IF_TREE_5_RIGHT,
                              IF_TREE_5_SAMPLES, IF_TREE_5_NODES);
    case 6:
      return tree_path_length(x, IF_TREE_6_FEATURE, IF_TREE_6_THRESHOLD,
                              IF_TREE_6_LEFT, IF_TREE_6_RIGHT,
                              IF_TREE_6_SAMPLES, IF_TREE_6_NODES);
    case 7:
      return tree_path_length(x, IF_TREE_7_FEATURE, IF_TREE_7_THRESHOLD,
                              IF_TREE_7_LEFT, IF_TREE_7_RIGHT,
                              IF_TREE_7_SAMPLES, IF_TREE_7_NODES);
    case 8:
      return tree_path_length(x, IF_TREE_8_FEATURE, IF_TREE_8_THRESHOLD,
                              IF_TREE_8_LEFT, IF_TREE_8_RIGHT,
                              IF_TREE_8_SAMPLES, IF_TREE_8_NODES);
    case 9:
      return tree_path_length(x, IF_TREE_9_FEATURE, IF_TREE_9_THRESHOLD,
                              IF_TREE_9_LEFT, IF_TREE_9_RIGHT,
                              IF_TREE_9_SAMPLES, IF_TREE_9_NODES);
    case 10:
      return tree_path_length(x, IF_TREE_10_FEATURE, IF_TREE_10_THRESHOLD,
                              IF_TREE_10_LEFT, IF_TREE_10_RIGHT,
                              IF_TREE_10_SAMPLES, IF_TREE_10_NODES);
    case 11:
      return tree_path_length(x, IF_TREE_11_FEATURE, IF_TREE_11_THRESHOLD,
                              IF_TREE_11_LEFT, IF_TREE_11_RIGHT,
                              IF_TREE_11_SAMPLES, IF_TREE_11_NODES);
    case 12:
      return tree_path_length(x, IF_TREE_12_FEATURE, IF_TREE_12_THRESHOLD,
                              IF_TREE_12_LEFT, IF_TREE_12_RIGHT,
                              IF_TREE_12_SAMPLES, IF_TREE_12_NODES);
    case 13:
      return tree_path_length(x, IF_TREE_13_FEATURE, IF_TREE_13_THRESHOLD,
                              IF_TREE_13_LEFT, IF_TREE_13_RIGHT,
                              IF_TREE_13_SAMPLES, IF_TREE_13_NODES);
    case 14:
      return tree_path_length(x, IF_TREE_14_FEATURE, IF_TREE_14_THRESHOLD,
                              IF_TREE_14_LEFT, IF_TREE_14_RIGHT,
                              IF_TREE_14_SAMPLES, IF_TREE_14_NODES);
    case 15:
      return tree_path_length(x, IF_TREE_15_FEATURE, IF_TREE_15_THRESHOLD,
                              IF_TREE_15_LEFT, IF_TREE_15_RIGHT,
                              IF_TREE_15_SAMPLES, IF_TREE_15_NODES);
    case 16:
      return tree_path_length(x, IF_TREE_16_FEATURE, IF_TREE_16_THRESHOLD,
                              IF_TREE_16_LEFT, IF_TREE_16_RIGHT,
                              IF_TREE_16_SAMPLES, IF_TREE_16_NODES);
    case 17:
      return tree_path_length(x, IF_TREE_17_FEATURE, IF_TREE_17_THRESHOLD,
                              IF_TREE_17_LEFT, IF_TREE_17_RIGHT,
                              IF_TREE_17_SAMPLES, IF_TREE_17_NODES);
    case 18:
      return tree_path_length(x, IF_TREE_18_FEATURE, IF_TREE_18_THRESHOLD,
                              IF_TREE_18_LEFT, IF_TREE_18_RIGHT,
                              IF_TREE_18_SAMPLES, IF_TREE_18_NODES);
    case 19:
      return tree_path_length(x, IF_TREE_19_FEATURE, IF_TREE_19_THRESHOLD,
                              IF_TREE_19_LEFT, IF_TREE_19_RIGHT,
                              IF_TREE_19_SAMPLES, IF_TREE_19_NODES);
    case 20:
      return tree_path_length(x, IF_TREE_20_FEATURE, IF_TREE_20_THRESHOLD,
                              IF_TREE_20_LEFT, IF_TREE_20_RIGHT,
                              IF_TREE_20_SAMPLES, IF_TREE_20_NODES);
    case 21:
      return tree_path_length(x, IF_TREE_21_FEATURE, IF_TREE_21_THRESHOLD,
                              IF_TREE_21_LEFT, IF_TREE_21_RIGHT,
                              IF_TREE_21_SAMPLES, IF_TREE_21_NODES);
    case 22:
      return tree_path_length(x, IF_TREE_22_FEATURE, IF_TREE_22_THRESHOLD,
                              IF_TREE_22_LEFT, IF_TREE_22_RIGHT,
                              IF_TREE_22_SAMPLES, IF_TREE_22_NODES);
    case 23:
      return tree_path_length(x, IF_TREE_23_FEATURE, IF_TREE_23_THRESHOLD,
                              IF_TREE_23_LEFT, IF_TREE_23_RIGHT,
                              IF_TREE_23_SAMPLES, IF_TREE_23_NODES);
    case 24:
      return tree_path_length(x, IF_TREE_24_FEATURE, IF_TREE_24_THRESHOLD,
                              IF_TREE_24_LEFT, IF_TREE_24_RIGHT,
                              IF_TREE_24_SAMPLES, IF_TREE_24_NODES);
    case 25:
      return tree_path_length(x, IF_TREE_25_FEATURE, IF_TREE_25_THRESHOLD,
                              IF_TREE_25_LEFT, IF_TREE_25_RIGHT,
                              IF_TREE_25_SAMPLES, IF_TREE_25_NODES);
    case 26:
      return tree_path_length(x, IF_TREE_26_FEATURE, IF_TREE_26_THRESHOLD,
                              IF_TREE_26_LEFT, IF_TREE_26_RIGHT,
                              IF_TREE_26_SAMPLES, IF_TREE_26_NODES);
    case 27:
      return tree_path_length(x, IF_TREE_27_FEATURE, IF_TREE_27_THRESHOLD,
                              IF_TREE_27_LEFT, IF_TREE_27_RIGHT,
                              IF_TREE_27_SAMPLES, IF_TREE_27_NODES);
    case 28:
      return tree_path_length(x, IF_TREE_28_FEATURE, IF_TREE_28_THRESHOLD,
                              IF_TREE_28_LEFT, IF_TREE_28_RIGHT,
                              IF_TREE_28_SAMPLES, IF_TREE_28_NODES);
    case 29:
      return tree_path_length(x, IF_TREE_29_FEATURE, IF_TREE_29_THRESHOLD,
                              IF_TREE_29_LEFT, IF_TREE_29_RIGHT,
                              IF_TREE_29_SAMPLES, IF_TREE_29_NODES);
    case 30:
      return tree_path_length(x, IF_TREE_30_FEATURE, IF_TREE_30_THRESHOLD,
                              IF_TREE_30_LEFT, IF_TREE_30_RIGHT,
                              IF_TREE_30_SAMPLES, IF_TREE_30_NODES);
    case 31:
      return tree_path_length(x, IF_TREE_31_FEATURE, IF_TREE_31_THRESHOLD,
                              IF_TREE_31_LEFT, IF_TREE_31_RIGHT,
                              IF_TREE_31_SAMPLES, IF_TREE_31_NODES);
    case 32:
      return tree_path_length(x, IF_TREE_32_FEATURE, IF_TREE_32_THRESHOLD,
                              IF_TREE_32_LEFT, IF_TREE_32_RIGHT,
                              IF_TREE_32_SAMPLES, IF_TREE_32_NODES);
    case 33:
      return tree_path_length(x, IF_TREE_33_FEATURE, IF_TREE_33_THRESHOLD,
                              IF_TREE_33_LEFT, IF_TREE_33_RIGHT,
                              IF_TREE_33_SAMPLES, IF_TREE_33_NODES);
    case 34:
      return tree_path_length(x, IF_TREE_34_FEATURE, IF_TREE_34_THRESHOLD,
                              IF_TREE_34_LEFT, IF_TREE_34_RIGHT,
                              IF_TREE_34_SAMPLES, IF_TREE_34_NODES);
    case 35:
      return tree_path_length(x, IF_TREE_35_FEATURE, IF_TREE_35_THRESHOLD,
                              IF_TREE_35_LEFT, IF_TREE_35_RIGHT,
                              IF_TREE_35_SAMPLES, IF_TREE_35_NODES);
    case 36:
      return tree_path_length(x, IF_TREE_36_FEATURE, IF_TREE_36_THRESHOLD,
                              IF_TREE_36_LEFT, IF_TREE_36_RIGHT,
                              IF_TREE_36_SAMPLES, IF_TREE_36_NODES);
    case 37:
      return tree_path_length(x, IF_TREE_37_FEATURE, IF_TREE_37_THRESHOLD,
                              IF_TREE_37_LEFT, IF_TREE_37_RIGHT,
                              IF_TREE_37_SAMPLES, IF_TREE_37_NODES);
    case 38:
      return tree_path_length(x, IF_TREE_38_FEATURE, IF_TREE_38_THRESHOLD,
                              IF_TREE_38_LEFT, IF_TREE_38_RIGHT,
                              IF_TREE_38_SAMPLES, IF_TREE_38_NODES);
    case 39:
      return tree_path_length(x, IF_TREE_39_FEATURE, IF_TREE_39_THRESHOLD,
                              IF_TREE_39_LEFT, IF_TREE_39_RIGHT,
                              IF_TREE_39_SAMPLES, IF_TREE_39_NODES);
    case 40:
      return tree_path_length(x, IF_TREE_40_FEATURE, IF_TREE_40_THRESHOLD,
                              IF_TREE_40_LEFT, IF_TREE_40_RIGHT,
                              IF_TREE_40_SAMPLES, IF_TREE_40_NODES);
    case 41:
      return tree_path_length(x, IF_TREE_41_FEATURE, IF_TREE_41_THRESHOLD,
                              IF_TREE_41_LEFT, IF_TREE_41_RIGHT,
                              IF_TREE_41_SAMPLES, IF_TREE_41_NODES);
    case 42:
      return tree_path_length(x, IF_TREE_42_FEATURE, IF_TREE_42_THRESHOLD,
                              IF_TREE_42_LEFT, IF_TREE_42_RIGHT,
                              IF_TREE_42_SAMPLES, IF_TREE_42_NODES);
    case 43:
      return tree_path_length(x, IF_TREE_43_FEATURE, IF_TREE_43_THRESHOLD,
                              IF_TREE_43_LEFT, IF_TREE_43_RIGHT,
                              IF_TREE_43_SAMPLES, IF_TREE_43_NODES);
    case 44:
      return tree_path_length(x, IF_TREE_44_FEATURE, IF_TREE_44_THRESHOLD,
                              IF_TREE_44_LEFT, IF_TREE_44_RIGHT,
                              IF_TREE_44_SAMPLES, IF_TREE_44_NODES);
    case 45:
      return tree_path_length(x, IF_TREE_45_FEATURE, IF_TREE_45_THRESHOLD,
                              IF_TREE_45_LEFT, IF_TREE_45_RIGHT,
                              IF_TREE_45_SAMPLES, IF_TREE_45_NODES);
    case 46:
      return tree_path_length(x, IF_TREE_46_FEATURE, IF_TREE_46_THRESHOLD,
                              IF_TREE_46_LEFT, IF_TREE_46_RIGHT,
                              IF_TREE_46_SAMPLES, IF_TREE_46_NODES);
    case 47:
      return tree_path_length(x, IF_TREE_47_FEATURE, IF_TREE_47_THRESHOLD,
                              IF_TREE_47_LEFT, IF_TREE_47_RIGHT,
                              IF_TREE_47_SAMPLES, IF_TREE_47_NODES);
    case 48:
      return tree_path_length(x, IF_TREE_48_FEATURE, IF_TREE_48_THRESHOLD,
                              IF_TREE_48_LEFT, IF_TREE_48_RIGHT,
                              IF_TREE_48_SAMPLES, IF_TREE_48_NODES);
    case 49:
      return tree_path_length(x, IF_TREE_49_FEATURE, IF_TREE_49_THRESHOLD,
                              IF_TREE_49_LEFT, IF_TREE_49_RIGHT,
                              IF_TREE_49_SAMPLES, IF_TREE_49_NODES);
    case 50:
      return tree_path_length(x, IF_TREE_50_FEATURE, IF_TREE_50_THRESHOLD,
                              IF_TREE_50_LEFT, IF_TREE_50_RIGHT,
                              IF_TREE_50_SAMPLES, IF_TREE_50_NODES);
    case 51:
      return tree_path_length(x, IF_TREE_51_FEATURE, IF_TREE_51_THRESHOLD,
                              IF_TREE_51_LEFT, IF_TREE_51_RIGHT,
                              IF_TREE_51_SAMPLES, IF_TREE_51_NODES);
    case 52:
      return tree_path_length(x, IF_TREE_52_FEATURE, IF_TREE_52_THRESHOLD,
                              IF_TREE_52_LEFT, IF_TREE_52_RIGHT,
                              IF_TREE_52_SAMPLES, IF_TREE_52_NODES);
    case 53:
      return tree_path_length(x, IF_TREE_53_FEATURE, IF_TREE_53_THRESHOLD,
                              IF_TREE_53_LEFT, IF_TREE_53_RIGHT,
                              IF_TREE_53_SAMPLES, IF_TREE_53_NODES);
    case 54:
      return tree_path_length(x, IF_TREE_54_FEATURE, IF_TREE_54_THRESHOLD,
                              IF_TREE_54_LEFT, IF_TREE_54_RIGHT,
                              IF_TREE_54_SAMPLES, IF_TREE_54_NODES);
    case 55:
      return tree_path_length(x, IF_TREE_55_FEATURE, IF_TREE_55_THRESHOLD,
                              IF_TREE_55_LEFT, IF_TREE_55_RIGHT,
                              IF_TREE_55_SAMPLES, IF_TREE_55_NODES);
    case 56:
      return tree_path_length(x, IF_TREE_56_FEATURE, IF_TREE_56_THRESHOLD,
                              IF_TREE_56_LEFT, IF_TREE_56_RIGHT,
                              IF_TREE_56_SAMPLES, IF_TREE_56_NODES);
    case 57:
      return tree_path_length(x, IF_TREE_57_FEATURE, IF_TREE_57_THRESHOLD,
                              IF_TREE_57_LEFT, IF_TREE_57_RIGHT,
                              IF_TREE_57_SAMPLES, IF_TREE_57_NODES);
    case 58:
      return tree_path_length(x, IF_TREE_58_FEATURE, IF_TREE_58_THRESHOLD,
                              IF_TREE_58_LEFT, IF_TREE_58_RIGHT,
                              IF_TREE_58_SAMPLES, IF_TREE_58_NODES);
    case 59:
      return tree_path_length(x, IF_TREE_59_FEATURE, IF_TREE_59_THRESHOLD,
                              IF_TREE_59_LEFT, IF_TREE_59_RIGHT,
                              IF_TREE_59_SAMPLES, IF_TREE_59_NODES);
    case 60:
      return tree_path_length(x, IF_TREE_60_FEATURE, IF_TREE_60_THRESHOLD,
                              IF_TREE_60_LEFT, IF_TREE_60_RIGHT,
                              IF_TREE_60_SAMPLES, IF_TREE_60_NODES);
    case 61:
      return tree_path_length(x, IF_TREE_61_FEATURE, IF_TREE_61_THRESHOLD,
                              IF_TREE_61_LEFT, IF_TREE_61_RIGHT,
                              IF_TREE_61_SAMPLES, IF_TREE_61_NODES);
    case 62:
      return tree_path_length(x, IF_TREE_62_FEATURE, IF_TREE_62_THRESHOLD,
                              IF_TREE_62_LEFT, IF_TREE_62_RIGHT,
                              IF_TREE_62_SAMPLES, IF_TREE_62_NODES);
    case 63:
      return tree_path_length(x, IF_TREE_63_FEATURE, IF_TREE_63_THRESHOLD,
                              IF_TREE_63_LEFT, IF_TREE_63_RIGHT,
                              IF_TREE_63_SAMPLES, IF_TREE_63_NODES);
    case 64:
      return tree_path_length(x, IF_TREE_64_FEATURE, IF_TREE_64_THRESHOLD,
                              IF_TREE_64_LEFT, IF_TREE_64_RIGHT,
                              IF_TREE_64_SAMPLES, IF_TREE_64_NODES);
    case 65:
      return tree_path_length(x, IF_TREE_65_FEATURE, IF_TREE_65_THRESHOLD,
                              IF_TREE_65_LEFT, IF_TREE_65_RIGHT,
                              IF_TREE_65_SAMPLES, IF_TREE_65_NODES);
    case 66:
      return tree_path_length(x, IF_TREE_66_FEATURE, IF_TREE_66_THRESHOLD,
                              IF_TREE_66_LEFT, IF_TREE_66_RIGHT,
                              IF_TREE_66_SAMPLES, IF_TREE_66_NODES);
    case 67:
      return tree_path_length(x, IF_TREE_67_FEATURE, IF_TREE_67_THRESHOLD,
                              IF_TREE_67_LEFT, IF_TREE_67_RIGHT,
                              IF_TREE_67_SAMPLES, IF_TREE_67_NODES);
    case 68:
      return tree_path_length(x, IF_TREE_68_FEATURE, IF_TREE_68_THRESHOLD,
                              IF_TREE_68_LEFT, IF_TREE_68_RIGHT,
                              IF_TREE_68_SAMPLES, IF_TREE_68_NODES);
    case 69:
      return tree_path_length(x, IF_TREE_69_FEATURE, IF_TREE_69_THRESHOLD,
                              IF_TREE_69_LEFT, IF_TREE_69_RIGHT,
                              IF_TREE_69_SAMPLES, IF_TREE_69_NODES);
    case 70:
      return tree_path_length(x, IF_TREE_70_FEATURE, IF_TREE_70_THRESHOLD,
                              IF_TREE_70_LEFT, IF_TREE_70_RIGHT,
                              IF_TREE_70_SAMPLES, IF_TREE_70_NODES);
    case 71:
      return tree_path_length(x, IF_TREE_71_FEATURE, IF_TREE_71_THRESHOLD,
                              IF_TREE_71_LEFT, IF_TREE_71_RIGHT,
                              IF_TREE_71_SAMPLES, IF_TREE_71_NODES);
    case 72:
      return tree_path_length(x, IF_TREE_72_FEATURE, IF_TREE_72_THRESHOLD,
                              IF_TREE_72_LEFT, IF_TREE_72_RIGHT,
                              IF_TREE_72_SAMPLES, IF_TREE_72_NODES);
    case 73:
      return tree_path_length(x, IF_TREE_73_FEATURE, IF_TREE_73_THRESHOLD,
                              IF_TREE_73_LEFT, IF_TREE_73_RIGHT,
                              IF_TREE_73_SAMPLES, IF_TREE_73_NODES);
    case 74:
      return tree_path_length(x, IF_TREE_74_FEATURE, IF_TREE_74_THRESHOLD,
                              IF_TREE_74_LEFT, IF_TREE_74_RIGHT,
                              IF_TREE_74_SAMPLES, IF_TREE_74_NODES);
    case 75:
      return tree_path_length(x, IF_TREE_75_FEATURE, IF_TREE_75_THRESHOLD,
                              IF_TREE_75_LEFT, IF_TREE_75_RIGHT,
                              IF_TREE_75_SAMPLES, IF_TREE_75_NODES);
    case 76:
      return tree_path_length(x, IF_TREE_76_FEATURE, IF_TREE_76_THRESHOLD,
                              IF_TREE_76_LEFT, IF_TREE_76_RIGHT,
                              IF_TREE_76_SAMPLES, IF_TREE_76_NODES);
    case 77:
      return tree_path_length(x, IF_TREE_77_FEATURE, IF_TREE_77_THRESHOLD,
                              IF_TREE_77_LEFT, IF_TREE_77_RIGHT,
                              IF_TREE_77_SAMPLES, IF_TREE_77_NODES);
    case 78:
      return tree_path_length(x, IF_TREE_78_FEATURE, IF_TREE_78_THRESHOLD,
                              IF_TREE_78_LEFT, IF_TREE_78_RIGHT,
                              IF_TREE_78_SAMPLES, IF_TREE_78_NODES);
    case 79:
      return tree_path_length(x, IF_TREE_79_FEATURE, IF_TREE_79_THRESHOLD,
                              IF_TREE_79_LEFT, IF_TREE_79_RIGHT,
                              IF_TREE_79_SAMPLES, IF_TREE_79_NODES);
    case 80:
      return tree_path_length(x, IF_TREE_80_FEATURE, IF_TREE_80_THRESHOLD,
                              IF_TREE_80_LEFT, IF_TREE_80_RIGHT,
                              IF_TREE_80_SAMPLES, IF_TREE_80_NODES);
    case 81:
      return tree_path_length(x, IF_TREE_81_FEATURE, IF_TREE_81_THRESHOLD,
                              IF_TREE_81_LEFT, IF_TREE_81_RIGHT,
                              IF_TREE_81_SAMPLES, IF_TREE_81_NODES);
    case 82:
      return tree_path_length(x, IF_TREE_82_FEATURE, IF_TREE_82_THRESHOLD,
                              IF_TREE_82_LEFT, IF_TREE_82_RIGHT,
                              IF_TREE_82_SAMPLES, IF_TREE_82_NODES);
    case 83:
      return tree_path_length(x, IF_TREE_83_FEATURE, IF_TREE_83_THRESHOLD,
                              IF_TREE_83_LEFT, IF_TREE_83_RIGHT,
                              IF_TREE_83_SAMPLES, IF_TREE_83_NODES);
    case 84:
      return tree_path_length(x, IF_TREE_84_FEATURE, IF_TREE_84_THRESHOLD,
                              IF_TREE_84_LEFT, IF_TREE_84_RIGHT,
                              IF_TREE_84_SAMPLES, IF_TREE_84_NODES);
    case 85:
      return tree_path_length(x, IF_TREE_85_FEATURE, IF_TREE_85_THRESHOLD,
                              IF_TREE_85_LEFT, IF_TREE_85_RIGHT,
                              IF_TREE_85_SAMPLES, IF_TREE_85_NODES);
    case 86:
      return tree_path_length(x, IF_TREE_86_FEATURE, IF_TREE_86_THRESHOLD,
                              IF_TREE_86_LEFT, IF_TREE_86_RIGHT,
                              IF_TREE_86_SAMPLES, IF_TREE_86_NODES);
    case 87:
      return tree_path_length(x, IF_TREE_87_FEATURE, IF_TREE_87_THRESHOLD,
                              IF_TREE_87_LEFT, IF_TREE_87_RIGHT,
                              IF_TREE_87_SAMPLES, IF_TREE_87_NODES);
    case 88:
      return tree_path_length(x, IF_TREE_88_FEATURE, IF_TREE_88_THRESHOLD,
                              IF_TREE_88_LEFT, IF_TREE_88_RIGHT,
                              IF_TREE_88_SAMPLES, IF_TREE_88_NODES);
    case 89:
      return tree_path_length(x, IF_TREE_89_FEATURE, IF_TREE_89_THRESHOLD,
                              IF_TREE_89_LEFT, IF_TREE_89_RIGHT,
                              IF_TREE_89_SAMPLES, IF_TREE_89_NODES);
    case 90:
      return tree_path_length(x, IF_TREE_90_FEATURE, IF_TREE_90_THRESHOLD,
                              IF_TREE_90_LEFT, IF_TREE_90_RIGHT,
                              IF_TREE_90_SAMPLES, IF_TREE_90_NODES);
    case 91:
      return tree_path_length(x, IF_TREE_91_FEATURE, IF_TREE_91_THRESHOLD,
                              IF_TREE_91_LEFT, IF_TREE_91_RIGHT,
                              IF_TREE_91_SAMPLES, IF_TREE_91_NODES);
    case 92:
      return tree_path_length(x, IF_TREE_92_FEATURE, IF_TREE_92_THRESHOLD,
                              IF_TREE_92_LEFT, IF_TREE_92_RIGHT,
                              IF_TREE_92_SAMPLES, IF_TREE_92_NODES);
    case 93:
      return tree_path_length(x, IF_TREE_93_FEATURE, IF_TREE_93_THRESHOLD,
                              IF_TREE_93_LEFT, IF_TREE_93_RIGHT,
                              IF_TREE_93_SAMPLES, IF_TREE_93_NODES);
    case 94:
      return tree_path_length(x, IF_TREE_94_FEATURE, IF_TREE_94_THRESHOLD,
                              IF_TREE_94_LEFT, IF_TREE_94_RIGHT,
                              IF_TREE_94_SAMPLES, IF_TREE_94_NODES);
    case 95:
      return tree_path_length(x, IF_TREE_95_FEATURE, IF_TREE_95_THRESHOLD,
                              IF_TREE_95_LEFT, IF_TREE_95_RIGHT,
                              IF_TREE_95_SAMPLES, IF_TREE_95_NODES);
    case 96:
      return tree_path_length(x, IF_TREE_96_FEATURE, IF_TREE_96_THRESHOLD,
                              IF_TREE_96_LEFT, IF_TREE_96_RIGHT,
                              IF_TREE_96_SAMPLES, IF_TREE_96_NODES);
    case 97:
      return tree_path_length(x, IF_TREE_97_FEATURE, IF_TREE_97_THRESHOLD,
                              IF_TREE_97_LEFT, IF_TREE_97_RIGHT,
                              IF_TREE_97_SAMPLES, IF_TREE_97_NODES);
    case 98:
      return tree_path_length(x, IF_TREE_98_FEATURE, IF_TREE_98_THRESHOLD,
                              IF_TREE_98_LEFT, IF_TREE_98_RIGHT,
                              IF_TREE_98_SAMPLES, IF_TREE_98_NODES);
    case 99:
      return tree_path_length(x, IF_TREE_99_FEATURE, IF_TREE_99_THRESHOLD,
                              IF_TREE_99_LEFT, IF_TREE_99_RIGHT,
                              IF_TREE_99_SAMPLES, IF_TREE_99_NODES);
    default:
      return 0.0f;
  }
}

/*
 * Returns the sklearn-style decision_function value:
 *
 *   decision = score_samples - offset
 *
 * Negative => anomaly
 * Zero/positive => normal
 */
float isolation_score(const float x[IF_NUM_FEATURES]) {
  float total_path = 0.0f;

  for (int t = 0; t < IF_NUM_TREES; ++t) {
    total_path += tree_path_length_by_id(t, x);
  }

  float average_path = total_path / (float)IF_NUM_TREES;

  // sklearn IsolationForest score_samples is:
  //   -2^(-average_path / c(max_samples))
  //
  // Then decision_function:
  //   score_samples - offset
  float cmax = average_path_length(IF_MAX_SAMPLES);
  float score_samples = -powf(2.0f, -average_path / cmax);

  return score_samples - IF_OFFSET;
}

int predict_anomaly(const float x[IF_NUM_FEATURES]) {
  return (isolation_score(x) >= 0.0f) ? 1 : -1;
}

// ============================================================
// Sensor conversion
// ============================================================

float readVoltage() {
  int raw = analogRead(PV_VOLTAGE_PIN);
  return ((float)raw / 4095.0f) * SIM_VOLTAGE_MAX;
}

float readCurrent() {
  int raw = analogRead(PV_CURRENT_PIN);
  return ((float)raw / 4095.0f) * SIM_CURRENT_MAX;
}

float readLightLux() {
  int raw = analogRead(LIGHT_PIN);

  // Wokwi's photoresistor AO is an electrical simulation, not a calibrated
  // lux meter. For this prototype we map the ADC range to the same numerical
  // lux domain used by the training dataset.
  return SIM_LUX_MIN +
         ((float)raw / 4095.0f) * (SIM_LUX_MAX - SIM_LUX_MIN);
}

float readPanelTemperature() {
  ds18b20.requestTemperatures();
  float t = ds18b20.getTempCByIndex(0);

  if (t == DEVICE_DISCONNECTED_C || isnan(t)) {
    return 30.0f;
  }

  return t;
}

float readAmbientTemperature() {
  float t = dht.readTemperature();

  if (isnan(t)) {
    return 29.0f;
  }

  return t;
}

// ============================================================
// Outputs
// ============================================================

void showResult(int result) {
  if (result == -1) {
    digitalWrite(NORMAL_LED_PIN, LOW);
    digitalWrite(ANOMALY_LED_PIN, HIGH);

    // Short alarm pattern
    tone(BUZZER_PIN, 2000, 250);

    Serial.println(">>> ANOMALY DETECTED <<<");
  } else {
    digitalWrite(NORMAL_LED_PIN, HIGH);
    digitalWrite(ANOMALY_LED_PIN, LOW);
    noTone(BUZZER_PIN);

    Serial.println(">>> NORMAL OPERATION <<<");
  }
}

// ============================================================
// Firebase
// ============================================================

void connectWiFi() {
#if USE_FIREBASE
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  Serial.print("Connecting to WiFi");

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - start < 10000) {
    delay(300);
    Serial.print(".");
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi unavailable - continuing offline");
  }
#endif
}

void uploadFirebase(
    float voltage,
    float current,
    float power,
    float panelTemp,
    float lightLux,
    float ambientTemp,
    float score,
    int result)
{
#if USE_FIREBASE
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;

  http.begin(FIREBASE_URL);
  http.addHeader("Content-Type", "application/json");

  String status = (result == -1) ? "ANOMALY" : "NORMAL";

  String payload = "{";
  payload += "\"voltage_v\":" + String(voltage, 3) + ",";
  payload += "\"current_a\":" + String(current, 3) + ",";
  payload += "\"power_w\":" + String(power, 3) + ",";
  payload += "\"panel_temperature_c\":" + String(panelTemp, 2) + ",";
  payload += "\"light_lux\":" + String(lightLux, 1) + ",";
  payload += "\"ambient_temperature_c\":" + String(ambientTemp, 2) + ",";
  payload += "\"anomaly_score\":" + String(score, 6) + ",";
  payload += "\"prediction\":" + String(result) + ",";
  payload += "\"status\":\"" + status + "\",";
  payload += "\"uptime_ms\":" + String(millis());
  payload += "}";

  int code = http.PUT(payload);

  Serial.print("Firebase HTTP: ");
  Serial.println(code);

  http.end();
#endif
}

// ============================================================
// Setup
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("==============================================");
  Serial.println(" Solar PV Anomaly Detection - ESP32 / Wokwi");
  Serial.println("==============================================");

  analogReadResolution(12);

  pinMode(PV_VOLTAGE_PIN, INPUT);
  pinMode(PV_CURRENT_PIN, INPUT);
  pinMode(LIGHT_PIN, INPUT);

  pinMode(NORMAL_LED_PIN, OUTPUT);
  pinMode(ANOMALY_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(NORMAL_LED_PIN, LOW);
  digitalWrite(ANOMALY_LED_PIN, LOW);
  noTone(BUZZER_PIN);

  dht.begin();
  ds18b20.begin();

  Serial.print("DS18B20 devices: ");
  Serial.println(ds18b20.getDeviceCount());

  Serial.print("Isolation Forest trees: ");
  Serial.println(IF_NUM_TREES);

  Serial.print("Model features: ");
  Serial.println(IF_NUM_FEATURES);

  connectWiFi();
}

// ============================================================
// Main loop
// ============================================================

void loop() {

  // -------- Read sensors --------
  float voltage = readVoltage();
  float current = readCurrent();
  float power = voltage * current;

  float panelTemp = readPanelTemperature();
  float lightLux = readLightLux();
  float ambientTemp = readAmbientTemperature();

  // -------- EXACT MODEL INPUT ORDER --------
  float input[6] = {
    voltage,       // Input 0 -> voltage_v
    current,       // Input 1 -> current_a
    power,         // Input 2 -> power_w
    panelTemp,     // Input 3 -> panel_temperature_c
    lightLux,      // Input 4 -> light_lux
    ambientTemp    // Input 5 -> ambient_temperature_c
  };

  // -------- ML inference --------
  float score = isolation_score(input);
  int result = predict_anomaly(input);

  // -------- Serial dashboard --------
  Serial.println();
  Serial.println("----------------------------------------------");

  Serial.printf("Voltage       : %.3f V\n", voltage);
  Serial.printf("Current       : %.3f A\n", current);
  Serial.printf("Power         : %.3f W\n", power);
  Serial.printf("Panel Temp    : %.2f C\n", panelTemp);
  Serial.printf("Light         : %.1f lux\n", lightLux);
  Serial.printf("Ambient Temp  : %.2f C\n", ambientTemp);

  Serial.printf("ML Score      : %.6f\n", score);
  Serial.printf("Prediction    : %d\n", result);

  showResult(result);

  Serial.println("----------------------------------------------");

  // -------- Firebase --------
  if (millis() - lastUpload >= UPLOAD_INTERVAL_MS) {
    uploadFirebase(
      voltage,
      current,
      power,
      panelTemp,
      lightLux,
      ambientTemp,
      score,
      result
    );

    lastUpload = millis();
  }

  delay(1000);
}