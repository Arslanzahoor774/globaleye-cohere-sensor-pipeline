"""
=============================================================================
GlobalEye Multi-Modal Sensor Data Bridge — Cohere LLM Integration Pipeline
=============================================================================

Author      : Muhammad Arslan Zahoor
Purpose     : Proof-of-concept pipeline demonstrating how raw multi-modal
              aerospace sensor data from the Saab GlobalEye AEW&C system
              can be converted into structured natural language and fed into
              Cohere's enterprise LLM for real-time operator decision support.

Context     : Built in response to the Saab–Cohere MOU (2025) which aims to
              integrate AI into GlobalEye's mission support, maintenance, and
              information processing workflows within secure on-premises
              environments.

Pipeline    : Sensor Data → Feature Engineering → Threat Scoring →
              Natural Language Conversion → Cohere LLM → Operator Guidance

Note        : Sensor data in this script is simulated. Real deployment would
              ingest live data from GlobalEye's Track Data Fusion Engine (TDFE).
=============================================================================
"""

import pandas as pd
import numpy as np
import cohere
import json
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME     = "command-a-03-2025"    # Cohere's latest model (2025)
NUM_TARGETS    = 15                     # Number of simulated contacts
RANDOM_SEED    = 42                     # Reproducibility


# =============================================================================
# LAYER 1: DATA SIMULATION
# Simulates the kind of multi-sensor data GlobalEye's TDFE would produce.
# In production this layer would be replaced by a live data ingestion module.
# =============================================================================

def simulate_sensor_data(n_targets: int = NUM_TARGETS) -> pd.DataFrame:
    """
    Simulates multi-modal sensor data from GlobalEye's integrated sensor suite.

    Sensors modelled:
        - Erieye ER AESA Radar     → bearing, speed, altitude, RCS
        - Electro-Optical/IR       → ir_signature
        - AIS Transponder          → ais_active, vessel_id
        - Zone Monitoring System   → zone

    Returns:
        pd.DataFrame: One row per tracked contact with all sensor fields.
    """
    np.random.seed(RANDOM_SEED)

    ir_options   = ["LOW", "MEDIUM", "HIGH"]
    zone_options = ["Clear", "Restricted", "Danger"]
    type_options = ["Unknown", "Commercial", "Military", "Unregistered"]

    data = {
        "target_id"     : [f"T-{i:03d}" for i in range(n_targets)],
        "timestamp"     : [datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")] * n_targets,
        "bearing_deg"   : np.random.uniform(0, 360, n_targets).round(1),
        "speed_knots"   : np.random.uniform(50, 600, n_targets).round(1),
        "altitude_ft"   : np.random.uniform(100, 40000, n_targets).round(0).astype(int),
        "rcs_sqm"       : np.random.uniform(0.1, 50, n_targets).round(2),
        "ir_signature"  : np.random.choice(ir_options, n_targets),
        "ais_active"    : np.random.choice([True, False], n_targets),
        "contact_type"  : np.random.choice(type_options, n_targets),
        "zone"          : np.random.choice(zone_options, n_targets),
        "track_quality" : np.random.uniform(0.5, 1.0, n_targets).round(2),
    }

    return pd.DataFrame(data)


# =============================================================================
# LAYER 2: FEATURE ENGINEERING
# Extracts meaningful features from raw sensor values.
# This is the core Data Science contribution — turning numbers into signals.
# =============================================================================

def bearing_to_direction(degrees: float) -> str:
    """Converts a bearing in degrees to a compass direction string."""
    directions = ["North", "Northeast", "East", "Southeast",
                  "South", "Southwest", "West", "Northwest"]
    index = int((degrees + 22.5) / 45) % 8
    return directions[index]


def classify_speed(knots: float) -> str:
    """Classifies speed into a human-readable category."""
    if knots > 450:
        return "very high speed — possible missile or fast jet"
    elif knots > 300:
        return "high speed — likely military or fast commercial"
    elif knots > 150:
        return "moderate speed — typical commercial aviation"
    else:
        return "low speed — helicopter or slow-moving aircraft"


def classify_altitude(feet: int) -> str:
    """Classifies altitude into an operational category."""
    if feet < 500:
        return "extremely low — terrain-hugging profile, potential threat"
    elif feet < 3000:
        return "low altitude — below standard commercial traffic lanes"
    elif feet < 10000:
        return "medium altitude — within standard approach corridors"
    else:
        return "high altitude — standard commercial cruising level"


def classify_rcs(sqm: float) -> str:
    """
    Classifies radar cross-section (RCS) — a measure of how visible
    an object is on radar. Small RCS can indicate stealth design.
    """
    if sqm < 1:
        return "very small — consistent with missile, drone or stealth aircraft"
    elif sqm < 5:
        return "small — light aircraft or small military jet"
    elif sqm < 20:
        return "medium — standard commercial or military aircraft"
    else:
        return "large — heavy transport, bomber or large commercial jet"


def compute_threat_score(row: pd.Series) -> int:
    """
    Computes a threat score (0–100) for each tracked contact
    based on weighted sensor features.

    Scoring logic:
        - High speed at low altitude = missile/attack profile
        - Small RCS = stealth/low-observable design
        - High IR signature = active engine heat source
        - No AIS transponder = unidentified/non-cooperative
        - Zone violation = unauthorized airspace entry
        - Track quality penalty for unreliable data

    Returns:
        int: Threat score between 0 and 100.
    """
    score = 0

    # Speed scoring
    if row["speed_knots"] > 450:
        score += 30
    elif row["speed_knots"] > 300:
        score += 15

    # Altitude scoring — low altitude is high threat
    if row["altitude_ft"] < 500:
        score += 25
    elif row["altitude_ft"] < 2000:
        score += 12

    # RCS scoring — small RCS = stealth
    if row["rcs_sqm"] < 1:
        score += 20
    elif row["rcs_sqm"] < 5:
        score += 10

    # IR signature
    if row["ir_signature"] == "HIGH":
        score += 15
    elif row["ir_signature"] == "MEDIUM":
        score += 5

    # Transponder — no AIS = unidentified
    if not row["ais_active"]:
        score += 20

    # Zone violation
    if row["zone"] == "Danger":
        score += 30
    elif row["zone"] == "Restricted":
        score += 15

    # Track quality penalty — low quality data = less certain
    if row["track_quality"] < 0.7:
        score = int(score * 0.85)

    return min(score, 100)


def classify_threat_level(score: int) -> str:
    """Maps a numeric threat score to a categorical threat level."""
    if score >= 70:
        return "CRITICAL"
    elif score >= 45:
        return "ELEVATED"
    elif score >= 20:
        return "MODERATE"
    else:
        return "LOW"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering functions to the raw sensor DataFrame.
    Adds derived columns used for both threat scoring and NL conversion.

    Returns:
        pd.DataFrame: Enriched DataFrame with all computed features.
    """
    df = df.copy()
    df["direction"]      = df["bearing_deg"].apply(bearing_to_direction)
    df["speed_class"]    = df["speed_knots"].apply(classify_speed)
    df["altitude_class"] = df["altitude_ft"].apply(classify_altitude)
    df["rcs_class"]      = df["rcs_sqm"].apply(classify_rcs)
    df["threat_score"]   = df.apply(compute_threat_score, axis=1)
    df["threat_level"]   = df["threat_score"].apply(classify_threat_level)
    df = df.sort_values("threat_score", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# LAYER 3: NATURAL LANGUAGE CONVERSION
# This is the key bridge — converts structured sensor data into text that
# Cohere's LLM can understand, reason about, and respond to.
# =============================================================================

def sensor_to_natural_language(row: pd.Series) -> str:
    """
    Converts a single sensor data row into a structured natural language
    report suitable for input to Cohere's Command model.

    This is the core innovation of this pipeline — LLMs cannot directly
    interpret raw numbers like RCS = 0.8 sqm or speed = 487 knots. This
    function translates those values into contextual language the model
    can reason about.

    Args:
        row (pd.Series): A single enriched sensor data row.

    Returns:
        str: A structured natural language sensor report.
    """
    ais_text = (
        "No AIS transponder signal detected — contact is non-cooperative "
        "and identity cannot be confirmed through standard means."
        if not row["ais_active"]
        else f"AIS transponder is active. Contact type registered as: {row['contact_type']}."
    )

    report = f"""
GLOBALEYE SENSOR REPORT
========================
Timestamp     : {row['timestamp']}
Target ID     : {row['target_id']}
Threat Level  : {row['threat_level']} ({row['threat_score']}/100)
Track Quality : {row['track_quality']} (1.0 = high confidence)

MOVEMENT PROFILE
Contact is approaching from the {row['direction']} (bearing {row['bearing_deg']}°).
Speed: {row['speed_knots']} knots — {row['speed_class']}.
Altitude: {row['altitude_ft']:,} ft — {row['altitude_class']}.

SENSOR READINGS
Radar Cross-Section (RCS): {row['rcs_sqm']} sqm — {row['rcs_class']}.
Infrared (IR) Signature: {row['ir_signature']} — {"active heat source detected" if row['ir_signature'] == 'HIGH' else "moderate thermal signature" if row['ir_signature'] == 'MEDIUM' else "minimal thermal output"}.
{ais_text}

ZONE STATUS
Contact is currently located within a {row['zone']} zone.
{"ZONE VIOLATION ALERT: Contact has entered a restricted or danger zone without authorization." if row['zone'] != 'Clear' else "No zone violations detected at this time."}
========================
    """.strip()

    return report


# =============================================================================
# LAYER 4: COHERE LLM INTEGRATION
# Feeds the natural language report to Cohere's Command model and retrieves
# an actionable threat assessment for the human operator.
# =============================================================================

SYSTEM_PROMPT = """
You are an advanced AI military airspace analyst embedded within the Saab
GlobalEye AEW&C (Airborne Early Warning and Control) mission system,
powered by Cohere's enterprise AI platform.

Your role is to receive structured sensor reports from GlobalEye's Track
Data Fusion Engine and provide concise, accurate, and actionable threat
assessments to human operators in real time.

RESPONSE FORMAT:
1. THREAT ASSESSMENT: One sentence summary of the contact's threat status.
2. KEY INDICATORS: 2-3 bullet points explaining the most significant factors.
3. CONFIDENCE: State your confidence level (Low / Medium / High) and why.
4. RECOMMENDED ACTION: One clear, specific action for the human operator.
5. WATCH POINTS: Any additional factors the operator should monitor.

RULES:
- Be direct and concise — operators make time-critical decisions.
- Never fabricate sensor data not present in the report.
- Always flag low track quality as a source of uncertainty.
- Recommend human oversight for all CRITICAL-level contacts.
- Use standard military brevity where appropriate.
"""


def get_operator_guidance(sensor_report: str, api_key: str) -> str:
    """
    Sends a natural language sensor report to Cohere's Command model
    and returns an actionable operator guidance response.

    Args:
        sensor_report (str): Natural language sensor report from Layer 3.
        api_key (str): Cohere API key.

    Returns:
        str: LLM-generated threat assessment and operator recommendation.
    """
    try:
        co = cohere.ClientV2(api_key)
        response = co.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": sensor_report}
            ],
        )
        return response.message.content[0].text

    except Exception as e:
        return f"[API ERROR] Cohere call failed: {str(e)}"


# =============================================================================
# PIPELINE ORCHESTRATOR
# Runs all four layers end-to-end and produces a full mission report.
# =============================================================================

def run_pipeline(api_key: str = COHERE_API_KEY,
                 n_targets: int = NUM_TARGETS,
                 top_n: int = 3) -> dict:
    """
    Orchestrates the full multi-modal sensor data to LLM pipeline.

    Steps:
        1. Simulate (or ingest) sensor data
        2. Engineer features and compute threat scores
        3. Convert top-N highest threat contacts to natural language
        4. Feed each to Cohere LLM and collect operator guidance
        5. Return full structured mission report

    Args:
        api_key  (str): Cohere API key.
        n_targets (int): Number of sensor contacts to simulate.
        top_n    (int): Number of highest-threat contacts to send to LLM.

    Returns:
        dict: Full mission report with sensor data and LLM assessments.
    """
    print("\n" + "="*60)
    print("  GLOBALEYE AI MISSION SUPPORT PIPELINE")
    print("  Saab × Cohere — Multi-Modal Data Bridge")
    print("="*60)

    # Layer 1 — Simulate sensor data
    print(f"\n[Layer 1] Simulating sensor data for {n_targets} contacts...")
    raw_df = simulate_sensor_data(n_targets)
    print(f"          {len(raw_df)} contacts acquired from sensor suite.")

    # Layer 2 — Feature engineering and threat scoring
    print("\n[Layer 2] Running feature engineering and threat scoring...")
    enriched_df = engineer_features(raw_df)
    critical = (enriched_df["threat_level"] == "CRITICAL").sum()
    elevated = (enriched_df["threat_level"] == "ELEVATED").sum()
    print(f"          Threat summary → CRITICAL: {critical} | ELEVATED: {elevated}")

    # Layer 3 + 4 — NL conversion and LLM assessment for top threats
    print(f"\n[Layer 3+4] Processing top {top_n} highest-threat contacts...\n")
    mission_report = {
        "generated_at"   : datetime.now().isoformat(),
        "total_contacts" : n_targets,
        "critical_count" : int(critical),
        "elevated_count" : int(elevated),
        "assessments"    : []
    }

    for i, row in enriched_df.head(top_n).iterrows():
        print(f"  Processing {row['target_id']} (Score: {row['threat_score']}/100 — {row['threat_level']})...")

        # Layer 3 — Convert to natural language
        nl_report = sensor_to_natural_language(row)

        # Layer 4 — Get Cohere LLM guidance
        guidance = get_operator_guidance(nl_report, api_key)

        mission_report["assessments"].append({
            "target_id"      : row["target_id"],
            "threat_score"   : row["threat_score"],
            "threat_level"   : row["threat_level"],
            "sensor_report"  : nl_report,
            "llm_guidance"   : guidance
        })

        print(f"  Assessment complete for {row['target_id']}.\n")

    print("[Pipeline Complete] Mission report generated.\n")
    return mission_report, enriched_df


def display_report(mission_report: dict) -> None:
    """Prints the mission report in a readable format."""
    print("\n" + "="*60)
    print("  MISSION REPORT SUMMARY")
    print("="*60)
    print(f"Generated  : {mission_report['generated_at']}")
    print(f"Contacts   : {mission_report['total_contacts']}")
    print(f"Critical   : {mission_report['critical_count']}")
    print(f"Elevated   : {mission_report['elevated_count']}")

    for a in mission_report["assessments"]:
        print(f"\n{'─'*60}")
        print(f"TARGET: {a['target_id']}  |  {a['threat_level']}  ({a['threat_score']}/100)")
        print(f"{'─'*60}")
        print("\n[SENSOR REPORT]\n")
        print(a["sensor_report"])
        print("\n[COHERE LLM OPERATOR GUIDANCE]\n")
        print(a["llm_guidance"])

    print("\n" + "="*60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Run the full pipeline
    # Set your Cohere API key in COHERE_API_KEY at the top of this file
    # or pass it directly: run_pipeline(api_key="your-key-here")
    report, sensor_df = run_pipeline(
        api_key=COHERE_API_KEY,
        n_targets=15,
        top_n=3
    )

    # Display readable report in terminal
    display_report(report)

    # Save full report to JSON
    with open("mission_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nFull report saved to mission_report.json")

    # Save enriched sensor data to CSV
    sensor_df.to_csv("sensor_data_enriched.csv", index=False)
    print("Enriched sensor data saved to sensor_data_enriched.csv")
