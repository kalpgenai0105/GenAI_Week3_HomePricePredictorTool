# Date: 9/18/2026
# Author: Kalpesh Sane
# Week-3, Homework#1

"""
Few-shot prompting with Gemini to predict California house prices.

Workflow:
  1. Load california_housing_train.csv and sample 60-70 rows as few-shot examples.
  2. Build a prompt that contains those worked examples.
  3. Collect a new house's attributes from the user (or fall back to defaults).
  4. Call the Gemini API and parse the predicted price from the response.
"""

import os
import re
import sys
import random
import pandas as pd

# ── single constant: change this if gemini-3.5-flash is unavailable ──────────
MODEL_NAME = "gemini-3.5-flash"

# ── column names as they appear in the CSV ────────────────────────────────────
FEATURE_COLS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
TARGET_COL = "median_house_value"

# ── sensible fallback values used when the user skips manual input ────────────
DEFAULT_ATTRIBUTES = {
    "longitude":          -118.25,
    "latitude":            34.05,
    "housing_median_age":  29.0,
    "total_rooms":         2127.0,
    "total_bedrooms":      435.0,
    "population":          1166.0,
    "households":          409.0,
    "median_income":       3.53,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading and sampling
# ─────────────────────────────────────────────────────────────────────────────

def load_and_sample(csv_path: str, seed: int | None = None) -> pd.DataFrame:
    """
    Read the CSV and return a random sample of 60-70 rows.

    Parameters
    ----------
    csv_path : str
        Path to california_housing_train.csv.
    seed : int or None
        Random seed for reproducibility.  None (default) → truly random.

    Returns
    -------
    pd.DataFrame
        Sampled rows containing both feature columns and the target column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: '{csv_path}'\n"
            "Please make sure california_housing_train.csv is in the same "
            "directory as this script, or pass the correct path."
        )

    df = pd.read_csv(csv_path)

    # Verify all expected columns are present
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected column(s): {missing}")

    # Determine sample size randomly within [60, 70] on every run
    rng = random.Random(seed)          # isolated RNG so global state is unaffected
    n_samples = rng.randint(60, 70)

    sampled = df.sample(n=n_samples, random_state=seed)
    print(f"[info] Sampled {n_samples} rows from {len(df)} available rows.")
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def build_few_shot_prompt(sampled_df: pd.DataFrame, new_house: dict) -> str:
    """
    Construct the complete few-shot prompt.

    Each sampled row is formatted as a worked example showing the input
    attributes and the known median_house_value.  The prompt ends with the
    new house's attributes and asks for a numeric price prediction.

    Parameters
    ----------
    sampled_df : pd.DataFrame
        The sampled rows that serve as few-shot examples.
    new_house : dict
        Attribute values for the house whose price we want to predict.

    Returns
    -------
    str
        The fully assembled prompt string.
    """
    lines = [
        "You are a real-estate valuation expert for California.",
        "Below are examples of California houses with their attributes and "
        "actual median sale prices.",
        "Study these examples carefully and use the same pricing patterns to "
        "predict the price for the new house at the end.",
        "",
        "=== FEW-SHOT EXAMPLES ===",
        "",
    ]

    # Format each sampled row as a labelled example block
    for idx, (_, row) in enumerate(sampled_df.iterrows(), start=1):
        lines.append(f"Example {idx}:")
        for col in FEATURE_COLS:
            lines.append(f"  {col}: {row[col]}")
        # Show the answer so the model learns the mapping
        lines.append(f"  → median_house_value: {row[TARGET_COL]:.2f}")
        lines.append("")

    # Append the prediction request with the new house's attributes
    lines += [
        "=== PREDICTION REQUEST ===",
        "",
        "Now predict the median_house_value for this new house:",
    ]
    for col in FEATURE_COLS:
        lines.append(f"  {col}: {new_house[col]}")

    lines += [
        "",
        "IMPORTANT: respond with ONLY a single numeric value representing the "
        "predicted median_house_value in US dollars.  Do not include any "
        "explanation, currency symbol, or additional text—just the number.",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3. User-attribute collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_user_attributes() -> dict:
    """
    Prompt the user to enter each feature value interactively.

    If the user presses Enter without typing anything, the corresponding
    DEFAULT_ATTRIBUTES value is used so the script stays runnable without
    manual input.

    Returns
    -------
    dict
        Mapping of feature name → float value.
    """
    print("\nEnter attributes for the house you want to price.")
    print("Press Enter to accept the default value shown in brackets.\n")

    attributes = {}
    for col in FEATURE_COLS:
        default = DEFAULT_ATTRIBUTES[col]
        raw = input(f"  {col} [{default}]: ").strip()
        if raw == "":
            attributes[col] = default
        else:
            try:
                attributes[col] = float(raw)
            except ValueError:
                print(f"  [warn] Could not parse '{raw}' as a number; "
                      f"using default {default}.")
                attributes[col] = default

    return attributes


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gemini API call
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    """
    Send the prompt to the Gemini model and return the raw text response.

    Reads GEMINI_API_KEY from the environment; raises RuntimeError if absent.

    Parameters
    ----------
    prompt : str
        The fully assembled few-shot prompt.

    Returns
    -------
    str
        Raw text of the model's response.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Export it before running:  export GEMINI_API_KEY='your-key-here'"
        )

    # Import here so the script fails with a clear message if the package
    # is missing rather than at the top-level import.
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "The 'google-genai' package is not installed.\n"
            "Install it with:  pip install google-genai"
        )

    # Initialise the client with the API key
    client = genai.Client(api_key=api_key)

    print(f"\n[info] Calling model '{MODEL_NAME}' …")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    # Extract the text from the response object
    return response.text


# ─────────────────────────────────────────────────────────────────────────────
# 5. Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_price(raw_response: str) -> float | None:
    """
    Extract a numeric price from the model's response text.

    Strips leading/trailing whitespace, currency symbols ($), and commas,
    then searches for the first valid float or integer.

    Parameters
    ----------
    raw_response : str
        The raw text returned by the model.

    Returns
    -------
    float or None
        Parsed price, or None if no valid number could be found.
    """
    # Remove common formatting artefacts before searching
    cleaned = raw_response.strip().replace("$", "").replace(",", "")

    # Match an optional sign, digits, and an optional decimal part
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if match:
        return float(match.group())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(csv_path: str = "california_housing_train.csv", seed: int | None = None) -> None:
    """
    Orchestrate all steps: load data → build prompt → call Gemini → print result.

    Parameters
    ----------
    csv_path : str
        Path to the training CSV.
    seed : int or None
        Optional random seed for reproducible sampling.
    """
    # ── Step 1: load and sample the dataset ──────────────────────────────────
    try:
        sampled_df = load_and_sample(csv_path, seed=seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: collect the new house's attributes from the user ─────────────
    new_house = collect_user_attributes()

    # ── Step 3: build the few-shot prompt ────────────────────────────────────
    prompt = build_few_shot_prompt(sampled_df, new_house)

    # Uncomment the next line to inspect the full prompt before sending:
    # print("\n--- PROMPT ---\n", prompt, "\n--- END PROMPT ---\n")

    # ── Step 4: call the Gemini API ──────────────────────────────────────────
    try:
        raw_response = call_gemini(prompt)
    except (RuntimeError, ImportError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Raw model response: {raw_response!r}")

    # ── Step 5: parse and display the predicted price ────────────────────────
    predicted_price = parse_price(raw_response)
    if predicted_price is None:
        print(
            "[warn] Could not extract a numeric price from the model's response.\n"
            f"       Full response: {raw_response}"
        )
    else:
        print(f"\nPredicted median house value: ${predicted_price:,.2f}")


if __name__ == "__main__":
    # Pass an integer seed via the command line for reproducible runs, e.g.:
    #   python housing_price_predictor.py 42
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(seed=seed_arg)
