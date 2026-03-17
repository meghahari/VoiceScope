# Audio processing constants
TIME_STEPS = 128
NUM_FEATURES = 120
TARGET_SR = 16000
MAX_RECORD_DURATION = 10

# Model labels (must match model output shapes)
GENDER_LABELS = {0: "Female", 1: "Male"}
AGE_LABELS = {0: "Twenties (20-29)", 1: "Thirties (30-39)", 2: "Fifties (50-59)"}
ACCENT_LABELS = {
    0: "American English",
    1: "Indian English", 
    2: "British English",
    3: "Australian English",
    4: "Other"
}

