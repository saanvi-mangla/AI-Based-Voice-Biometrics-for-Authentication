# ==============================================================================
# SECTION 1: SETUP AND IMPORTS
# ==============================================================================
print("--- Running Section 1: Setup and Imports ---")
import os
import numpy as np
import librosa # For feature extraction
import soundfile as sf # For loading audio robustly
import pickle # For saving/loading GMM models
from sklearn.mixture import GaussianMixture # The GMM model
import warnings
import glob # For finding audio files
from collections import defaultdict
import matplotlib.pyplot as plt # For plotting (optional)
import shutil # For file operations if needed
import re
import time # To time the batch enrollment

# Check if essential libraries are installed
try:
    import sklearn
except ImportError:
    print("Error: scikit-learn not found. Please install using: pip install scikit-learn")
    exit()
try:
    import librosa
except ImportError:
    print("Error: librosa not found. Please install using: pip install librosa")
    exit()
try:
    import soundfile
except ImportError:
    print("Error: soundfile not found. Please install using: pip install soundfile")
    exit()


# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning) # Ignore potential GMM warnings

print("--- Section 1 Complete ---")


# ==============================================================================
# SECTION 2: CONFIGURATION AND TIMIT DATA PREPARATION
# ==============================================================================
print("\n--- Running Section 2: Configuration and TIMIT Data Preparation ---")

# --- Feature Extraction Parameters ---
SAMPLE_RATE = 16000       # Target sample rate (TIMIT is already 16k)
N_MFCC = 13               # Number of base MFCC coefficients
FRAME_LENGTH_MS = 25      # Frame duration in milliseconds
HOP_LENGTH_MS = 10        # Frame step in milliseconds
INCLUDE_DELTAS = True     # Include delta and delta-delta features?
TOTAL_FEATURES = N_MFCC * 3 if INCLUDE_DELTAS else N_MFCC
print(f"Feature Params: SR={SAMPLE_RATE}, N_MFCC={N_MFCC}, Deltas={INCLUDE_DELTAS}, TotalFeatures={TOTAL_FEATURES}")

# --- GMM Parameters ---
N_COMPONENTS = 32         # Number of Gaussian components per GMM (Hyperparameter: Tune this!)
COVARIANCE_TYPE = 'diag'  # Covariance type for GMM ('diag' is common and faster)
REG_COVAR = 1e-6          # Regularization added to covariance diagonal (prevents singularity)
MAX_ITER_GMM = 100        # Max iterations for GMM training
print(f"GMM Params: Components={N_COMPONENTS}, CovType={COVARIANCE_TYPE}, MaxIter={MAX_ITER_GMM}")

# --- Paths (LOCAL EXECUTION) ---
LOCAL_BASE_PATH = '.' # Outputs saved in the script's directory
print(f"Using Base Path for Outputs: {os.path.abspath(LOCAL_BASE_PATH)}")

# *** ACTION REQUIRED: Set the path to your main LOCAL TIMIT folder ***
# This folder should contain the TRAIN and TEST subdirectories directly inside it.
TIMIT_DATASET_FOLDER_ON_DISK = '/Users/saanvimangla/Downloads/archive-3/data' # SET YOUR PATH HERE

GMM_MODEL_FOLDER_NAME = 'Voice_Biometrics_TIMIT_GMM/GMM_Models' # Subfolder name for GMM models
FEATURE_FOLDER_NAME = 'Voice_Biometrics_TIMIT_GMM/Features' # Optional subfolder for features

# Construct full paths
TIMIT_PATH = TIMIT_DATASET_FOLDER_ON_DISK # Use the direct local path
MODEL_DIR = os.path.join(LOCAL_BASE_PATH, GMM_MODEL_FOLDER_NAME)
FEATURE_DIR = os.path.join(LOCAL_BASE_PATH, FEATURE_FOLDER_NAME) # Optional

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True) # For saving GMM models
os.makedirs(FEATURE_DIR, exist_ok=True) # Optional

print(f"Expecting TIMIT Dataset Root at: {TIMIT_PATH}")
print(f"Saving/Loading GMM Models at: {MODEL_DIR}")

# --- Workflow Parameters ---
# ** IMPORTANT: SET A THRESHOLD BASED ON EVALUATION **
VERIFICATION_THRESHOLD = -20.0 # EXAMPLE THRESHOLD - MUST BE DETERMINED EMPIRICALLY!
print(f"Verification Threshold (Log-Likelihood per frame): {VERIFICATION_THRESHOLD}")

# --- TIMIT Data Loading & Preparation ---
# Dictionaries to map speaker_id -> list of file paths
enroll_files = defaultdict(list) # Files from TRAIN for enrollment
test_files = defaultdict(list)   # Files from TEST for verification/evaluation

print(f"\n--- Attempting to load TIMIT .WAV file paths from: {TIMIT_PATH} ---")
print(f"[DEBUG] Checking existence of TIMIT_PATH: {os.path.exists(TIMIT_PATH)}")

def process_timit_directory(subset_dir, file_dict):
    """ Scans a TIMIT subset (TRAIN or TEST) directory and populates the file dictionary. """
    files_found_in_subset = 0
    speakers_found_in_subset = set()
    subset_path = os.path.join(TIMIT_PATH, subset_dir)
    if not os.path.exists(subset_path):
        print(f"Warning: TIMIT subset directory not found: {subset_path}")
        return 0, set()
    dialect_regions = glob.glob(os.path.join(subset_path, 'DR*'))
    if not dialect_regions:
         print(f"Warning: No dialect regions (DR*) found in {subset_path}")
         speaker_dirs_alt = glob.glob(os.path.join(subset_path, '*'))
         if any(os.path.isdir(p) for p in speaker_dirs_alt):
              dialect_regions = [subset_path]
              print(f"[DEBUG] Found speaker folders directly under {subset_dir}, proceeding without DR level.")
         else: print(f"Warning: No speaker folders found directly under {subset_dir} either."); return 0, set()
    for dr_path in dialect_regions:
        speaker_paths = glob.glob(os.path.join(dr_path, '*'))
        for spk_path in speaker_paths:
            if not os.path.isdir(spk_path): continue
            speaker_id = os.path.basename(spk_path)
            wav_files = glob.glob(os.path.join(spk_path, '*.WAV')) + \
                        glob.glob(os.path.join(spk_path, '*.wav'))
            if wav_files:
                file_dict[speaker_id].extend(wav_files)
                files_found_in_subset += len(wav_files)
                speakers_found_in_subset.add(speaker_id)
    return files_found_in_subset, speakers_found_in_subset

if os.path.exists(TIMIT_PATH):
    print("\nProcessing TRAIN directory...")
    train_files_count, train_speakers = process_timit_directory('TRAIN', enroll_files)
    print(f"Found {train_files_count} files for {len(train_speakers)} speakers in TRAIN set (for enrollment).")
    print("\nProcessing TEST directory...")
    test_files_count, test_speakers = process_timit_directory('TEST', test_files)
    print(f"Found {test_files_count} files for {len(test_speakers)} speakers in TEST set (for verification/evaluation).")
    if not enroll_files and not test_files: print("\nWarning: No speaker files loaded...")
    else:
        if enroll_files: print(f"\n[DEBUG] Example enrollment speaker: {list(enroll_files.keys())[0]}")
        else: print("\n[DEBUG] No enrollment speakers found.")
        if test_files: print(f"[DEBUG] Example test speaker: {list(test_files.keys())[0]}")
        else: print("[DEBUG] No test speakers found.")
else:
    print(f"Error: TIMIT dataset path does not exist: {TIMIT_PATH}"); exit()

# Check if speakers were actually found for enrollment
if not enroll_files:
    print("\nError: No speaker data loaded from the TRAIN directory. Cannot proceed with enrollment.")
    exit()

print("--- Section 2 Complete ---")


# ==============================================================================
# SECTION 3: FEATURE EXTRACTION FUNCTION (MFCCs using Librosa)
# ==============================================================================
print("\n--- Defining Section 3: Feature Extraction Function (MFCCs) ---")

def extract_mfcc_features(audio_path, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
                          frame_length_ms=FRAME_LENGTH_MS, hop_length_ms=HOP_LENGTH_MS,
                          include_deltas=INCLUDE_DELTAS):
    """
    Extracts MFCCs (optionally with delta and delta-delta) from an audio file.
    Applies Cepstral Mean and Variance Normalization (CMVN).
    """
    if not os.path.exists(audio_path):
        # print(f"Warning in extract_mfcc: File not found {audio_path}") # Less verbose warning
        return None
    try:
        y, sr_orig = sf.read(audio_path)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr_orig != sample_rate: y = librosa.resample(y, orig_sr=sr_orig, target_sr=sample_rate)
        sr = sample_rate
        min_len_sec = 0.1
        if len(y) < int(min_len_sec * sr): return None
        n_fft = int(frame_length_ms / 1000 * sr)
        hop_length = int(hop_length_ms / 1000 * sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        if include_deltas:
            delta1 = librosa.feature.delta(mfccs)
            delta2 = librosa.feature.delta(mfccs, order=2)
            features = np.concatenate((mfccs, delta1, delta2), axis=0)
        else: features = mfccs
        if features.shape[1] > 1:
            mean, std = np.mean(features, axis=1, keepdims=True), np.std(features, axis=1, keepdims=True)
            std[std < 1e-8] = 1e-8
            features = (features - mean) / std
        elif features.shape[1] == 1: features = features - np.mean(features, axis=1, keepdims=True)
        else: return None
        return features.T
    except Exception as e:
        print(f"Warning: Error processing {os.path.basename(audio_path)} in extract_mfcc: {e}")
        return None

print("--- Section 3 Defined ---")


# ==============================================================================
# SECTION 4: GMM ENROLLMENT FUNCTION (using scikit-learn)
# ==============================================================================
print("\n--- Defining Section 4: GMM Enrollment Function ---")

def enroll_speaker_gmm(speaker_id, enrollment_wav_files):
    """
    Extracts MFCC features from enrollment files, trains, and saves a GMM.
    (Modified to be less verbose during batch enrollment).
    """
    model_path = os.path.join(MODEL_DIR, f"{speaker_id}.gmm")
    all_enroll_features = []

    # print(f"Attempting to enroll speaker '{speaker_id}' using GMM...") # Less verbose
    if not enrollment_wav_files: print(f"  Error enrolling {speaker_id}: No enrollment files provided."); return False

    # print(f"  Extracting features from {len(enrollment_wav_files)} enrollment WAV files...") # Less verbose
    for audio_file in enrollment_wav_files:
        features = extract_mfcc_features(audio_file) # Use MFCC extraction
        if features is not None and features.shape[0] > 0: all_enroll_features.append(features)
        # else: print(f"  Warning: Skipping file (feature extraction failed): {os.path.basename(audio_file)}") # Less verbose

    if not all_enroll_features: print(f"Error enrolling {speaker_id}: No valid features extracted."); return False

    try: enroll_features_matrix = np.vstack(all_enroll_features)
    except ValueError as e_stack: print(f"Error stacking features for {speaker_id}: {e_stack}"); return False

    n_frames, n_features = enroll_features_matrix.shape
    # print(f"  Total valid enrollment frames for {speaker_id}: {n_frames}") # Less verbose

    min_frames_approx = n_features * N_COMPONENTS
    if n_frames < N_COMPONENTS: print(f"Error enrolling {speaker_id}: Fewer frames ({n_frames}) than GMM components ({N_COMPONENTS})."); return False
    # if n_frames < min_frames_approx: print(f"Warning for {speaker_id}: Low frame count ({n_frames}) relative to model complexity (~{min_frames_approx}).") # Less verbose

    try:
        # print(f"  Training GMM for {speaker_id}...") # Less verbose
        gmm = GaussianMixture(n_components=N_COMPONENTS, covariance_type=COVARIANCE_TYPE,
                              random_state=0, reg_covar=REG_COVAR, max_iter=MAX_ITER_GMM,
                              n_init=1, warm_start=False, verbose=0) # verbose=0
        gmm.fit(enroll_features_matrix)

        # if not gmm.converged_: print(f"Warning: GMM training for {speaker_id} did not converge.") # Less verbose

        # print(f"  Saving trained GMM model to: {model_path}") # Less verbose
        with open(model_path, 'wb') as f_model: pickle.dump(gmm, f_model)
        # print(f"  Successfully enrolled speaker '{speaker_id}'.") # Less verbose
        return True
    except Exception as e_train: print(f"Error training GMM for '{speaker_id}': {e_train}"); return False

print("--- Section 4 Defined ---")


# ==============================================================================
# SECTION 5: **NEW** BATCH ENROLLMENT OF ALL TRAIN SPEAKERS
# ==============================================================================
print("\n--- Running Section 5: Batch Enrollment ---")

enrolled_speaker_count = 0
failed_enrollment_speakers = []
total_speakers_to_enroll = len(enroll_files)
start_time_enrollment = time.time()

print(f"Starting batch enrollment for {total_speakers_to_enroll} speakers found in TRAIN set...")

# Loop through all speakers found in the TRAIN set
for i, (speaker_id, files) in enumerate(enroll_files.items()):
    print(f"Enrolling speaker {i+1}/{total_speakers_to_enroll}: {speaker_id}...")
    success = enroll_speaker_gmm(speaker_id, files)
    if success:
        enrolled_speaker_count += 1
    else:
        failed_enrollment_speakers.append(speaker_id)
        print(f"  --> Enrollment FAILED for {speaker_id}")

end_time_enrollment = time.time()
elapsed_time = end_time_enrollment - start_time_enrollment

print("\n--- Batch Enrollment Summary ---")
print(f"Successfully enrolled {enrolled_speaker_count} out of {total_speakers_to_enroll} speakers.")
if failed_enrollment_speakers:
    print(f"Failed to enroll {len(failed_enrollment_speakers)} speakers: {', '.join(failed_enrollment_speakers)}")
print(f"Total enrollment time: {elapsed_time:.2f} seconds.")
print("-------------------------------")

# Optional: Check if any speakers were enrolled before proceeding
if enrolled_speaker_count == 0:
    print("\nError: No speakers were successfully enrolled during batch process. Exiting.")
    exit()

print("--- Section 5 Complete ---")


# ==============================================================================
# SECTION 6: GMM VERIFICATION FUNCTION (using scikit-learn)
# ==============================================================================
# Renumbered section
print("\n--- Defining Section 6: GMM Verification Function ---")

def verify_speaker_gmm(claimed_speaker_id, test_wav_file):
    """
    Extracts MFCC features from a test file and scores it against a claimed speaker's GMM.
    """
    model_path = os.path.join(MODEL_DIR, f"{claimed_speaker_id}.gmm")
    if not os.path.exists(model_path): print(f"Error: Enrollment GMM not found for '{claimed_speaker_id}' at {model_path}"); return None

    test_features = extract_mfcc_features(test_wav_file) # Use MFCC extraction
    if test_features is None or test_features.shape[0] == 0: print(f"Error: Could not extract valid features from test file: {os.path.basename(test_wav_file)}"); return None

    try:
        with open(model_path, 'rb') as f_model: gmm = pickle.load(f_model) # Load GMM using pickle
        log_likelihood_per_frame = gmm.score(test_features)
        return log_likelihood_per_frame
    except FileNotFoundError: print(f"Error loading GMM model file: {model_path}"); return None
    except Exception as e_verify: print(f"Error during GMM verification scoring for '{claimed_speaker_id}': {e_verify}"); return None

print("--- Section 6 Defined ---")


# ==============================================================================
# SECTION 7: BASIC COMMAND-LINE INTERFACE (CLI) FUNCTIONS
# ==============================================================================
# Renumbered section
print("\n--- Defining Section 7: CLI Functions ---")

# list_available_speakers_enroll/test remain the same (use enroll_files/test_files)
def list_available_speakers_enroll():
    """Lists speakers found in the TIMIT TRAIN set."""
    print("\n--- Available Speakers for Enrollment (from TIMIT TRAIN set) ---")
    if not enroll_files: print("No speakers found/loaded from TRAIN set."); return []
    speakers = sorted(enroll_files.keys())
    if not speakers: print("Enrollment data dictionary is empty."); return []
    print(f"Found {len(speakers)} speakers in TRAIN set:")
    for spk in speakers: print(f"- {spk} ({len(enroll_files[spk])} files)")
    print("-------------------------------------------------------")
    return speakers

def list_available_speakers_test():
    """Lists speakers found in the TIMIT TEST set."""
    print("\n--- Available Speakers for Testing (from TIMIT TEST set) ---")
    if not test_files: print("No speakers found/loaded from TEST set."); return []
    speakers = sorted(test_files.keys())
    if not speakers: print("Test data dictionary is empty."); return []
    print(f"Found {len(speakers)} speakers in TEST set:")
    for spk in speakers: print(f"- {spk} ({len(test_files[spk])} files)")
    print("-------------------------------------------------------")
    return speakers

# list_enrolled_models needs to look for .gmm files now
def list_enrolled_models():
    """Lists speaker GMM models found in the model directory."""
    print(f"\n--- Enrolled Speaker GMM Models (in {MODEL_DIR}) ---")
    try:
        model_files = glob.glob(os.path.join(MODEL_DIR, '*.gmm')) # Look for .gmm
        enrolled = []
        if not model_files: print("No enrolled GMM models (.gmm files) found.")
        else:
            print(f"Found {len(model_files)} enrolled GMM models:")
            for mf in model_files: speaker_id = os.path.basename(mf).replace('.gmm', ''); print(f"- {speaker_id}"); enrolled.append(speaker_id) # Adjust suffix
        print("---------------------------------------------")
        return sorted(enrolled)
    except Exception as e: print(f"Error listing GMM model files: {e}"); return []

# cli_enroll_speaker is no longer needed as enrollment is automatic
# def cli_enroll_speaker(): ...

# cli_verify_speaker remains mostly the same
def cli_verify_speaker():
    """Handles the GMM verification process via CLI prompts using TIMIT TEST data."""
    print("\n===== Speaker Verification (GMM vs TIMIT TEST file) =====")
    enrolled_speakers = list_enrolled_models()
    if not enrolled_speakers: print("Cannot verify: No speakers enrolled yet."); return
    print(f"Enrolled GMM models available for: {', '.join(enrolled_speakers)}")
    claimed_speaker_id = input(f"Enter the claimed Speaker ID (must be enrolled): ").strip()
    if claimed_speaker_id not in enrolled_speakers: print(f"Error: Speaker '{claimed_speaker_id}' is not enrolled."); return
    print("\nSelect a speaker from the TEST set whose file you want to use for verification:")
    test_speaker_list = list_available_speakers_test()
    if not test_speaker_list: print("Cannot verify: No speakers loaded from TEST set."); return
    test_speaker_id = input("Enter Speaker ID from TEST set: ").strip()
    if test_speaker_id not in test_files: print(f"Error: Speaker ID '{test_speaker_id}' not found in TEST set."); return
    test_speaker_paths = test_files[test_speaker_id]
    if not test_speaker_paths: print(f"Error: No files found for speaker '{test_speaker_id}' in TEST set."); return
    test_audio_file = test_speaker_paths[0] # Use the first file for simplicity
    print(f"Using test file: {test_audio_file}")
    print(f"(True speaker: {test_speaker_id})")
    print(f"\nVerifying file '{os.path.basename(test_audio_file)}' against claimed speaker '{claimed_speaker_id}'...")
    score = verify_speaker_gmm(claimed_speaker_id, test_audio_file) # Call GMM verification
    if score is not None:
        print(f"\nVerification Score (Avg Log-Likelihood per Frame): {score:.4f}") # Score interpretation
        print(f"Decision Threshold: {VERIFICATION_THRESHOLD}")
        is_accepted = score > VERIFICATION_THRESHOLD # Higher score is better
        is_genuine_claim = (claimed_speaker_id == test_speaker_id)
        if is_accepted: print("Result: ACCEPTED")
        else: print("Result: REJECTED")
        if is_genuine_claim and not is_accepted: print("    -> False Rejection!")
        elif not is_genuine_claim and is_accepted: print("    -> False Acceptance!")
        elif is_genuine_claim and is_accepted: print("    -> Correct Acceptance")
        elif not is_genuine_claim and not is_accepted: print("    -> Correct Rejection")
    else: print("\nResult: Verification FAILED (Score could not be computed).")
    print("==============================================================")

print("--- Section 7 Defined ---")


# ==============================================================================
# SECTION 8: MAIN EXECUTION LOGIC (Starts the CLI)
# ==============================================================================
# Renumbered section
print("\n--- Running Section 8: Main Execution Logic ---")

def main_cli_loop():
    """Runs the main command-line interface loop."""
    print("\n===========================================")
    print("==== Voice Biometrics GMM System (TIMIT) ====")
    print("       (All TRAIN speakers enrolled)       ") # Note auto-enrollment
    print(f" GMM Models Dir: {MODEL_DIR}")
    print(f" TIMIT Path: {TIMIT_PATH}")
    print(f" Verification Threshold: {VERIFICATION_THRESHOLD}")
    print("===========================================")
    # Checks moved to main execution block

    while True:
        print("\n--- Main Menu ---")
        # Updated menu options (removed enrollment)
        print("1: Verify Speaker (using a file from TIMIT TEST set)")
        print("2: List Available Speakers (TRAIN set)")
        print("3: List Available Speakers (TEST set)")
        print("4: List Enrolled Speakers (GMM Models)")
        print("5: Exit")
        choice = input("Enter your choice (1-5): ").strip()
        if choice == '1': cli_verify_speaker()
        elif choice == '2': list_available_speakers_enroll()
        elif choice == '3': list_available_speakers_test()
        elif choice == '4': list_enrolled_models()
        elif choice == '5': print("Exiting CLI."); break
        else: print("Invalid choice.")

# Start the CLI loop directly when the script is run
if __name__ == '__main__':
     print("Starting script execution...")
     # Perform final checks before starting loop
     if not os.path.exists(TIMIT_PATH): print("\nFINAL CHECK FAILED: TIMIT dataset path does not exist.")
     elif not enroll_files and not test_files: print("\nFINAL CHECK FAILED: No data loaded from TIMIT TRAIN/TEST dirs.")
     elif enrolled_speaker_count == 0: print("\nFINAL CHECK FAILED: No speakers were successfully enrolled in batch process.") # Check if batch enrollment worked
     else:
         main_cli_loop() # Start CLI if checks pass

print("\n--- Script Execution Finished ---")


# ==============================================================================
# SECTION 9: EVALUATION CONSIDERATIONS (Using TIMIT TRAIN/TEST split with GMMs)
# ==============================================================================
# Renumbered section
# Systematic evaluation using TIMIT with GMMs:
# 1.  Enrollment: Already done automatically in Section 5.
# 2.  Verification Trials: Script needed to loop through all test files and speakers.
# 3.  EER Calculation & Threshold: Analyze collected scores.
# 4.  DET Curve: Plot performance.
# ==============================================================================

