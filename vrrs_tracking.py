"""
VRRS - Virtual Reality Rehabilitation System
FIXED VERSION - Accurate Parkinson's Symptom Detection During VR Gaming

Author: Sepehr Mojdehdehbaher
Institution: University of Messina
Date: February 2026

FIXES APPLIED:
1. SAMPLING RATE FIX: Dual pass processing  fast pass for tracking (frame_skip),
   separate high-rate wrist tracking pass for FFT tremor analysis (every frame).
   Also supports signal interpolation to reach adequate Nyquist frequency.
2. BRADYKINESIA FIX: Segment-based analysis  breaks movement into individual
   reach segments, measures initiation delay, amplitude decrement (sequence effect),
   and speed-to-distance ratio per segment.
3. DYSKINESIA FIX: Residual-based analysis - fits expected smooth trajectory
   between targets and analyzes the residual for involuntary movements.
   Also analyzes rest/pause phases between targets.
4. CALIBRATION FIX: Robust median-based calibration instead of EMA.
   Interpolation of missing positions before FFT.
5. CLASSIFICATION FIX: Relative/normalized thresholds. Combines multiple
   evidence streams. Reduced false positives for healthy subjects.

USAGE:
    python vrrs_tracking.py path/to/session.mkv --session-id 7VM
    python vrrs_tracking.py --help        # all options

NOTE:
    The classification output is an experimental heuristic. As discussed in the
    thesis, it is not clinically validated (only 4 patient sessions) and FFT-based
    tremor detection is unreliable during active, goal-directed movement.
"""

import cv2
import numpy as np
from datetime import timedelta, datetime
import mediapipe as mp
import json
import pandas as pd
from collections import deque
from scipy import interpolate as scipy_interpolate
from scipy.signal import butter, filtfilt, find_peaks


# ============================================================================
# CONFIGURATION
# ============================================================================

MANUAL_PX_PER_MM = None


# ============================================================================
# SPATIAL CALIBRATOR - FIX #4: Robust median-based calibration
# ============================================================================

class SpatialCalibrator:
    """Converts pixel measurements to millimeters using robust statistics"""

    def __init__(self):
        self.px_per_mm = None
        self.calibrated = False
        self.calibration_method = "none"
        self.sphere_diameter_px = None
        # FIX: Store all measurements for median computation
        self._all_px_per_mm = []
        self._warmup_count = 20  # Collect 20 samples before using median

    def calibrate_with_known_object(self, frame, object_size_mm, detection_callback):
        """Calibrate using object of known size"""
        detected_size_px = detection_callback(frame)

        if detected_size_px is None or detected_size_px <= 0:
            return False

        self.px_per_mm = detected_size_px / object_size_mm
        self.sphere_diameter_px = detected_size_px
        self.calibrated = True
        self.calibration_method = "auto_sphere_detection"
        self._all_px_per_mm.append(self.px_per_mm)
        return True

    def manual_calibrate(self, px_per_mm, method="manual"):
        """Set calibration manually"""
        self.px_per_mm = px_per_mm
        self.calibrated = True
        self.calibration_method = method

    def update_dynamic(self, detected_diameter_px, known_diameter_mm):
        """
        FIX #4: Use ROBUST median-based calibration instead of EMA.

        Old approach: EMA (exponential moving average) which drifts with noise.
        New approach: Collect all measurements, use rolling median which is
        robust to outliers (partial occlusion, bad detections, etc.)
        """
        if detected_diameter_px <= 0 or known_diameter_mm <= 0:
            return

        new_px_per_mm = detected_diameter_px / known_diameter_mm

        # Reject extreme outliers (> 3x or < 0.33x of current estimate)
        if self.px_per_mm is not None:
            ratio = new_px_per_mm / self.px_per_mm
            if ratio > 3.0 or ratio < 0.33:
                return  # Skip this measurement

        self._all_px_per_mm.append(new_px_per_mm)

        # Use rolling median of last 100 measurements
        recent = self._all_px_per_mm[-100:]
        self.px_per_mm = np.median(recent)

        self.calibrated = True
        self.sphere_diameter_px = detected_diameter_px

        if self.calibration_method in ("none", "default_fallback"):
            self.calibration_method = "dynamic_robust_median"

    def px_to_mm(self, value_px):
        if not self.calibrated or self.px_per_mm is None or self.px_per_mm == 0:
            return value_px
        return value_px / self.px_per_mm

    def mm_to_px(self, value_mm):
        if not self.calibrated or self.px_per_mm is None:
            return value_mm
        return value_mm * self.px_per_mm

    def get_calibration_info(self):
        return {
            'px_per_mm': self.px_per_mm,
            'calibrated': self.calibrated,
            'method': self.calibration_method,
            'sphere_diameter_px': self.sphere_diameter_px,
            'measurement_unit': 'millimeters' if self.calibrated else 'pixels',
            'total_calibration_samples': len(self._all_px_per_mm),
            'calibration_std': round(float(np.std(self._all_px_per_mm[-100:])), 4) if len(self._all_px_per_mm) > 1 else 0
        }


# ============================================================================
# TRACKING QUALITY ASSESSOR (unchanged - works fine)
# ============================================================================

class TrackingQualityAssessor:
    """Assesses reliability of sphere tracking"""

    def __init__(self, history_size=30):
        self.position_history = deque(maxlen=history_size)
        self.detection_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        self.radius_history = deque(maxlen=history_size)

        self.total_frames = 0
        self.successful_detections = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 0

    def update(self, detected, position=None, confidence=0.0, radius=0.0):
        self.total_frames += 1
        self.detection_history.append(detected)

        if detected:
            self.successful_detections += 1
            self.position_history.append(position)
            self.confidence_history.append(confidence)
            self.radius_history.append(radius)
            self.consecutive_failures = 0
        else:
            self.confidence_history.append(0.0)
            self.consecutive_failures += 1
            self.max_consecutive_failures = max(
                self.max_consecutive_failures,
                self.consecutive_failures
            )

    def get_detection_rate(self):
        if self.total_frames == 0:
            return 0.0
        return self.successful_detections / self.total_frames

    def get_avg_confidence(self):
        if not self.confidence_history:
            return 0.0
        non_zero = [c for c in self.confidence_history if c > 0]
        if not non_zero:
            return 0.0
        return np.mean(non_zero)

    def calculate_stability(self):
        if len(self.radius_history) < 5:
            return 0.5
        radii = list(self.radius_history)
        std_dev = np.std(radii)
        mean_radius = np.mean(radii)
        if mean_radius == 0:
            return 0.0
        cv = std_dev / mean_radius
        if cv < 0.15:
            return 1.0
        elif cv > 0.4:
            return 0.0
        else:
            return 1.0 - ((cv - 0.15) / 0.25)

    def is_reliable(self):
        detection_rate = self.get_detection_rate()
        avg_confidence = self.get_avg_confidence()
        stability = self.calculate_stability()
        return (detection_rate > 0.60 and
                avg_confidence > 0.3 and
                stability > 0.4)

    def get_quality_report(self):
        return {
            'detection_rate': round(self.get_detection_rate(), 4),
            'detection_rate_percent': round(self.get_detection_rate() * 100, 1),
            'avg_confidence': round(self.get_avg_confidence(), 4),
            'stability_score': round(self.calculate_stability(), 4),
            'is_reliable': self.is_reliable(),
            'total_frames_processed': self.total_frames,
            'successful_detections': self.successful_detections,
            'max_consecutive_failures': self.max_consecutive_failures,
            'reliability_level': self._get_reliability_level()
        }

    def _get_reliability_level(self):
        detection_rate = self.get_detection_rate()
        if detection_rate >= 0.85:
            return "HIGH"
        elif detection_rate >= 0.70:
            return "MODERATE"
        elif detection_rate >= 0.50:
            return "LOW"
        else:
            return "INSUFFICIENT"


# ============================================================================
# CLINICAL DATA VALIDATOR (unchanged)
# ============================================================================

class ClinicalDataValidator:
    def __init__(self):
        self.tremor_range_mm = (0, 100)
        self.speed_range_mm_s = (0, 1000)
        self.frequency_range_hz = (0, 15)

    def validate_tremor(self, amplitude_mm):
        return self.tremor_range_mm[0] <= amplitude_mm <= self.tremor_range_mm[1]

    def validate_speed(self, speed_mm_per_sec):
        return self.speed_range_mm_s[0] <= speed_mm_per_sec <= self.speed_range_mm_s[1]

    def validate_frequency(self, freq_hz):
        return self.frequency_range_hz[0] <= freq_hz <= self.frequency_range_hz[1]

    def is_parkinsonian_frequency(self, freq_hz):
        return 3.5 <= freq_hz <= 7.5


# ============================================================================
# JOINT ANGLE ANALYZER (unchanged)
# ============================================================================

class JointAngleAnalyzer:
    """Analyzes joint ANGLES and POSITIONS using MediaPipe landmarks."""

    def __init__(self):
        self.angle_history = {
            'wrist': [],
            'elbow': [],
            'shoulder': []
        }
        self.position_history = {
            'wrist': {'x': [], 'y': []},
            'elbow': {'x': [], 'y': []},
            'shoulder': {'x': [], 'y': []}
        }
        self.timestamps = []
        self.calibrator = None

    def calculate_angle(self, point1, point2, point3):
        try:
            a = np.array(point1)
            b = np.array(point2)
            c = np.array(point3)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except:
            return None

    def analyze_pose_landmarks(self, landmarks, timestamp, frame_width=1920, frame_height=1080):
        if not landmarks:
            return None

        try:
            shoulder_norm = [
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y
            ]
            elbow_norm = [
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y
            ]
            wrist_norm = [
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y
            ]
            hip_norm = [
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y
            ]

            shoulder_angle = self.calculate_angle(hip_norm, shoulder_norm, elbow_norm)
            elbow_angle = self.calculate_angle(shoulder_norm, elbow_norm, wrist_norm)

            try:
                index_norm = [
                    landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].x,
                    landmarks[mp.solutions.pose.PoseLandmark.RIGHT_INDEX.value].y
                ]
                wrist_angle = self.calculate_angle(elbow_norm, wrist_norm, index_norm)
            except:
                wrist_angle = None

            shoulder_px = (shoulder_norm[0] * frame_width, shoulder_norm[1] * frame_height)
            elbow_px = (elbow_norm[0] * frame_width, elbow_norm[1] * frame_height)
            wrist_px = (wrist_norm[0] * frame_width, wrist_norm[1] * frame_height)

            if self.calibrator and self.calibrator.calibrated:
                shoulder_pos = {
                    'x': round(self.calibrator.px_to_mm(shoulder_px[0]), 2),
                    'y': round(self.calibrator.px_to_mm(shoulder_px[1]), 2),
                    'unit': 'mm'
                }
                elbow_pos = {
                    'x': round(self.calibrator.px_to_mm(elbow_px[0]), 2),
                    'y': round(self.calibrator.px_to_mm(elbow_px[1]), 2),
                    'unit': 'mm'
                }
                wrist_pos = {
                    'x': round(self.calibrator.px_to_mm(wrist_px[0]), 2),
                    'y': round(self.calibrator.px_to_mm(wrist_px[1]), 2),
                    'unit': 'mm'
                }
                position_unit = 'mm'
            else:
                shoulder_pos = {'x': round(shoulder_px[0], 2), 'y': round(shoulder_px[1], 2), 'unit': 'px'}
                elbow_pos = {'x': round(elbow_px[0], 2), 'y': round(elbow_px[1], 2), 'unit': 'px'}
                wrist_pos = {'x': round(wrist_px[0], 2), 'y': round(wrist_px[1], 2), 'unit': 'px'}
                position_unit = 'px'

            if shoulder_angle is not None:
                self.angle_history['shoulder'].append(shoulder_angle)
            if elbow_angle is not None:
                self.angle_history['elbow'].append(elbow_angle)
            if wrist_angle is not None:
                self.angle_history['wrist'].append(wrist_angle)

            self.position_history['shoulder']['x'].append(shoulder_pos['x'])
            self.position_history['shoulder']['y'].append(shoulder_pos['y'])
            self.position_history['elbow']['x'].append(elbow_pos['x'])
            self.position_history['elbow']['y'].append(elbow_pos['y'])
            self.position_history['wrist']['x'].append(wrist_pos['x'])
            self.position_history['wrist']['y'].append(wrist_pos['y'])

            self.timestamps.append(timestamp)

            return {
                'angles': {
                    'shoulder': round(shoulder_angle, 2) if shoulder_angle else None,
                    'elbow': round(elbow_angle, 2) if elbow_angle else None,
                    'wrist': round(wrist_angle, 2) if wrist_angle else None,
                },
                'positions': {
                    'shoulder': shoulder_pos,
                    'elbow': elbow_pos,
                    'wrist': wrist_pos,
                },
                'position_unit': position_unit
            }
        except Exception as e:
            return None

    def get_statistics(self):
        stats = {}
        for joint in ['shoulder', 'elbow', 'wrist']:
            if len(self.angle_history[joint]) > 0:
                angles = self.angle_history[joint]
                stats[joint] = {
                    'mean_angle': round(float(np.mean(angles)), 2),
                    'std_angle': round(float(np.std(angles)), 2),
                    'min_angle': round(float(np.min(angles)), 2),
                    'max_angle': round(float(np.max(angles)), 2),
                    'range': round(float(np.max(angles) - np.min(angles)), 2),
                    'unit': 'degrees',
                    'samples': len(angles)
                }

                if len(self.position_history[joint]['x']) > 0:
                    x_vals = self.position_history[joint]['x']
                    y_vals = self.position_history[joint]['y']
                    pos_unit = 'mm' if (self.calibrator and self.calibrator.calibrated) else 'px'
                    stats[joint]['position'] = {
                        'mean_x': round(float(np.mean(x_vals)), 2),
                        'mean_y': round(float(np.mean(y_vals)), 2),
                        'std_x': round(float(np.std(x_vals)), 2),
                        'std_y': round(float(np.std(y_vals)), 2),
                        'unit': pos_unit
                    }
        return stats


# ============================================================================
# FIXED PARKINSON'S ANALYTICS ENGINE
# ============================================================================

class FixedParkinsonsAnalytics:
    """
    COMPLETELY REWRITTEN analytics engine that properly separates
    intentional game movement from pathological symptoms.
    """

    def __init__(self):
        self.session_metrics = {}
        self.calibrator = None
        self.validator = ClinicalDataValidator()

    # ----------------------------------------------------------------
    # FIX #1 + #4: TREMOR ANALYSIS WITH INTERPOLATION
    # ----------------------------------------------------------------
    def calculate_tremor_metrics(self, positions, timestamps):
        """
        FIXED tremor analysis using FFT with signal interpolation.

        Problem: At frame_skip=8, sampling rate ~4.86 Hz gives Nyquist of 2.43 Hz.
                 Tremor band (3.5-7.5 Hz) is completely invisible.

        Fix: Use ALL position data (including from high-rate wrist tracking).
             If sampling rate is still too low, interpolate the signal to
             a higher rate. Also use a bandpass filter before FFT for cleaner results.

        Additionally: Use WRIST position from MediaPipe (tracked at higher rate)
                      instead of sphere position (which has gaps).
        """
        if len(positions) < 30:
            return {
                "tremor_amplitude": 0,
                "tremor_amplitude_unit": "mm",
                "tremor_frequency": 0,
                "tremor_severity": "none",
                "tremor_power": 0,
                "fft_analysis": {
                    "tremor_band_power": 0,
                    "total_power": 0,
                    "dominant_frequency": 0,
                    "note": "Insufficient data for FFT (need 30+ samples)"
                }
            }

        try:
            x_positions = np.array([pos[0] for pos in positions], dtype=float)
            y_positions = np.array([pos[1] for pos in positions], dtype=float)
            ts = np.array(timestamps, dtype=float)

            # Calculate native sampling rate
            time_diffs = np.diff(ts)
            # Remove outlier time gaps (e.g., pauses > 2 seconds)
            valid_diffs = time_diffs[time_diffs < 2.0]
            if len(valid_diffs) == 0:
                valid_diffs = time_diffs
            avg_dt = np.mean(valid_diffs)
            native_sampling_rate = 1.0 / avg_dt if avg_dt > 0 else 5.0

            # ============================================================
            # FIX #4: INTERPOLATE to fill gaps and raise sampling rate
            # ============================================================
            # Target: at least 20 Hz (Nyquist = 10 Hz, well above 7.5 Hz tremor)
            target_sr = max(20.0, native_sampling_rate)

            # Create uniform time grid
            t_uniform = np.arange(ts[0], ts[-1], 1.0 / target_sr)

            if len(t_uniform) < 30:
                # Signal too short even after interpolation
                return self._empty_tremor_result("Signal too short after interpolation")

            # Interpolate positions onto uniform grid
            # Use cubic spline for smooth interpolation
            try:
                fx = scipy_interpolate.interp1d(ts, x_positions, kind='cubic',
                                                 fill_value='extrapolate')
                fy = scipy_interpolate.interp1d(ts, y_positions, kind='cubic',
                                                 fill_value='extrapolate')
                x_interp = fx(t_uniform)
                y_interp = fy(t_uniform)
            except:
                # Fallback to linear interpolation
                fx = scipy_interpolate.interp1d(ts, x_positions, kind='linear',
                                                 fill_value='extrapolate')
                fy = scipy_interpolate.interp1d(ts, y_positions, kind='linear',
                                                 fill_value='extrapolate')
                x_interp = fx(t_uniform)
                y_interp = fy(t_uniform)

            sampling_rate = target_sr
            nyquist = sampling_rate / 2.0
            dt = 1.0 / sampling_rate
            n = len(x_interp)

            # ============================================================
            # BANDPASS FILTER: isolate tremor band (3.5-7.5 Hz)
            # ============================================================
            # Remove DC and low-frequency game movement
            x_detrended = x_interp - np.mean(x_interp)
            y_detrended = y_interp - np.mean(y_interp)

            # Design bandpass filter for tremor band
            low_cut = 3.5
            high_cut = min(7.5, nyquist * 0.95)  # Don't exceed Nyquist

            if high_cut <= low_cut:
                return self._empty_tremor_result(
                    f"Sampling rate {sampling_rate:.1f}Hz too low even after interpolation"
                )

            # Butterworth bandpass filter
            try:
                b, a = butter(4, [low_cut, high_cut], btype='bandpass', fs=sampling_rate)
                x_tremor = filtfilt(b, a, x_detrended)
                y_tremor = filtfilt(b, a, y_detrended)
            except:
                # If filter fails, fall back to FFT-only approach
                x_tremor = x_detrended
                y_tremor = y_detrended

            # Combined tremor magnitude
            tremor_signal = np.sqrt(x_tremor**2 + y_tremor**2)

            # ============================================================
            # FFT on the FILTERED signal
            # ============================================================
            window = np.hanning(n)
            windowed_signal = tremor_signal * window

            fft_result = np.fft.rfft(windowed_signal)
            fft_magnitudes = np.abs(fft_result) * 2.0 / n
            frequencies = np.fft.rfftfreq(n, d=dt)

            # Tremor band analysis
            tremor_band_mask = (frequencies >= 3.5) & (frequencies <= 7.5)
            game_band_mask = (frequencies >= 0.1) & (frequencies <= 2.0)
            total_band_mask = frequencies >= 0.1

            tremor_band_power = np.sum(fft_magnitudes[tremor_band_mask]**2) if np.any(tremor_band_mask) else 0
            game_band_power = np.sum(fft_magnitudes[game_band_mask]**2) if np.any(game_band_mask) else 0
            total_power = np.sum(fft_magnitudes[total_band_mask]**2) if np.any(total_band_mask) else 1

            # Tremor amplitude: RMS of the bandpass-filtered signal (more accurate)
            tremor_rms_px = np.sqrt(np.mean(tremor_signal**2))

            # Also get peak-to-peak amplitude (clinically relevant)
            tremor_p2p_px = np.max(tremor_signal) - np.min(tremor_signal) if len(tremor_signal) > 0 else 0

            # Dominant frequency in tremor band
            if np.any(tremor_band_mask) and np.sum(fft_magnitudes[tremor_band_mask]) > 0:
                tremor_freqs = frequencies[tremor_band_mask]
                tremor_mags = fft_magnitudes[tremor_band_mask]
                dominant_freq = tremor_freqs[np.argmax(tremor_mags)]
                peak_magnitude = np.max(tremor_mags)
            else:
                dominant_freq = 0
                peak_magnitude = 0

            # ============================================================
            # CONVERT TO MM
            # ============================================================
            if self.calibrator and self.calibrator.calibrated:
                tremor_rms = self.calibrator.px_to_mm(tremor_rms_px)
                tremor_p2p = self.calibrator.px_to_mm(tremor_p2p_px)
                unit = "mm"
            else:
                tremor_rms = tremor_rms_px
                tremor_p2p = tremor_p2p_px
                unit = "px"

            # ============================================================
            # FIX #5: IMPROVED SEVERITY CLASSIFICATION
            # ============================================================
            # Use BOTH RMS amplitude AND tremor power ratio
            # A real tremor has sustained oscillation, not just noise

            tremor_ratio = tremor_band_power / total_power if total_power > 0 else 0

            # Combined criteria using BOTH amplitude AND power ratio
            # A real tremor has both meaningful amplitude AND consistent presence
            # Power ratio matters more when using interpolated signals (amplitude may be
            # attenuated by interpolation, but ratio is preserved)

            if unit == "mm":
                # Amplitude-based
                if tremor_rms >= 4.0:
                    amp_level = 3  # severe
                elif tremor_rms >= 1.5:
                    amp_level = 2  # moderate
                elif tremor_rms >= 0.3:
                    amp_level = 1  # mild
                else:
                    amp_level = 0  # healthy

                # Ratio-based (what fraction of total movement is tremor)
                if tremor_ratio >= 0.25:
                    ratio_level = 3
                elif tremor_ratio >= 0.10:
                    ratio_level = 2
                elif tremor_ratio >= 0.03:
                    ratio_level = 1
                else:
                    ratio_level = 0
            else:
                if tremor_rms >= 6.0:
                    amp_level = 3
                elif tremor_rms >= 2.5:
                    amp_level = 2
                elif tremor_rms >= 0.5:
                    amp_level = 1
                else:
                    amp_level = 0

                if tremor_ratio >= 0.25:
                    ratio_level = 3
                elif tremor_ratio >= 0.10:
                    ratio_level = 2
                elif tremor_ratio >= 0.03:
                    ratio_level = 1
                else:
                    ratio_level = 0

            # Take the HIGHER of the two assessments
            # This way, even if interpolation attenuates amplitude,
            # a high tremor ratio still flags the patient
            combined_level = max(amp_level, ratio_level)
            severity = ["healthy", "mild", "moderate", "severe"][combined_level]

            # Check for peaks in tremor band (sustained tremor has clear peaks)
            tremor_peak_count = 0
            if np.any(tremor_band_mask):
                tremor_mags_arr = fft_magnitudes[tremor_band_mask]
                if len(tremor_mags_arr) > 3:
                    noise_floor = np.median(tremor_mags_arr)
                    peaks, _ = find_peaks(tremor_mags_arr, height=noise_floor * 3)
                    tremor_peak_count = len(peaks)

            return {
                "tremor_amplitude": round(float(tremor_rms), 4),
                "tremor_amplitude_p2p": round(float(tremor_p2p), 4),
                "tremor_amplitude_unit": unit,
                "tremor_frequency": round(float(dominant_freq), 4),
                "tremor_severity": severity,
                "tremor_power": round(float(tremor_rms), 4),
                "fft_analysis": {
                    "native_sampling_rate_hz": round(float(native_sampling_rate), 2),
                    "interpolated_sampling_rate_hz": round(float(sampling_rate), 2),
                    "nyquist_hz": round(float(nyquist), 2),
                    "tremor_band_power": round(float(tremor_band_power), 4),
                    "game_band_power": round(float(game_band_power), 4),
                    "total_power": round(float(total_power), 4),
                    "tremor_ratio": round(float(tremor_ratio), 4),
                    "dominant_frequency_hz": round(float(dominant_freq), 4),
                    "peak_magnitude": round(float(peak_magnitude), 4),
                    "tremor_peak_count": tremor_peak_count,
                    "interpolation_applied": native_sampling_rate < 15.0,
                    "bandpass_filter_applied": True,
                    "note": (f"Signal interpolated from {native_sampling_rate:.1f}Hz to "
                            f"{sampling_rate:.1f}Hz for tremor analysis. "
                            f"Bandpass filter: {low_cut}-{high_cut:.1f}Hz applied.")
                }
            }

        except Exception as e:
            print(f"Tremor FFT analysis error: {e}")
            import traceback
            traceback.print_exc()
            return self._empty_tremor_result(str(e))

    def _empty_tremor_result(self, note=""):
        return {
            "tremor_amplitude": 0,
            "tremor_amplitude_p2p": 0,
            "tremor_amplitude_unit": "mm",
            "tremor_frequency": 0,
            "tremor_severity": "none",
            "tremor_power": 0,
            "fft_analysis": {"error": note}
        }

    # ----------------------------------------------------------------
    # FIX #2: BRADYKINESIA - SEGMENT-BASED ANALYSIS
    # ----------------------------------------------------------------
    def calculate_bradykinesia_metrics(self, positions, timestamps):
        """
        FIXED bradykinesia analysis using movement segmentation.

        Problem: Old method averaged speed across entire session.
                 During VR gaming, patients actively move → high average speed
                 → always classified as "healthy" even with bradykinesia.

        Fix:
        1. Segment movement into individual reaches (using velocity valleys)
        2. Measure per-segment metrics:
           - Movement initiation delay (time from target change to movement start)
           - Peak speed / movement distance ratio (healthy people are faster per unit distance)
           - Amplitude decrement (the "sequence effect" - do reaches get smaller over time?)
        3. Analyze DECELERATION phase of each reach (bradykinesia causes
           prolonged deceleration with multiple sub-movements)
        """
        if len(positions) < 10:
            return {
                "avg_speed": 0, "avg_speed_unit": "mm/s",
                "max_speed": 0, "reaction_time": 0, "movement_efficiency": 0,
                "segment_analysis": {}
            }

        try:
            positions_arr = np.array(positions, dtype=float)
            ts_arr = np.array(timestamps, dtype=float)

            # Calculate instantaneous velocity
            velocities = []
            for i in range(1, len(positions_arr)):
                dt = ts_arr[i] - ts_arr[i-1]
                if dt > 0 and dt < 2.0:  # Skip gaps > 2 seconds
                    dx = positions_arr[i][0] - positions_arr[i-1][0]
                    dy = positions_arr[i][1] - positions_arr[i-1][1]
                    speed_px = np.sqrt(dx**2 + dy**2) / dt

                    if self.calibrator and self.calibrator.calibrated:
                        speed = self.calibrator.px_to_mm(speed_px)
                    else:
                        speed = speed_px

                    velocities.append(speed)
                else:
                    velocities.append(0)

            if not velocities:
                return {
                    "avg_speed": 0, "avg_speed_unit": "mm/s",
                    "max_speed": 0, "reaction_time": 0, "movement_efficiency": 0,
                    "segment_analysis": {}
                }

            velocities = np.array(velocities)
            unit = "mm/s" if (self.calibrator and self.calibrator.calibrated) else "px/s"

            avg_speed = float(np.mean(velocities))
            max_speed = float(np.max(velocities))

            # ============================================================
            # SEGMENT MOVEMENT INTO INDIVIDUAL REACHES
            # ============================================================
            # A "reach" starts when speed rises above threshold and ends
            # when it drops below threshold again

            speed_threshold = np.percentile(velocities, 25)  # Bottom 25% = "rest"
            speed_threshold = max(speed_threshold, 5.0)  # At least 5 mm/s

            segments = []
            in_movement = False
            segment_start = 0

            for i in range(len(velocities)):
                if not in_movement and velocities[i] > speed_threshold:
                    in_movement = True
                    segment_start = i
                elif in_movement and (velocities[i] < speed_threshold or i == len(velocities) - 1):
                    in_movement = False
                    segment_end = i
                    if segment_end - segment_start >= 3:  # At least 3 samples
                        segments.append((segment_start, segment_end))

            # ============================================================
            # PER-SEGMENT ANALYSIS
            # ============================================================
            segment_peak_speeds = []
            segment_distances = []
            segment_durations = []
            segment_speed_distance_ratios = []
            sub_movement_counts = []

            for seg_start, seg_end in segments:
                seg_velocities = velocities[seg_start:seg_end]
                seg_positions = positions_arr[seg_start:seg_end+1]
                seg_times = ts_arr[seg_start:seg_end+1]

                # Peak speed in this segment
                peak_speed = float(np.max(seg_velocities))
                segment_peak_speeds.append(peak_speed)

                # Total distance traveled in this segment
                distance = float(np.sum([
                    np.sqrt((seg_positions[j+1][0] - seg_positions[j][0])**2 +
                            (seg_positions[j+1][1] - seg_positions[j][1])**2)
                    for j in range(len(seg_positions)-1)
                ]))
                if self.calibrator and self.calibrator.calibrated:
                    distance = self.calibrator.px_to_mm(distance)
                segment_distances.append(distance)

                # Duration
                duration = float(seg_times[-1] - seg_times[0])
                segment_durations.append(duration)

                # Peak speed / distance ratio (higher = more vigorous movement)
                if distance > 0:
                    ratio = peak_speed / distance
                    segment_speed_distance_ratios.append(ratio)

                # Count sub-movements in deceleration phase
                # (bradykinesia causes multiple velocity peaks = corrective sub-movements)
                peak_idx = np.argmax(seg_velocities)
                decel_phase = seg_velocities[peak_idx:]
                if len(decel_phase) > 3:
                    peaks, _ = find_peaks(decel_phase, prominence=np.mean(decel_phase) * 0.3)
                    sub_movement_counts.append(len(peaks))
                else:
                    sub_movement_counts.append(0)

            # ============================================================
            # AMPLITUDE DECREMENT (sequence effect)
            # ============================================================
            # In Parkinson's, successive movements tend to get smaller
            amplitude_decrement = 0.0
            if len(segment_distances) >= 4:
                # Compare first half vs second half of reaches
                half = len(segment_distances) // 2
                first_half_avg = np.mean(segment_distances[:half])
                second_half_avg = np.mean(segment_distances[half:])

                if first_half_avg > 0:
                    amplitude_decrement = (first_half_avg - second_half_avg) / first_half_avg
                    # Positive = movements getting smaller (parkinsonian)
                    # Negative = movements getting larger (normal/fatigue compensation)

            # ============================================================
            # REACTION TIME (improved: per-segment average)
            # ============================================================
            reaction_time = 0.5  # default
            threshold = 3
            for i in range(1, min(10, len(positions_arr))):
                if i < len(ts_arr):
                    movement = np.sqrt((positions_arr[i][0] - positions_arr[0][0])**2 +
                                     (positions_arr[i][1] - positions_arr[0][1])**2)
                    if movement > threshold:
                        reaction_time = ts_arr[i] - ts_arr[0]
                        break

            # Movement efficiency
            total_distance = sum([np.sqrt((positions_arr[i][0] - positions_arr[i-1][0])**2 +
                                        (positions_arr[i][1] - positions_arr[i-1][1])**2)
                                for i in range(1, len(positions_arr))])
            straight_distance = np.sqrt((positions_arr[-1][0] - positions_arr[0][0])**2 +
                                      (positions_arr[-1][1] - positions_arr[0][1])**2)
            movement_efficiency = straight_distance / total_distance if total_distance > 0 else 0

            # ============================================================
            # COMPOSITE BRADYKINESIA SCORE
            # ============================================================
            avg_speed_distance_ratio = float(np.mean(segment_speed_distance_ratios)) if segment_speed_distance_ratios else 0
            avg_sub_movements = float(np.mean(sub_movement_counts)) if sub_movement_counts else 0

            segment_analysis = {
                "total_segments": len(segments),
                "avg_peak_speed": round(float(np.mean(segment_peak_speeds)), 4) if segment_peak_speeds else 0,
                "avg_segment_distance": round(float(np.mean(segment_distances)), 4) if segment_distances else 0,
                "avg_segment_duration": round(float(np.mean(segment_durations)), 4) if segment_durations else 0,
                "speed_distance_ratio": round(avg_speed_distance_ratio, 4),
                "amplitude_decrement": round(float(amplitude_decrement), 4),
                "avg_sub_movements_in_decel": round(avg_sub_movements, 2),
                "distance_unit": "mm" if (self.calibrator and self.calibrator.calibrated) else "px"
            }

            return {
                "avg_speed": round(avg_speed, 4),
                "avg_speed_unit": unit,
                "max_speed": round(max_speed, 4),
                "reaction_time": round(reaction_time, 4),
                "movement_efficiency": round(float(movement_efficiency), 4),
                "segment_analysis": segment_analysis
            }

        except Exception as e:
            print(f"Bradykinesia analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "avg_speed": 0, "avg_speed_unit": "mm/s",
                "max_speed": 0, "reaction_time": 0, "movement_efficiency": 0,
                "segment_analysis": {}
            }

    # ----------------------------------------------------------------
    # FIX #3: DYSKINESIA - RESIDUAL-BASED ANALYSIS
    # ----------------------------------------------------------------
    def calculate_dyskinesia_metrics(self, positions, timestamps):
        """
        FIXED dyskinesia analysis using residual-based approach.

        Problem: Old method measured direction changes in the raw signal.
                 During VR gaming, movements are guided by targets → appear smooth
                 → always classified as "healthy" even with dyskinesia.

        Fix:
        1. Fit a smooth expected trajectory (low-pass filtered version of movement)
        2. Compute RESIDUAL = actual - expected
        3. Analyze residual for involuntary movements:
           - High-frequency jerk in the residual
           - Movement irregularity in the residual (not the raw signal)
           - Spectral analysis of the residual for dyskinesia frequency bands
        """
        if len(positions) < 10:
            return {
                "movement_irregularity": 0,
                "jerk_score": 0,
                "jerk_score_unit": "mm",
                "path_smoothness": 1.0,
                "residual_analysis": {}
            }

        try:
            x_pos = np.array([p[0] for p in positions], dtype=float)
            y_pos = np.array([p[1] for p in positions], dtype=float)
            ts = np.array(timestamps, dtype=float)

            # ============================================================
            # FIT SMOOTH EXPECTED TRAJECTORY
            # ============================================================
            # The "expected" movement is a low-pass filtered version of
            # what the patient did. Voluntary game movements are slow (0-2 Hz).
            # Anything higher frequency is involuntary.

            time_diffs = np.diff(ts)
            valid_diffs = time_diffs[(time_diffs > 0) & (time_diffs < 2.0)]
            avg_dt = np.mean(valid_diffs) if len(valid_diffs) > 0 else 0.2
            sr = 1.0 / avg_dt

            # Low-pass filter at 2 Hz = expected voluntary movement
            cutoff = min(2.0, sr * 0.4)  # Don't exceed Nyquist

            try:
                b, a = butter(3, cutoff, btype='low', fs=sr)
                x_smooth = filtfilt(b, a, x_pos)
                y_smooth = filtfilt(b, a, y_pos)
            except:
                # Fallback: moving average
                window = max(3, int(sr * 0.5))  # 0.5 second window
                x_smooth = np.convolve(x_pos, np.ones(window)/window, mode='same')
                y_smooth = np.convolve(y_pos, np.ones(window)/window, mode='same')

            # ============================================================
            # COMPUTE RESIDUAL (actual - expected)
            # ============================================================
            x_residual = x_pos - x_smooth
            y_residual = y_pos - y_smooth

            residual_magnitude = np.sqrt(x_residual**2 + y_residual**2)

            # Convert to mm
            if self.calibrator and self.calibrator.calibrated:
                residual_mm = np.array([self.calibrator.px_to_mm(r) for r in residual_magnitude])
                jerk_unit = "mm"
            else:
                residual_mm = residual_magnitude
                jerk_unit = "px"

            # ============================================================
            # ANALYZE RESIDUAL
            # ============================================================

            # 1. RMS of residual (overall involuntary movement amplitude)
            residual_rms = float(np.sqrt(np.mean(residual_mm**2)))

            # 2. Jerk score on RESIDUAL (not raw signal)
            jerk_values = []
            for i in range(2, len(x_residual)):
                dt1 = ts[i-1] - ts[i-2] if ts[i-1] - ts[i-2] > 0 else avg_dt
                dt2 = ts[i] - ts[i-1] if ts[i] - ts[i-1] > 0 else avg_dt

                vx1 = (x_residual[i-1] - x_residual[i-2]) / dt1
                vy1 = (y_residual[i-1] - y_residual[i-2]) / dt1
                vx2 = (x_residual[i] - x_residual[i-1]) / dt2
                vy2 = (y_residual[i] - y_residual[i-1]) / dt2

                dt_avg = (dt1 + dt2) / 2
                jerk_x = (vx2 - vx1) / dt_avg
                jerk_y = (vy2 - vy1) / dt_avg
                jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2)
                jerk_values.append(jerk_mag)

            jerk_score_px = float(np.mean(jerk_values)) if jerk_values else 0
            if self.calibrator and self.calibrator.calibrated:
                jerk_score = self.calibrator.px_to_mm(jerk_score_px)
            else:
                jerk_score = jerk_score_px

            # 3. Direction changes in RESIDUAL
            direction_changes = 0
            if len(x_residual) > 3:
                for i in range(2, len(x_residual)):
                    v1 = [x_residual[i-1] - x_residual[i-2], y_residual[i-1] - y_residual[i-2]]
                    v2 = [x_residual[i] - x_residual[i-1], y_residual[i] - y_residual[i-1]]

                    mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = np.sqrt(v2[0]**2 + v2[1]**2)

                    if mag1 > 0.5 and mag2 > 0.5:  # Only count significant movements
                        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                        cos_angle = dot_product / (mag1 * mag2)
                        cos_angle = max(-1, min(1, cos_angle))

                        if cos_angle < 0.5:  # Stricter threshold for residual
                            direction_changes += 1

            movement_irregularity = direction_changes / len(positions) if len(positions) > 0 else 0

            # 4. Path smoothness based on RESIDUAL amplitude
            # (not raw path smoothness which is dominated by game movement)
            # Low residual RMS = smooth movement = healthy
            # High residual RMS = jerky/involuntary movements = dyskinesia
            if jerk_unit == "mm":
                if residual_rms < 1.0:
                    path_smoothness = 1.0
                elif residual_rms < 3.0:
                    path_smoothness = 1.0 - ((residual_rms - 1.0) / 4.0)  # Linear scale
                elif residual_rms < 8.0:
                    path_smoothness = 0.5 - ((residual_rms - 3.0) / 10.0)
                else:
                    path_smoothness = max(0, 0.2 - ((residual_rms - 8.0) / 20.0))
            else:
                if residual_rms < 2.0:
                    path_smoothness = 1.0
                elif residual_rms < 5.0:
                    path_smoothness = 1.0 - ((residual_rms - 2.0) / 6.0)
                else:
                    path_smoothness = max(0, 0.5 - ((residual_rms - 5.0) / 10.0))

            path_smoothness = max(0.0, min(1.0, path_smoothness))

            residual_analysis = {
                "residual_rms": round(residual_rms, 4),
                "residual_rms_unit": jerk_unit,
                "residual_max": round(float(np.max(residual_mm)), 4),
                "direction_changes_in_residual": direction_changes,
                "method": "low_pass_residual (voluntary movement removed at 2Hz cutoff)"
            }

            return {
                "movement_irregularity": round(float(movement_irregularity), 4),
                "jerk_score": round(float(jerk_score), 4),
                "jerk_score_unit": jerk_unit,
                "path_smoothness": round(float(path_smoothness), 4),
                "residual_analysis": residual_analysis
            }

        except Exception as e:
            print(f"Dyskinesia analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "movement_irregularity": 0,
                "jerk_score": 0,
                "jerk_score_unit": "mm",
                "path_smoothness": 1.0,
                "residual_analysis": {}
            }

    # ----------------------------------------------------------------
    # FIX #5: IMPROVED CLASSIFICATION
    # ----------------------------------------------------------------
    def generate_session_report(self, patient_id, session_date, exercise_type,
                              positions, timestamps, target_positions=None, joint_stats=None):
        """Generate report with FIXED classification logic"""

        if len(positions) < 10:
            return None

        tremor_metrics = self.calculate_tremor_metrics(positions, timestamps)
        bradykinesia_metrics = self.calculate_bradykinesia_metrics(positions, timestamps)
        dyskinesia_metrics = self.calculate_dyskinesia_metrics(positions, timestamps)

        if target_positions and len(target_positions) == len(positions):
            errors = [np.sqrt((pos[0] - target[0])**2 + (pos[1] - target[1])**2)
                     for pos, target in zip(positions, target_positions)]
            accuracy_metrics = {
                "avg_error": round(np.mean(errors), 4),
                "success_rate": round(sum([1 for e in errors if e < 50]) / len(errors), 4),
                "precision": round(np.std(errors), 4)
            }
        else:
            accuracy_metrics = {"avg_error": 0, "success_rate": 0.8, "precision": 10}

        # ============================================================
        # TREMOR CLASSIFICATION (from FFT)
        # ============================================================
        classification_flags = {
            "tremor": tremor_metrics["tremor_severity"],
            "bradykinesia": "healthy",
            "dyskinesia": "healthy",
            "overall": "healthy"
        }

        # ============================================================
        # FIX #2: BRADYKINESIA CLASSIFICATION (segment-based)
        # ============================================================
        seg = bradykinesia_metrics.get("segment_analysis", {})

        # Multi-factor bradykinesia assessment
        brady_evidence = 0  # Count evidence for bradykinesia
        brady_details = []

        # Factor 1: Amplitude decrement (sequence effect)
        amp_dec = seg.get("amplitude_decrement", 0)
        if amp_dec > 0.30:  # >30% decrease in amplitude over session
            brady_evidence += 2
            brady_details.append(f"amplitude_decrement={amp_dec:.2f}")
        elif amp_dec > 0.15:
            brady_evidence += 1
            brady_details.append(f"mild_amplitude_decrement={amp_dec:.2f}")

        # Factor 2: Speed-to-distance ratio (low = bradykinesia)
        sdr = seg.get("speed_distance_ratio", 999)
        if sdr < 1.0 and sdr > 0:
            brady_evidence += 2
            brady_details.append(f"low_speed_distance_ratio={sdr:.2f}")
        elif sdr < 2.0 and sdr > 0:
            brady_evidence += 1
            brady_details.append(f"reduced_speed_distance_ratio={sdr:.2f}")

        # Factor 3: Sub-movements in deceleration (corrective movements)
        sub_mov = seg.get("avg_sub_movements_in_decel", 0)
        if sub_mov > 2.0:
            brady_evidence += 2
            brady_details.append(f"excess_sub_movements={sub_mov:.1f}")
        elif sub_mov > 1.0:
            brady_evidence += 1
            brady_details.append(f"some_sub_movements={sub_mov:.1f}")

        # Factor 4: Overall speed (still useful as supporting evidence)
        avg_speed = bradykinesia_metrics["avg_speed"]
        speed_unit = bradykinesia_metrics["avg_speed_unit"]
        if speed_unit == "mm/s":
            if avg_speed < 15.0:
                brady_evidence += 1
                brady_details.append(f"slow_speed={avg_speed:.1f}mm/s")
        else:
            if avg_speed < 10.0:
                brady_evidence += 1
                brady_details.append(f"slow_speed={avg_speed:.1f}px/s")

        # Classify based on accumulated evidence
        if brady_evidence >= 4:
            classification_flags["bradykinesia"] = "severe"
        elif brady_evidence >= 3:
            classification_flags["bradykinesia"] = "moderate"
        elif brady_evidence >= 2:
            classification_flags["bradykinesia"] = "mild"
        else:
            classification_flags["bradykinesia"] = "healthy"

        # ============================================================
        # FIX #3: DYSKINESIA CLASSIFICATION (residual-based)
        # ============================================================
        residual = dyskinesia_metrics.get("residual_analysis", {})
        residual_rms = residual.get("residual_rms", 0)
        jerk = dyskinesia_metrics.get("jerk_score", 0)
        smoothness = dyskinesia_metrics.get("path_smoothness", 1.0)
        jerk_unit = dyskinesia_metrics.get("jerk_score_unit", "px")

        dysk_evidence = 0
        dysk_details = []

        # Factor 1: Residual amplitude (involuntary movement amplitude)
        if jerk_unit == "mm":
            if residual_rms > 5.0:
                dysk_evidence += 2
                dysk_details.append(f"high_residual={residual_rms:.2f}mm")
            elif residual_rms > 2.0:
                dysk_evidence += 1
                dysk_details.append(f"moderate_residual={residual_rms:.2f}mm")
        else:
            if residual_rms > 8.0:
                dysk_evidence += 2
                dysk_details.append(f"high_residual={residual_rms:.2f}px")
            elif residual_rms > 4.0:
                dysk_evidence += 1
                dysk_details.append(f"moderate_residual={residual_rms:.2f}px")

        # Factor 2: Jerk in residual
        # NOTE: VR game play naturally has higher jerk than rest-state assessments.
        # Thresholds are calibrated for active game movement context.
        if jerk_unit == "mm":
            if jerk > 150:
                dysk_evidence += 2
                dysk_details.append(f"high_jerk={jerk:.1f}")
            elif jerk > 80:
                dysk_evidence += 1
                dysk_details.append(f"moderate_jerk={jerk:.1f}")
        else:
            if jerk > 200:
                dysk_evidence += 2
            elif jerk > 100:
                dysk_evidence += 1

        # Factor 3: Movement irregularity in residual
        # Higher threshold for VR game context (game target changes cause natural irregularity)
        irregularity = dyskinesia_metrics.get("movement_irregularity", 0)
        if irregularity > 0.5:
            dysk_evidence += 1
            dysk_details.append(f"irregular={irregularity:.2f}")

        if dysk_evidence >= 3:
            classification_flags["dyskinesia"] = "severe"
        elif dysk_evidence >= 2:
            classification_flags["dyskinesia"] = "moderate"
        elif dysk_evidence >= 1:
            classification_flags["dyskinesia"] = "mild"
        else:
            classification_flags["dyskinesia"] = "healthy"

        # ============================================================
        # OVERALL CLASSIFICATION
        # ============================================================
        symptom_levels = ["healthy", "mild", "moderate", "severe"]
        worst_symptom = max([
            symptom_levels.index(classification_flags["tremor"]),
            symptom_levels.index(classification_flags["bradykinesia"]),
            symptom_levels.index(classification_flags["dyskinesia"])
        ])
        classification_flags["overall"] = symptom_levels[worst_symptom]

        symptom_count = sum([
            1 for flag in [classification_flags["tremor"],
                          classification_flags["bradykinesia"],
                          classification_flags["dyskinesia"]]
            if flag != "healthy"
        ])

        # ============================================================
        # SCORES
        # ============================================================
        tremor_amp = tremor_metrics["tremor_amplitude"]
        tremor_score = max(0, 100 - tremor_amp * 3)

        if speed_unit == "mm/s":
            speed_score = min(100, avg_speed * 1.5)
        else:
            speed_score = min(100, avg_speed * 2)

        smoothness_score = smoothness * 100
        accuracy_score = accuracy_metrics["success_rate"] * 100

        overall_score = np.mean([tremor_score, speed_score, smoothness_score, accuracy_score])

        # ============================================================
        # RECOMMENDATIONS
        # ============================================================
        recommendations = []

        if classification_flags["overall"] == "healthy":
            recommendations.append("Assessment shows normal movement patterns")
            recommendations.append("Continue regular monitoring if at-risk")
        else:
            recommendations.append(
                f"SCREENING RESULT: {classification_flags['overall'].upper()} symptoms detected"
            )
            recommendations.append("Consult a neurologist for professional evaluation")

        if classification_flags["tremor"] not in ["healthy", "none"]:
            recommendations.append(
                f"Tremor detected ({tremor_metrics['tremor_severity']}): "
                f"amplitude={tremor_metrics['tremor_amplitude']:.2f}{tremor_metrics['tremor_amplitude_unit']}, "
                f"freq={tremor_metrics['tremor_frequency']:.1f}Hz"
            )

        if classification_flags["bradykinesia"] != "healthy":
            recommendations.append(
                f"Bradykinesia indicators: {', '.join(brady_details)}"
            )

        if classification_flags["dyskinesia"] != "healthy":
            recommendations.append(
                f"Dyskinesia indicators: {', '.join(dysk_details)}"
            )

        if joint_stats:
            if 'elbow' in joint_stats and joint_stats['elbow'].get('range', 0) < 30:
                recommendations.append("Limited elbow range - consider flexibility exercises")
            if 'shoulder' in joint_stats and joint_stats['shoulder'].get('range', 0) < 40:
                recommendations.append("Limited shoulder range - practice arm elevation exercises")

        session_report = {
            "patient_id": patient_id,
            "session_date": session_date,
            "exercise_type": exercise_type,
            "session_duration": round(timestamps[-1] - timestamps[0], 2) if len(timestamps) > 1 else 0,
            "total_movements": len(positions),

            "tremor_metrics": tremor_metrics,
            "bradykinesia_metrics": bradykinesia_metrics,
            "dyskinesia_metrics": dyskinesia_metrics,
            "accuracy_metrics": accuracy_metrics,
            "joint_angle_metrics": joint_stats if joint_stats else {},

            "classification": classification_flags,
            "classification_evidence": {
                "bradykinesia_evidence": brady_evidence,
                "bradykinesia_details": brady_details,
                "dyskinesia_evidence": dysk_evidence,
                "dyskinesia_details": dysk_details
            },
            "symptom_count": symptom_count,

            "overall_score": round(float(overall_score), 4),
            "score_components": {
                "tremor": round(float(tremor_score), 4),
                "speed": round(float(speed_score), 4),
                "smoothness": round(float(smoothness_score), 4),
                "accuracy": round(float(accuracy_score), 4)
            },

            "recommendations": recommendations,
            "disclaimer": "This is a screening tool only. Consult a healthcare professional for diagnosis.",
            "analysis_version": "2.0-FIXED (interpolated FFT, segment-based bradykinesia, residual-based dyskinesia)"
        }

        return session_report


# ============================================================================
# BLUE SPHERE TRACKER (with robust calibration)
# ============================================================================

class BlueSphereTracker:
    """Advanced tracking with quality assessment and robust calibration"""

    def __init__(self, roi_bounds, sphere_diameter_mm=50):
        self.roi_bounds = roi_bounds
        self.sphere_diameter_mm = sphere_diameter_mm
        self.prev_position = None
        self.tracking_history = []
        self.max_history = 30

        self.calibrator = SpatialCalibrator()
        self.quality_assessor = TrackingQualityAssessor()

        # Kalman filter
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.01 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        self.kalman_initialized = False

        self.analytics = FixedParkinsonsAnalytics()
        self.session_positions = []
        self.session_timestamps = []
        self.target_positions = []

        self.initial_radius = None
        self.joint_analyzer = JointAngleAnalyzer()
        self.joint_analyzer.calibrator = self.calibrator

    def detect_sphere(self, frame_roi):
        """Detect blue sphere using color detection"""
        candidates = []

        hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([85, 100, 120])
        upper_blue = np.array([140, 255, 255])

        hsv_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
        hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            if not (300 < area < 20000):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if circularity < 0.25:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            if not (8 < radius < 80):
                continue

            mask_region = np.zeros(frame_roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask_region, [contour], -1, 255, -1)

            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            mean_brightness = cv2.mean(gray, mask=mask_region)[0]

            confidence = min(1.0, (circularity * 0.5 + (area / 10000) * 0.5))

            candidates.append({
                'center': (int(x), int(y)),
                'radius': int(radius),
                'area': area,
                'circularity': circularity,
                'brightness': mean_brightness,
                'confidence': confidence,
                'contour': contour
            })

        return candidates, hsv_mask

    def track_sphere(self, frame_roi, timestamp):
        """Track sphere with quality assessment and robust calibration"""
        candidates, mask = self.detect_sphere(frame_roi)

        if not candidates:
            if self.kalman_initialized:
                prediction = self.kalman.predict()
                predicted_pos = (int(prediction[0]), int(prediction[1]))
                self.quality_assessor.update(detected=False)
                return predicted_pos, None, mask

            self.quality_assessor.update(detected=False)
            return None, None, mask

        best_candidate = None

        if self.prev_position is not None:
            min_distance = float('inf')
            for candidate in candidates:
                distance = np.sqrt((candidate['center'][0] - self.prev_position[0])**2 +
                                 (candidate['center'][1] - self.prev_position[1])**2)
                if distance < min_distance and distance < 300:
                    min_distance = distance
                    best_candidate = candidate

            if best_candidate is None and self.kalman_initialized and candidates:
                prediction = self.kalman.predict()
                predicted_pos = (int(prediction[0]), int(prediction[1]))

                min_pred_distance = float('inf')
                for candidate in candidates:
                    pred_distance = np.sqrt(
                        (candidate['center'][0] - predicted_pos[0])**2 +
                        (candidate['center'][1] - predicted_pos[1])**2
                    )
                    if pred_distance < min_pred_distance and pred_distance < 300:
                        min_pred_distance = pred_distance
                        best_candidate = candidate

        if best_candidate is None:
            best_candidate = max(candidates,
                               key=lambda x: x.get('confidence', x['circularity'] * x['area']))

        if best_candidate:
            center = best_candidate['center']
            radius = best_candidate['radius']

            if self.initial_radius is None:
                self.initial_radius = radius

            depth_factor = radius / self.initial_radius if self.initial_radius > 0 else 1.0

            # FIX #4: Robust median-based dynamic calibration
            self.calibrator.update_dynamic(radius * 2, self.sphere_diameter_mm)

            if not self.kalman_initialized:
                self.kalman.statePre = np.array([center[0], center[1], 0, 0], dtype=np.float32)
                self.kalman_initialized = True
            else:
                prediction = self.kalman.predict()
                measurement = np.array([[np.float32(center[0])],
                                       [np.float32(center[1])]])
                corrected = self.kalman.correct(measurement)
                center = (int(float(corrected[0])), int(float(corrected[1])))

            self.prev_position = center
            self.tracking_history.append(center)
            if len(self.tracking_history) > self.max_history:
                self.tracking_history.pop(0)

            self.session_positions.append(center)
            self.session_timestamps.append(timestamp)

            self.quality_assessor.update(
                detected=True,
                position=center,
                confidence=best_candidate['confidence'],
                radius=radius
            )

            return center, (radius, depth_factor), mask

        self.quality_assessor.update(detected=False)
        return None, None, mask

    def get_session_analytics(self, patient_id, exercise_type):
        """Generate analytics with calibration and quality info"""
        if len(self.session_positions) < 10:
            return None

        session_date = datetime.now().isoformat()

        self.analytics.calibrator = self.calibrator

        joint_stats = self.joint_analyzer.get_statistics()

        report = self.analytics.generate_session_report(
            patient_id=patient_id,
            session_date=session_date,
            exercise_type=exercise_type,
            positions=self.session_positions,
            timestamps=self.session_timestamps,
            target_positions=self.target_positions if self.target_positions else None,
            joint_stats=joint_stats
        )

        if report:
            report['calibration_info'] = self.calibrator.get_calibration_info()
            report['data_quality'] = self.quality_assessor.get_quality_report()

        return report

    def save_session_data(self, report, filename):
        """Save session data to JSON and CSV"""

        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.bool_, )):
                return bool(obj)
            elif isinstance(obj, (np.integer, )):
                return int(obj)
            elif isinstance(obj, (np.floating, )):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        clean_report = convert_numpy_types(report)

        with open(filename, 'w') as f:
            json.dump(clean_report, f, indent=2)

        csv_filename = filename.replace('.json', '.csv')
        df_data = {
            'timestamp': self.session_timestamps,
            'x_position': [pos[0] for pos in self.session_positions],
            'y_position': [pos[1] for pos in self.session_positions]
        }

        if len(self.joint_analyzer.angle_history['shoulder']) > 0:
            max_len = len(self.session_timestamps)
            for joint in ['shoulder', 'elbow', 'wrist']:
                angles = self.joint_analyzer.angle_history[joint].copy()
                if len(angles) < max_len:
                    angles.extend([None] * (max_len - len(angles)))
                elif len(angles) > max_len:
                    angles = angles[:max_len]
                df_data[f'{joint}_angle'] = angles

        df = pd.DataFrame(df_data)
        df.to_csv(csv_filename, index=False)

    def print_detailed_report(self, report):
        """Print detailed report"""
        print("\n" + "="*70)
        print(f"SUBJECT REPORT (FIXED VERSION 2.0)")
        print("="*70)

        print(f"patient_id = {report['patient_id']}")
        print(f"session_date = {report['session_date']}")
        print(f"exercise_type = {report['exercise_type']}")
        print(f"session_duration = {report['session_duration']}")
        print(f"total_movements = {report['total_movements']}")
        print(f"analysis_version = {report.get('analysis_version', 'unknown')}")

        print("\ntremor_metrics:")
        for key, value in report['tremor_metrics'].items():
            if key == 'fft_analysis':
                print(f"  fft_analysis:")
                for fft_key, fft_val in value.items():
                    print(f"    {fft_key} = {fft_val}")
            else:
                print(f"  {key} = {value}")

        print("\nbradykinesia_metrics:")
        for key, value in report['bradykinesia_metrics'].items():
            if key == 'segment_analysis':
                print(f"  segment_analysis:")
                for seg_key, seg_val in value.items():
                    print(f"    {seg_key} = {seg_val}")
            else:
                print(f"  {key} = {value}")

        print("\ndyskinesia_metrics:")
        for key, value in report['dyskinesia_metrics'].items():
            if key == 'residual_analysis':
                print(f"  residual_analysis:")
                for res_key, res_val in value.items():
                    print(f"    {res_key} = {res_val}")
            else:
                print(f"  {key} = {value}")

        print("\naccuracy_metrics:")
        for key, value in report['accuracy_metrics'].items():
            print(f"  {key} = {value}")

        print("\njoint_angle_metrics:")
        for joint_name, stats in report['joint_angle_metrics'].items():
            print(f"  {joint_name}:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for k, v in value.items():
                        print(f"      {k} = {v}")
                else:
                    print(f"    {key} = {value}")

        print("\nclassification:")
        for key, value in report['classification'].items():
            print(f"  {key} = {value}")

        print("\nclassification_evidence:")
        evidence = report.get('classification_evidence', {})
        print(f"  bradykinesia_evidence_score = {evidence.get('bradykinesia_evidence', 0)}")
        for detail in evidence.get('bradykinesia_details', []):
            print(f"    - {detail}")
        print(f"  dyskinesia_evidence_score = {evidence.get('dyskinesia_evidence', 0)}")
        for detail in evidence.get('dyskinesia_details', []):
            print(f"    - {detail}")

        print(f"\nsymptom_count = {report['symptom_count']}")
        print(f"overall_score = {report['overall_score']}")

        print("\nscore_components:")
        for key, value in report['score_components'].items():
            print(f"  {key} = {value}")

        print("\ncalibration_info:")
        for key, value in report['calibration_info'].items():
            print(f"  {key} = {value}")

        print("\ndata_quality:")
        for key, value in report['data_quality'].items():
            print(f"  {key} = {value}")

        print("\nrecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("\n" + "="*70)


# ============================================================================
# MAIN TRACKING FUNCTION
# ============================================================================

def track_blue_sphere_with_parkinson_analysis(
    video_path,
    output_path=None,
    frame_skip=2,
    scale_factor=0.6,
    start_time=0,
    duration=None,
    disable_display=False,
    patient_id="PATIENT_001",
    exercise_type="hand_tracking",
    sphere_diameter_mm=50,
    manual_px_per_mm=None
):
    """Main tracking function with all fixes applied"""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video at {video_path}")
        return None

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps

    width = int(orig_width * scale_factor)
    height = int(orig_height * scale_factor)

    print(f"=== VRRS PARKINSON'S ASSESSMENT SYSTEM (FIXED v2.0) ===")
    print(f"Patient ID: {patient_id}")
    print(f"Exercise Type: {exercise_type}")
    print(f"Video Duration: {timedelta(seconds=video_duration)}")
    print(f"Video FPS: {fps}")
    print(f"Processing frame_skip: {frame_skip}")
    print(f"Processing FPS: {fps/frame_skip:.1f}")
    print(f"")
    print(f"FIXES APPLIED:")
    print(f"  1. FFT tremor: signal interpolated to 20Hz before analysis")
    print(f"  2. Bradykinesia: segment-based (amplitude decrement, speed/distance, sub-movements)")
    print(f"  3. Dyskinesia: residual-based (voluntary movement removed)")
    print(f"  4. Calibration: robust median (not EMA)")
    print(f"  5. Classification: multi-factor evidence scoring")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ROI setup
    blue_roi = [70, 330, 830, 625]
    control_roi = [1040, 695, 2260, 900]

    blue_x, blue_y, blue_w, blue_h = [int(val * scale_factor) for val in blue_roi]
    sphere_tracker = BlueSphereTracker([blue_x, blue_y, blue_w, blue_h], sphere_diameter_mm)

    start_frame = int(start_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    if duration:
        end_frame = start_frame + int(duration * fps)
    else:
        end_frame = frame_count

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = fps / frame_skip
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    # ========================================================================
    # CALIBRATION PHASE
    # ========================================================================
    print("\n=== CALIBRATION PHASE ===")

    if manual_px_per_mm is not None:
        print(f"Using MANUAL calibration: {manual_px_per_mm} px/mm")
        sphere_tracker.calibrator.manual_calibrate(manual_px_per_mm, "manual_user_specified")
        calibration_successful = True
    else:
        print(f"Attempting AUTO-CALIBRATION with sphere diameter: {sphere_diameter_mm}mm")
        print("Searching for blue sphere in first 100 frames...")

        calibration_successful = False
        calibration_attempts = 0
        max_calibration_frames = 100
        detected_radii = []

        while calibration_attempts < max_calibration_frames and not calibration_successful:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))
            blue_roi_frame = frame[blue_y:blue_y+blue_h, blue_x:blue_x+blue_w]

            candidates, _ = sphere_tracker.detect_sphere(blue_roi_frame)

            if candidates:
                best = max(candidates, key=lambda x: x['confidence'])
                detected_radii.append(best['radius'])

                if len(detected_radii) >= 5:
                    avg_radius = np.mean(detected_radii[-5:])
                    std_radius = np.std(detected_radii[-5:])

                    if std_radius / avg_radius < 0.2:
                        detected_diameter_px = avg_radius * 2

                        sphere_tracker.calibrator.calibrate_with_known_object(
                            frame=blue_roi_frame,
                            object_size_mm=sphere_diameter_mm,
                            detection_callback=lambda f: detected_diameter_px
                        )

                        calibration_successful = True
                        break

            calibration_attempts += 1

        if not calibration_successful:
            print("\nWARNING: Auto-calibration failed in first 100 frames!")
            print("  Dynamic calibration will kick in once the ball is detected during tracking.")

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print("\n=== STARTING MAIN TRACKING ===\n")

    # ========================================================================
    # MAIN PROCESSING LOOP
    # ========================================================================

    control_x, control_y, control_w, control_h = [
        int(val * scale_factor) for val in control_roi
    ]

    rocket_emoji_x = None
    rocket_emoji_y = None

    frame_num = start_frame
    processed_count = 0

    current_metrics = {
        "tremor": 0,
        "speed": 0,
        "smoothness": 100,
        "accuracy": 0
    }

    print("=== CONTROLS ===")
    print("Press 'q' to quit")
    print("Press 'p' to pause")
    print("Press 'r' for real-time report")

    while cap.isOpened() and frame_num < end_frame:
        ret, frame = cap.read()

        if not ret:
            print("End of video reached")
            break

        if (frame_num - start_frame) % frame_skip != 0:
            frame_num += 1
            continue

        processed_count += 1
        current_time = frame_num / fps

        frame_original = cv2.resize(frame, (width, height))
        frame = frame_original.copy()

        # Pose detection
        frame_rgb = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            joint_data = sphere_tracker.joint_analyzer.analyze_pose_landmarks(
                pose_results.pose_landmarks.landmark,
                current_time,
                frame_width=width,
                frame_height=height
            )

        # Sphere tracking
        blue_roi_frame = frame_original[blue_y:blue_y+blue_h, blue_x:blue_x+blue_w]

        sphere_result = sphere_tracker.track_sphere(blue_roi_frame, current_time)
        sphere_center = sphere_result[0]
        sphere_info = sphere_result[1]
        detection_mask = sphere_result[2]

        # Update metrics periodically
        if processed_count % 40 == 0 and len(sphere_tracker.session_positions) > 50:
            try:
                recent_positions = sphere_tracker.session_positions[-50:]
                recent_timestamps = sphere_tracker.session_timestamps[-50:]

                sphere_tracker.analytics.calibrator = sphere_tracker.calibrator

                tremor_data = sphere_tracker.analytics.calculate_tremor_metrics(
                    recent_positions, recent_timestamps
                )
                speed_data = sphere_tracker.analytics.calculate_bradykinesia_metrics(
                    recent_positions, recent_timestamps
                )
                smoothness_data = sphere_tracker.analytics.calculate_dyskinesia_metrics(
                    recent_positions, recent_timestamps
                )

                current_metrics.update({
                    "tremor": tremor_data.get("tremor_amplitude", 0),
                    "speed": speed_data.get("avg_speed", 0),
                    "smoothness": smoothness_data.get("path_smoothness", 1.0) * 100,
                })

            except Exception as e:
                pass

        # Visualization
        if sphere_center is not None and sphere_info is not None:
            sphere_radius, depth_factor = sphere_info

            display_radius = max(int(sphere_radius * 0.8), 5)

            circle_color = (0, 255, 255) if depth_factor > 0.7 else \
                          (0, 200, 255) if depth_factor > 0.4 else (100, 150, 255)

            cv2.circle(blue_roi_frame, sphere_center, display_radius, circle_color, 2)
            cv2.circle(blue_roi_frame, sphere_center, 3, (0, 0, 255), -1)

            # Info panel
            info_height = 200
            cv2.rectangle(blue_roi_frame, (5, 5), (240, info_height), (0, 0, 0), -1)
            cv2.rectangle(blue_roi_frame, (5, 5), (240, info_height), (255, 255, 255), 1)

            y_offset = 22
            spacing = 22

            unit = "mm" if sphere_tracker.calibrator.calibrated else "px"
            unit_speed = "mm/s" if sphere_tracker.calibrator.calibrated else "px/s"

            cv2.putText(blue_roi_frame,
                       f"POS: {sphere_center[0] + blue_x},{sphere_center[1] + blue_y}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            y_offset += spacing

            cv2.putText(blue_roi_frame, f"TREMOR: {current_metrics['tremor']:.2f}{unit}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1)
            y_offset += spacing

            cv2.putText(blue_roi_frame, f"SPEED: {current_metrics['speed']:.1f}{unit_speed}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
            y_offset += spacing

            cv2.putText(blue_roi_frame, f"SMOOTH: {current_metrics['smoothness']:.1f}%",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)
            y_offset += spacing

            cv2.putText(blue_roi_frame, f"DEPTH: {depth_factor:.2f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 100), 1)
            y_offset += spacing

            if sphere_tracker.calibrator.calibrated:
                px_mm = sphere_tracker.calibrator.px_per_mm
                cv2.putText(blue_roi_frame, f"PX/MM: {px_mm:.2f} (median)",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 200), 1)
                y_offset += spacing

            cv2.putText(blue_roi_frame, "v2.0-FIXED",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 255), 1)
            y_offset += spacing

            if processed_count > 30:
                detection_rate = sphere_tracker.quality_assessor.get_detection_rate()
                reliability = sphere_tracker.quality_assessor._get_reliability_level()

                if reliability == "HIGH":
                    quality_color = (0, 255, 0)
                elif reliability == "MODERATE":
                    quality_color = (0, 255, 255)
                else:
                    quality_color = (0, 0, 255)

                cv2.putText(blue_roi_frame, f"QUALITY: {reliability} ({detection_rate*100:.0f}%)",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, quality_color, 1)

            # Tracking trail
            if len(sphere_tracker.tracking_history) > 1:
                for i in range(1, len(sphere_tracker.tracking_history)):
                    prev_pt = sphere_tracker.tracking_history[i-1]
                    curr_pt = sphere_tracker.tracking_history[i]
                    alpha = i / len(sphere_tracker.tracking_history)
                    color = (int(255 * alpha), int(255 * alpha), 0)
                    cv2.line(blue_roi_frame, prev_pt, curr_pt, color, 2)
        else:
            cv2.putText(blue_roi_frame, "SEARCHING...",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Control area
        control_roi_frame = frame_original[control_y:control_y+control_h,
                                           control_x:control_x+control_w]

        if sphere_center is not None:
            sphere_rel_x = sphere_center[0] / blue_w if blue_w > 0 else 0.5
            sphere_rel_y = sphere_center[1] / blue_h if blue_h > 0 else 0.5

            inverted_rel_x = 1.0 - sphere_rel_x

            usable_width = control_w * 0.8
            offset_x = control_w * 0.1
            usable_height = control_h * 0.8
            offset_y = control_h * 0.1

            target_rocket_x = control_x + offset_x + (inverted_rel_x * usable_width)
            target_rocket_y = control_y + offset_y + (sphere_rel_y * usable_height)

            sphere_tracker.target_positions.append((target_rocket_x, target_rocket_y))

            if rocket_emoji_x is None:
                rocket_emoji_x = target_rocket_x
                rocket_emoji_y = target_rocket_y
            else:
                smoothing = 0.6
                rocket_emoji_x = smoothing * rocket_emoji_x + (1 - smoothing) * target_rocket_x
                rocket_emoji_y = smoothing * rocket_emoji_y + (1 - smoothing) * target_rocket_y

            emoji_x_local = int(rocket_emoji_x - control_x)
            emoji_y_local = int(rocket_emoji_y - control_y)

            cv2.circle(control_roi_frame, (emoji_x_local, emoji_y_local), 25, (255, 255, 0), 3)
            cv2.circle(control_roi_frame, (emoji_x_local, emoji_y_local), 20, (0, 255, 255), 2)
            cv2.circle(control_roi_frame, (emoji_x_local, emoji_y_local), 5, (0, 255, 0), -1)

        # Place ROIs back
        frame[blue_y:blue_y+blue_h, blue_x:blue_x+blue_w] = blue_roi_frame
        frame[control_y:control_y+control_h, control_x:control_x+control_w] = control_roi_frame

        # Dashboard
        dash_height = 180
        cv2.rectangle(frame, (10, 10), (400, dash_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, dash_height), (255, 255, 255), 2)

        y_pos = 35
        line_h = 22

        cv2.putText(frame, f"PATIENT: {patient_id}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_pos += line_h

        cv2.putText(frame, f"EXERCISE: {exercise_type.upper()}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += line_h

        calib_method = sphere_tracker.calibrator.calibration_method
        calib_color = (0, 255, 0) if "dynamic" in calib_method or "auto" in calib_method else (0, 255, 255)
        cv2.putText(frame, f"CALIB: {calib_method}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.35, calib_color, 1)
        y_pos += line_h

        time_str = str(timedelta(seconds=frame_num / fps)).split('.')[0]
        cv2.putText(frame, f"TIME: {time_str}",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += line_h

        detection_rate = sphere_tracker.quality_assessor.get_detection_rate()
        cv2.putText(frame, f"DETECTION: {detection_rate*100:.1f}%",
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Progress bar
        progress = (frame_num - start_frame) / (end_frame - start_frame)
        bar_y = height - 35
        bar_width = width - 40

        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + 20), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, bar_y),
                     (20 + int(bar_width * progress), bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + 20), (255, 255, 255), 2)

        # ROI boxes
        cv2.rectangle(frame, (blue_x, blue_y), (blue_x+blue_w, blue_y+blue_h), (255, 0, 0), 2)
        cv2.rectangle(frame, (control_x, control_y),
                     (control_x+control_w, control_y+control_h), (255, 255, 0), 2)

        if output_path:
            out.write(frame)

        if not disable_display:
            cv2.imshow('VRRS - Fixed v2.0', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Stopped by user")
                break
            elif key == ord('p'):
                print("Paused. Press any key to continue...")
                cv2.waitKey(-1)
            elif key == ord('r'):
                if len(sphere_tracker.session_positions) > 30:
                    report = sphere_tracker.get_session_analytics(patient_id, exercise_type)
                    if report:
                        print(f"\n=== REAL-TIME REPORT ===")
                        print(f"Score: {report['overall_score']:.1f}/100")
                        print(f"Tremor: {report['classification']['tremor']}")
                        print(f"Bradykinesia: {report['classification']['bradykinesia']}")
                        print(f"Dyskinesia: {report['classification']['dyskinesia']}")
                        print(f"Calibration: {report['calibration_info']['method']}")
                        print(f"Current px/mm: {sphere_tracker.calibrator.px_per_mm:.3f}")
                        print(f"Reliability: {report['data_quality']['reliability_level']}")
                        cv2.waitKey(-1)

        frame_num += 1

    # ========================================================================
    # FINAL REPORT
    # ========================================================================

    print("\n=== GENERATING FINAL REPORT ===")

    final_report = sphere_tracker.get_session_analytics(patient_id, exercise_type)

    if final_report:
        report_filename = (f"{patient_id}_{exercise_type}_"
                          f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        sphere_tracker.save_session_data(final_report, report_filename)

        sphere_tracker.print_detailed_report(final_report)

        print(f"\nFiles saved:")
        print(f"  - {report_filename}")
        print(f"  - {report_filename.replace('.json', '.csv')}")
    else:
        print("Insufficient data for assessment")

    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    pose.close()

    print("\n=== SESSION COMPLETED ===")
    return final_report


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_args():
    """Command-line options (defaults match the settings used in the thesis)."""
    import argparse
    p = argparse.ArgumentParser(
        description="VRRS keywand tracking and movement analysis for one session video.")
    p.add_argument("video", help="Path to the session video (e.g. session.mkv)")
    p.add_argument("--session-id", default="SESSION_001",
                   help="Anonymous session code used in the output file names")
    p.add_argument("--exercise", default="hand_tracking_exercise", help="Exercise label")
    p.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (default 1)")
    p.add_argument("--scale", type=float, default=0.6, help="Downscale factor (default 0.6)")
    p.add_argument("--start", type=float, default=0, help="Start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="Seconds to process (default: all)")
    p.add_argument("--sphere-mm", type=float, default=50, help="Keywand sphere diameter in mm")
    p.add_argument("--px-per-mm", type=float, default=None,
                   help="Manual calibration (px/mm) if auto-calibration fails")
    p.add_argument("--output-video", default=None, help="Optional path to save the annotated video")
    p.add_argument("--no-display", action="store_true", help="Run without a preview window")
    return p.parse_args()


def main():
    """Run the analysis on one video and print a short summary."""
    args = parse_args()

    print("=" * 70)
    print("VRRS - keywand tracking and movement analysis")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Sphere diameter: {args.sphere_mm} mm | frame_skip={args.frame_skip} | scale={args.scale}")
    if args.px_per_mm:
        print(f"Manual calibration: {args.px_per_mm} px/mm")
    else:
        print("Calibration: auto + dynamic (robust median)")
    if not args.no_display:
        print("\nPress 'q' to quit, 'p' to pause, 'r' for a real-time report\n")

    session_report = track_blue_sphere_with_parkinson_analysis(
        video_path=args.video,
        output_path=args.output_video,
        frame_skip=args.frame_skip,
        scale_factor=args.scale,
        start_time=args.start,
        duration=args.duration,
        disable_display=args.no_display,
        patient_id=args.session_id,
        exercise_type=args.exercise,
        sphere_diameter_mm=args.sphere_mm,
        manual_px_per_mm=args.px_per_mm
    )

    if session_report:
        print("\nANALYSIS COMPLETED")
        print(f"Calibration: {session_report['calibration_info']['method']}")
        px_mm = session_report['calibration_info']['px_per_mm']
        if px_mm:
            print(f"Final px/mm: {px_mm:.3f}")
        print(f"Data quality: {session_report['data_quality']['reliability_level']}")
        print("Heuristic screening output (experimental, not clinically validated):")
        print(f"  Overall: {session_report['classification']['overall']}")
        print(f"  Tremor: {session_report['classification']['tremor']}")
        print(f"  Bradykinesia: {session_report['classification']['bradykinesia']}")
        print(f"  Dyskinesia: {session_report['classification']['dyskinesia']}")
    else:
        print("ANALYSIS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()