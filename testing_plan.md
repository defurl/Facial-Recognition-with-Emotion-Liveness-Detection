# Facial Recognition System Verification Plan

This document outlines the testing strategy to verify that the migrated Web Application matches or exceeds the functionality of the original Python GUI.

## 🎯 Test Objectives
1.  **Parity**: Ensure Web App performs all core functions (Register, Verify, Log) as reliably as the Python version.
2.  **Liveness**: Validation of the new "Burst Capture" blink detection.
3.  **Security**: Confirm "Spoof Detected" masking works.
4.  **Integration**: Verify frontend actions correctly trigger backend database and logging operations.

---

## 🧪 Test Suite 1: Registration Flow
**Goal**: Verify the "Registration Wizard" correctly captures high-quality embeddings and saves them to `db.py`.

| ID | Test Case | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **R1** | **New User Registration** | 1. Click "Register New User".<br>2. Enter unique name (e.g., "TestUser1").<br>3. Follow wizard to capture 5 angles.<br>4. Submit. | - Wizard completes.<br> - "TestUser1" appears in "Registered Users" sidebar.<br> - Backend logs: "Saved employee db". |
| **R2** | **Duplicate User Handling** | 1. Click "Register New User".<br>2. Enter "TestUser1" again.<br>3. Try to proceed. | - Backend returns 400 Error ("Employee already exists").<br> - UI shows error message.<br> - (Alternatively: Check allow overwrite logic). |
| **R3** | **Quality Controls** | 1. Start Registration.<br>2. Cover camera (black/dark).<br>3. Attempt capture. | - UI warns about "Too dark" or "No face detected".<br> - Prevents proceeding to next step. |

## 🧪 Test Suite 2: Verification & Liveness
**Goal**: Verify "Real-time" detection, Identity matching, and Liveness Security.

| ID | Test Case | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **V1** | **Real Face (Pass)** | 1. User sits in front of camera.<br>2. **Blink naturally** within 2 seconds of verification scan.<br>3. Observe result. | - **Identity**: Matches User Name.<br> - **Liveness**: "Real"<br> - **Overlay**: Green Box.<br> - **Logs**: Attendance entry created. |
| **V2** | **Photo Attack (Spoof)** | 1. Hold a photo of specific user to camera.<br>2. Wait for verify scan.<br>3. (Do not blink, obviously). | - **Identity**: "Spoof Detected" (Name hidden).<br> - **Liveness**: "Spoof"<br> - **Overlay**: Red Box.<br> - **Logs**: NO attendance entry. |
| **V3** | **No Blink (Active Liveness)** | 1. Real user sits still, **staring without blinking** for >3 seconds during scan. | - **Identity**: "Spoof Detected".<br> - **Liveness**: "Spoof" (due to low EAR variance).<br> - Overlay shows "Needed: 1". |
| **V4** | **Unknown Person** | 1. Unregistered person sits in front of camera.<br>2. Blink naturally. | - **Identity**: "Unknown" or "Not Registered".<br> - **Liveness**: "Real" (if they blink). |

## 🧪 Test Suite 3: Attendance Logging
**Goal**: Verify the data pipeline from Frontend -> Backend -> CSV/UI.

| ID | Test Case | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **A1** | **Successful Log** | 1. Perform Test **V1** (Real Verify).<br>2. Check "Attendance Log" sidebar.<br>3. Check `backend/output/attendance_log.csv`. | - New row added with Timestamp, Name, Distance, Liveness="Real".<br> - Sidebar updates automatically/on refresh. |
| **A2** | **Spoof Rejection** | 1. Perform Test **V2** (Spoof).<br>2. Check logs. | - **NO** new entry should be created for the spoof attempt. |

## 🧪 Test Suite 4: UI/UX & Robustness
**Goal**: Verify layout, mirroring, and error handling.

| ID | Test Case | Steps | Expected Result |
| :--- | :--- | :--- | :--- |
| **U1** | **Mirror Toggle** | 1. Click "Mirror Camera" button.<br>2. Move head Left. | - **Standard**: Video moves Right (like Webcam). Box tracks correctly.<br> - **Mirrored**: Video moves Left (like Mirror). Box tracks correctly. |
| **U2** | **Backend Error** | 1. Kill `server.py` terminal.<br>2. Try to verify. | - UI shows "Offline" badge.<br> - Error message "Network Error" or similar displayed gracefully. |
