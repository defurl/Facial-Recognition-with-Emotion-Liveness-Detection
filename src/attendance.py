"""
Attendance logging system with CSV storage and cooldown management.
"""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import pandas as pd

from src.config import OUTPUT_DIR


class AttendanceLogger:
    """
    Manages attendance logging with CSV storage and cooldown prevention.
    """
    
    def __init__(self, csv_path=None, cooldown_minutes=60):
        """
        Initialize attendance logger.
        
        Args:
            csv_path: Path to CSV file for logging
            cooldown_minutes: Minimum minutes between attendance marks for same employee
        """
        default_path = OUTPUT_DIR / 'attendance_log.csv'
        self.csv_path = Path(csv_path) if csv_path is not None else default_path
        self.cooldown_minutes = cooldown_minutes
        self.lock = threading.Lock()
        self.last_attendance = {}  # {employee_name: datetime}
        self.today_attendees = set() # {employee_name}
        
        # Ensure output directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV with headers if not exists
        if not self.csv_path.exists():
            self._create_csv()
        else:
            # Load existing attendance records to populate cooldown cache and today's set
            self._load_recent_attendance()
            self._reload_today_attendees()
    
    def _create_csv(self):
        """Create CSV file with headers."""
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'timestamp', 'confidence', 
                                'emotion', 'liveness'])
            print(f"Created attendance log: {self.csv_path}")
        except Exception as e:
            print(f"Error creating attendance CSV: {e}")
    
    def _load_recent_attendance(self):
        """Load recent attendance records to initialize cooldown cache."""
        try:
            if not self.csv_path.exists():
                return
            
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        timestamp = datetime.fromisoformat(row['timestamp'])
                        name = row['name']
                        
                        # Only keep records within cooldown window
                        if datetime.now() - timestamp <= timedelta(minutes=self.cooldown_minutes):
                            if name not in self.last_attendance or timestamp > self.last_attendance[name]:
                                self.last_attendance[name] = timestamp
                    except (KeyError, ValueError) as e:
                        continue  # Skip malformed rows
        except Exception as e:
            print(f"Error loading recent attendance: {e}")
    def _reload_today_attendees(self):
        """Reload the set of employees who have checked in today."""
        try:
            today_str = datetime.now().strftime('%Y-%m-%d')
            self.today_attendees.clear()
            
            if not self.csv_path.exists():
                return
            
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse timestamp to check if it's today
                    # ISO format: YYYY-MM-DDTHH:MM:SS.mmmmmm
                    if row['timestamp'].startswith(today_str):
                        self.today_attendees.add(row['name'])
        except Exception as e:
            print(f"Error reloading today's attendees: {e}")

    def has_checked_in_today(self, employee_name):
        """Check if employee has already checked in today (ignoring cooldown)."""
        # If set is empty (maybe first run or new day), try reload to be safe
        # But for performance we rely on the in-memory set maintained by mark_attendance
        # We can do a lazy check: if empty, reload? No, safer to rely on init.
        # But we need to handle day rollover.
        
        # Simple check:
        return employee_name in self.today_attendees

    def can_mark_attendance(self, employee_name):
        """
        Check if employee can mark attendance (cooldown expired).
        
        Args:
            employee_name: Name of employee
        
        Returns:
            tuple: (can_mark: bool, reason: str)
        """
        with self.lock:
            if employee_name not in self.last_attendance:
                return (True, "OK")
            
            last_time = self.last_attendance[employee_name]
            elapsed = datetime.now() - last_time
            cooldown_delta = timedelta(minutes=self.cooldown_minutes)
            
            if elapsed >= cooldown_delta:
                return (True, "OK")
            else:
                remaining = cooldown_delta - elapsed
                minutes_left = int(remaining.total_seconds() / 60)
                return (False, f"Cooldown active: {minutes_left} minutes remaining")
    
    def mark_attendance(self, employee_name, distance, emotion, liveness):
        """
        Mark attendance for employee if cooldown allows.
        
        Args:
            employee_name: Name of employee
            distance: Confidence distance value
            emotion: Detected emotion
            liveness: Liveness status ("Real" or "Spoof")
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            with self.lock:
                # Check cooldown
                can_mark, reason = self.can_mark_attendance(employee_name)
                if not can_mark:
                    return (False, reason)
            
            # Attempt to write with retry on lock
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    timestamp = datetime.now()
                    
                    # Append to CSV (matching column order: name, timestamp, confidence, emotion, liveness)
                    with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            employee_name,
                            timestamp.isoformat(),
                            f"{distance:.4f}",
                            emotion,
                            liveness
                        ])
                    
                    # Update cooldown cache
                    self.last_attendance[employee_name] = timestamp
                    # Update today's set
                    self.today_attendees.add(employee_name)
                    
                    return (True, f"Attendance marked for {employee_name}")
                    
                except PermissionError as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    else:
                        return (False, f"File locked: {e}")
                except Exception as e:
                    return (False, f"Error writing attendance: {e}")
        except Exception as e:
            return (False, f"Attendance marking error: {e}")
    
    def get_today_records(self):
        """
        Get all attendance records for today.
        
        Returns:
            pandas.DataFrame: Today's attendance records
        """
        try:
            if not self.csv_path.exists():
                return pd.DataFrame(columns=['timestamp', 'employee_name', 'confidence_distance',
                                            'emotion', 'liveness_status'])
            
            # Read CSV
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return df
            
            # Parse timestamps and filter by today
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            today = datetime.now().date()
            today_df = df[df['timestamp'].dt.date == today]
            
            return today_df
            
        except Exception as e:
            print(f"Error reading today's records: {e}")
            return pd.DataFrame()
    
    def generate_daily_summary(self):
        """
        Generate summary statistics for today's attendance.
        
        Returns:
            dict: Summary statistics
        """
        try:
            today_df = self.get_today_records()
            
            if today_df.empty:
                return {
                    'total_count': 0,
                    'unique_employees': 0,
                    'avg_confidence': 0.0,
                    'liveness_real_count': 0,
                    'liveness_spoof_count': 0
                }
            
            # Calculate statistics
            total_count = len(today_df)
            unique_employees = today_df['employee_name'].nunique()
            
            # Average confidence (1 - distance)
            # Assuming threshold is 0.8 for normalization
            distances = pd.to_numeric(today_df['confidence_distance'], errors='coerce')
            avg_distance = distances.mean() if not distances.isna().all() else 0.0
            avg_confidence = max(0.0, 1 - (avg_distance / 0.8))
            
            # Liveness counts
            liveness_real_count = (today_df['liveness_status'] == 'Real').sum()
            liveness_spoof_count = (today_df['liveness_status'] == 'Spoof').sum()
            
            return {
                'total_count': int(total_count),
                'unique_employees': int(unique_employees),
                'avg_confidence': float(avg_confidence),
                'liveness_real_count': int(liveness_real_count),
                'liveness_spoof_count': int(liveness_spoof_count)
            }
            
        except Exception as e:
            print(f"Error generating daily summary: {e}")
            return {
                'total_count': 0,
                'unique_employees': 0,
                'avg_confidence': 0.0,
                'liveness_real_count': 0,
                'liveness_spoof_count': 0
            }
    
    def get_employee_attendance_today(self, employee_name):
        """
        Get attendance records for specific employee today.
        
        Args:
            employee_name: Name of employee
        
        Returns:
            pandas.DataFrame: Employee's attendance records today
        """
        try:
            today_df = self.get_today_records()
            if today_df.empty:
                return today_df
            
            return today_df[today_df['employee_name'] == employee_name]
        except Exception as e:
            print(f"Error getting employee attendance: {e}")
            return pd.DataFrame()
    
    def get_today_attendance(self):
        """
        Get all attendance records for today (alias for get_today_records).
        
        Returns:
            pandas.DataFrame: Today's attendance records
        """
        return self.get_today_records()
    
    def get_all_attendance(self):
        """
        Get all attendance records from the CSV file.
        
        Returns:
            pandas.DataFrame: All attendance records
        """
        try:
            if not self.csv_path.exists():
                return pd.DataFrame(columns=['timestamp', 'employee_name', 'confidence_distance',
                                            'emotion', 'liveness_status'])
            
            # Read entire CSV
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return df
            
            # Parse timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp descending (newest first)
            df = df.sort_values('timestamp', ascending=False)
            
            return df
            
        except Exception as e:
            print(f"Error reading all records: {e}")
            return pd.DataFrame()
    
    def get_attendance_summary(self, days=7):
        """
        Get attendance summary for the last N days.
        
        Args:
            days: Number of days to include in summary
        
        Returns:
            dict: Summary statistics for the period
        """
        try:
            if not self.csv_path.exists():
                return {
                    'total_marks': 0,
                    'unique_employees': 0,
                    'avg_confidence': 0.0,
                    'days_covered': 0
                }
            
            # Read CSV
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return {
                    'total_marks': 0,
                    'unique_employees': 0,
                    'avg_confidence': 0.0,
                    'days_covered': 0
                }
            
            # Parse timestamps and filter by days
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_df = df[df['timestamp'] >= cutoff_date]
            
            if recent_df.empty:
                return {
                    'total_marks': 0,
                    'unique_employees': 0,
                    'avg_confidence': 0.0,
                    'days_covered': 0
                }
            
            # Calculate statistics
            total_marks = len(recent_df)
            # Handle both 'name' and 'employee_name' column names for backward compatibility
            name_col = 'name' if 'name' in recent_df.columns else 'employee_name'
            unique_employees = recent_df[name_col].nunique()
            
            # Calculate days covered
            days_covered = (recent_df['timestamp'].max() - recent_df['timestamp'].min()).days + 1
            
            # Average confidence (1 - distance / threshold)
            # Handle both 'confidence' and 'confidence_distance' column names
            conf_col = 'confidence' if 'confidence' in recent_df.columns else 'confidence_distance'
            distances = pd.to_numeric(recent_df[conf_col], errors='coerce')
            avg_distance = distances.mean() if not distances.isna().all() else 0.0
            avg_confidence = max(0.0, min(1.0, 1 - (avg_distance / 1.0))) * 100
            
            return {
                'total_marks': int(total_marks),
                'unique_employees': int(unique_employees),
                'avg_confidence': float(avg_confidence),
                'days_covered': int(days_covered)
            }
            
        except Exception as e:
            print(f"Error generating attendance summary: {e}")
            return {
                'total_marks': 0,
                'unique_employees': 0,
                'avg_confidence': 0.0,
                'days_covered': 0
            }


if __name__ == "__main__":
    # Test the attendance logger
    print("Testing AttendanceLogger...")
    
    logger = AttendanceLogger(csv_path=OUTPUT_DIR / 'test_attendance.csv', cooldown_minutes=1)
    
    # Test marking attendance
    success, msg = logger.mark_attendance("John Doe", 0.35, "Happy", "Real")
    print(f"First mark: {success}, {msg}")
    
    # Test cooldown
    success, msg = logger.mark_attendance("John Doe", 0.32, "Neutral", "Real")
    print(f"Immediate retry: {success}, {msg}")
    
    # Test another employee
    success, msg = logger.mark_attendance("Jane Smith", 0.40, "Neutral", "Real")
    print(f"Different employee: {success}, {msg}")
    
    # Get today's records
    today = logger.get_today_records()
    print(f"\nToday's records:\n{today}")
    
    # Generate summary
    summary = logger.generate_daily_summary()
    print(f"\nDaily summary: {summary}")
    
    print("\n✓ AttendanceLogger test complete!")
