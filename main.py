import time
import threading
import csv
import queue
import os
import pandas as pd
import logging
from datetime import datetime
from collections import deque
import pytz
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app)

BANGKOK_TZ = pytz.timezone('Asia/Bangkok')

OSC_IP = "0.0.0.0"
OSC_PORT = 12345
ACC_THRESHOLD = 0.25
GYRO_THRESHOLD = 0.1
MAG_THRESHOLD = 1.0
RESET_TIMEOUT = 0.5
NOISE_THRESHOLD = 0.001
MEAN_THRESHOLD = 0.1  # ค่าเฉลี่ยที่ต่ำกว่านี้จะถูกตั้งเป็น 0.0

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WALK_DIR = os.path.join(BASE_DATA_DIR, "walk")
BED_DIR = os.path.join(BASE_DATA_DIR, "bed to toilet")

for directory in [BASE_DATA_DIR, WALK_DIR, BED_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

if not os.access(BASE_DATA_DIR, os.W_OK):
    logger.error(f"No write permission in directory: {BASE_DATA_DIR}")
    raise PermissionError(f"No write permission in {BASE_DATA_DIR}")

data_dict = {
    "ACC": {"x": 0.0, "y": 0.0, "z": 0.0},
    "GYRO": {"x": 0.0, "y": 0.0, "z": 0.0},
    "MAG": {"x": 0.0, "y": 0.0, "z": 0.0},
}
baseline = {
    "ACC": {"x": 0.0, "y": 0.0, "z": 0.0},
    "GYRO": {"x": 0.0, "y": 0.0, "z": 0.0},
}
previous_value = {
    "ACC": {"x": 0.0, "y": 0.0, "z": 0.0},
    "GYRO": {"x": 0.0, "y": 0.0, "z": 0.0},
}
last_active_time = {
    "ACC": time.time(),
    "GYRO": time.time(),
    "MAG": time.time(),
}
MAG_baseline = {"x": 1.0, "y": 1.0, "z": 1.0}
is_recording = False
record_lock = threading.Lock()
csv_queue = queue.Queue()
csv_thread = None
current_raw_file = None
current_resampled_file = None
resample_lock = threading.Lock()
current_activity = None

sensor_buffer = {
    "ACC": deque(),
    "GYRO": deque(),
}
buffer_lock = threading.Lock()

def normalize_zero(value):
    return 0.0 if abs(value) < NOISE_THRESHOLD else value

def compute_mean_and_emit():
    with buffer_lock:
        acc_data = list(sensor_buffer["ACC"])
        gyro_data = list(sensor_buffer["GYRO"])
        sensor_buffer["ACC"].clear()
        sensor_buffer["GYRO"].clear()

    def calculate_mean(data, sensor_type):
        if not data:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        df = pd.DataFrame(data, columns=["timestamp", "x", "y", "z"])
        mean_values = df[["x", "y", "z"]].mean().to_dict()
        mean_values = {k: normalize_zero(v) for k, v in mean_values.items()}
        # ตั้งค่าเฉลี่ยที่ต่ำกว่า MEAN_THRESHOLD เป็น 0.0
        mean_values = {k: 0.0 if abs(v) < MEAN_THRESHOLD else v for k, v in mean_values.items()}
        logger.debug(f"Mean {sensor_type}: x={mean_values['x']:.4f}, y={mean_values['y']:.4f}, z={mean_values['z']:.4f}")
        return mean_values

    data_dict["ACC"] = calculate_mean(acc_data, "ACC")
    data_dict["GYRO"] = calculate_mean(gyro_data, "GYRO")

    socketio.emit("sensor_data", data_dict)

    threading.Timer(1.0, compute_mean_and_emit).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/download/raw/<activity>/<filename>")
def download_raw_file(activity, filename):
    try:
        file_path = os.path.join(BASE_DATA_DIR, activity, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"Sending raw file: {file_path}, size: {file_size} bytes")
            with open(file_path, 'rb') as f:
                response = Response(
                    f.read(),
                    mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
            return response
        else:
            logger.error(f"Raw file not found: {file_path}")
            return f"Raw file not found: {file_path}", 404
    except Exception as e:
        logger.error(f"Error reading raw file {file_path}: {str(e)}")
        return f"Error reading raw file: {str(e)}", 500

@app.route("/download/resampled/<activity>/<filename>")
def download_resampled_file(activity, filename):
    try:
        file_path = os.path.join(BASE_DATA_DIR, activity, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"Sending resampled file: {file_path}, size: {file_size} bytes")
            with open(file_path, 'rb') as f:
                response = Response(
                    f.read(),
                    mimetype='text/csv',
                    headers={"Content-Disposition": f"attachment; filename={filename}"}
                )
            return response
        else:
            logger.error(f"Resampled file not found: {file_path}")
            return f"Resampled file not found: {file_path}", 404
    except Exception as e:
        logger.error(f"Error reading resampled file {file_path}: {str(e)}")
        return f"Error reading resampled file: {str(e)}", 500

@socketio.on("activity_started")
def handle_activity_started(data):
    global current_activity
    current_activity = data["activity"]
    logger.info(f"Activity started: {current_activity}")

def print_all(address, *args):
    global is_recording, baseline, previous_value, last_active_time, current_activity
    current_time = time.time()
    updated = False
    baseline_updated = False

    def process(sensor_type, threshold):
        nonlocal updated, baseline_updated
        key = address.split('/')[-1].lower()
        if ':' in key:
            key = key.split(':')[-1].lower()
        try:
            value = float(args[-1])
        except (ValueError, TypeError):
            logger.error(f"Invalid OSC value for {sensor_type} {key}: {args[-1]}")
            return

        if sensor_type in ["ACC", "GYRO"] and key in baseline[sensor_type]:
            delta = abs(value - previous_value[sensor_type][key])
            if delta < threshold / 2:
                baseline[sensor_type][key] = normalize_zero((baseline[sensor_type][key] + value) / 2.0)
                baseline_updated = True
            previous_value[sensor_type][key] = value
            value_corrected = normalize_zero(value - baseline[sensor_type][key])
            with buffer_lock:
                data_dict[sensor_type][key] = value_corrected if abs(value_corrected) > threshold else 0.0
                if current_time - last_active_time[sensor_type] > RESET_TIMEOUT:
                    for axis in ["x", "y", "z"]:
                        data_dict[sensor_type][axis] = 0.0
                        baseline[sensor_type][axis] = 0.0
                    sensor_buffer[sensor_type].append((current_time, 0.0, 0.0, 0.0))
                else:
                    sensor_buffer[sensor_type].append((current_time,
                                                       data_dict[sensor_type]["x"],
                                                       data_dict[sensor_type]["y"],
                                                       data_dict[sensor_type]["z"]))
            if abs(value_corrected) > threshold:
                last_active_time[sensor_type] = current_time
                updated = True
        elif sensor_type == "MAG":
            corrected_value = normalize_zero(0.0 if MAG_baseline.get(key, 1.0) == 0 else value / MAG_baseline[key])
            if abs(corrected_value) > threshold:
                data_dict[sensor_type][key] = corrected_value
                last_active_time[sensor_type] = current_time
                updated = True
            elif current_time - last_active_time[sensor_type] > RESET_TIMEOUT:
                if data_dict[sensor_type].get(key, 0.0) != 0.0:
                    data_dict[sensor_type][key] = 0.0
                    updated = True

    sensor_type = None
    if "ACC" in address:
        sensor_type = "ACC"
        process("ACC", ACC_THRESHOLD)
    elif "GYRO" in address:
        sensor_type = "GYRO"
        process("GYRO", GYRO_THRESHOLD)
    elif "MAG" in address:
        sensor_type = "MAG"
        process("MAG", MAG_THRESHOLD)

    with record_lock:
        recording = is_recording
    if recording:
        csv_row = {}
        for sensor in ["ACC", "GYRO", "MAG"]:
            for axis in ["x", "y", "z"]:
                value = data_dict.get(sensor, {}).get(axis, 0.0)
                csv_row[f"{sensor}_{axis.upper()}"] = f"{value:.4f}"
        now = datetime.now(BANGKOK_TZ)
        csv_row["Date"] = now.strftime('%d/%m/%Y')
        csv_row["Time"] = now.strftime('%H:%M:%S.%f')[:-3]
        csv_row["Activity"] = current_activity if current_activity else "unknown"
        csv_queue.put(csv_row)

    if baseline_updated:
        socketio.emit("baseline_data", baseline)

def csv_writer_thread():
    fieldnames = ["Date", "Time", "Activity"] + [f"{sensor}_{axis}" for sensor in ["ACC", "GYRO", "MAG"] for axis in ["X", "Y", "Z"]]
    try:
        if not os.access(os.path.dirname(current_raw_file) or '.', os.W_OK):
            logger.error(f"No write permission for file: {current_raw_file}")
            return
        with open(current_raw_file, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            csv_file.flush()
            logger.info(f"Created and initialized CSV file: {current_raw_file}")
        with open(current_raw_file, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            while True:
                row = csv_queue.get()
                if row is None:
                    logger.info("Received stop signal for CSV writer")
                    csv_file.flush()
                    break
                writer.writerow(row)
                csv_file.flush()
    except Exception as e:
        logger.error(f"Error in csv_writer_thread: {str(e)}")

@socketio.on("start_recording")
def start_recording(data):
    global is_recording, current_raw_file, current_resampled_file, csv_thread, current_activity
    activity = data["activity"]
    if activity not in ["walk", "bed to toilet"]:
        logger.error(f"Invalid activity: {activity}")
        socketio.emit("recording_stopped", {"status": "error", "message": "Invalid activity"})
        return

    with record_lock:
        if is_recording:
            logger.warning("Recording already in progress")
            return
        is_recording = True
        current_activity = activity
        while not csv_queue.empty():
            csv_queue.get()
        socketio.emit("activity_started", {"activity": activity})

    timestamp = datetime.now(BANGKOK_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    activity_dir = WALK_DIR if activity == "walk" else BED_DIR
    current_raw_file = os.path.join(activity_dir, f"{activity}_data_{timestamp}.csv")
    current_resampled_file = os.path.join(activity_dir, f"resampled_{activity}_data_{timestamp}.csv")
    logger.info(f"Recording started for {activity}. Raw file: {current_raw_file}, Resampled file: {current_resampled_file}")

    csv_thread = threading.Thread(target=csv_writer_thread, daemon=True)
    csv_thread.start()

@socketio.on("stop_recording")
def stop_recording():
    global is_recording, csv_thread, current_activity
    with record_lock:
        if not is_recording:
            logger.warning("No recording in progress")
            return
        is_recording = False

    logger.info("Stopping recording, waiting for CSV writer to process queue")
    while not csv_queue.empty():
        time.sleep(0.1)
    csv_queue.put(None)
    if csv_thread:
        csv_thread.join(timeout=2.0)
        logger.info("CSV writer thread joined")
    else:
        logger.warning("No CSV writer thread found")

    try:
        if os.path.exists(current_raw_file):
            file_size = os.path.getsize(current_raw_file)
            logger.info(f"Raw file size: {file_size} bytes")
            df = pd.read_csv(current_raw_file)
            logger.info(f"Raw CSV has {len(df)} rows")
            if len(df) > 0:
                with resample_lock:
                    try:
                        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S.%f')
                        df['timestamp'] = df['timestamp'].dt.tz_localize(BANGKOK_TZ)
                        df = df.drop_duplicates(subset=['timestamp'], keep='last')
                        df.set_index('timestamp', inplace=True)
                        numeric_cols = [f"{sensor}_{axis}" for sensor in ["ACC", "GYRO", "MAG"] for axis in ["X", "Y", "Z"]]
                        resampled_df = df[numeric_cols].resample('1S', closed='right', label='left').mean().fillna(0)
                        resampled_df['Activity'] = df['Activity'].resample('1S', closed='right', label='left').ffill()
                        if resampled_df.index[0] and pd.isna(resampled_df['Activity'].iloc[0]):
                            resampled_df['Activity'].iloc[0] = current_activity
                        resampled_df['Activity'] = resampled_df['Activity'].fillna(current_activity)
                        resampled_df['Date'] = resampled_df.index.strftime('%d/%m/%Y')
                        resampled_df['Time'] = resampled_df.index.strftime('%H:%M:%S')
                        cols = ['Date', 'Time', 'Activity'] + numeric_cols
                        resampled_df = resampled_df[cols]
                        resampled_df.to_csv(current_resampled_file, mode='w', index=False, float_format='%.4f')
                        logger.info(f"Resampled file saved: {current_resampled_file}")
                    except ValueError as e:
                        logger.error(f"Error parsing datetime in raw file: {str(e)}")
                        socketio.emit("recording_stopped", {"status": "error", "message": f"Error parsing datetime: {str(e)}"})
                        return
                socketio.emit("recording_stopped", {
                    "status": "success",
                    "download_urls": {
                        "raw": f"/download/raw/{current_activity}/{os.path.basename(current_raw_file)}",
                        "resampled": f"/download/resampled/{current_activity}/{os.path.basename(current_resampled_file)}"
                    }
                })
            else:
                logger.warning(f"Raw file {current_raw_file} is empty")
                cols = ['Date', 'Time', 'Activity'] + [f"{sensor}_{axis}" for sensor in ["ACC", "GYRO", "MAG"] for axis in ["X", "Y", "Z"]]
                empty_df = pd.DataFrame(columns=cols)
                with resample_lock:
                    empty_df.to_csv(current_resampled_file, mode='w', index=False)
                logger.info(f"Created empty resampled file: {current_resampled_file}")
                socketio.emit("recording_stopped", {
                    "status": "success",
                    "download_urls": {
                        "raw": f"/download/raw/{current_activity}/{os.path.basename(current_raw_file)}",
                        "resampled": f"/download/resampled/{current_activity}/{os.path.basename(current_resampled_file)}"
                    }
                })
        else:
            logger.error(f"Raw file not found: {current_raw_file}")
            cols = ['Date', 'Time', 'Activity'] + [f"{sensor}_{axis}" for sensor in ["ACC", "GYRO", "MAG"] for axis in ["X", "Y", "Z"]]
            empty_df = pd.DataFrame(columns=cols)
            with resample_lock:
                empty_df.to_csv(current_resampled_file, mode='w', index=False)
            logger.info(f"Created empty resampled file: {current_resampled_file}")
            socketio.emit("recording_stopped", {
                "status": "success",
                "download_urls": {
                    "raw": f"/download/raw/{current_activity}/{os.path.basename(current_raw_file)}",
                    "resampled": f"/download/resampled/{current_activity}/{os.path.basename(current_resampled_file)}"
                }
            })
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        socketio.emit("recording_stopped", {"status": "error", "message": str(e)})

@socketio.on("clear_data")
def clear_data():
    global data_dict, baseline
    for sensor in data_dict:
        for axis in data_dict[sensor]:
            data_dict[sensor][axis] = 0.0
    for sensor in baseline:
        for axis in baseline[sensor]:
            baseline[sensor][axis] = 0.0
    with buffer_lock:
        sensor_buffer["ACC"].clear()
        sensor_buffer["GYRO"].clear()
    socketio.emit("sensor_data", data_dict)
    logger.info("Data cleared.")

@socketio.on("set_mag_baseline")
def set_mag_baseline():
    global MAG_baseline
    for axis in ["x", "y", "z"]:
        current_value = data_dict["MAG"].get(axis, 1.0)
        if current_value != 0:
            MAG_baseline[axis] = current_value
    logger.info(f"Set MAG baseline: {MAG_baseline}")
    socketio.emit("mag_baseline_status", MAG_baseline)

def osc_server_thread():
    dispatcher = Dispatcher()
    dispatcher.set_default_handler(print_all)
    server = BlockingOSCUDPServer((OSC_IP, OSC_PORT), dispatcher)
    logger.info(f"OSC Server started at {OSC_IP}:{OSC_PORT}")
    server.serve_forever()

if __name__ == "__main__":
    threading.Timer(1.0, compute_mean_and_emit).start()
    osc_thread = threading.Thread(target=osc_server_thread, daemon=True)
    osc_thread.start()
    socketio.run(app, host="0.0.0.0", port=5000)