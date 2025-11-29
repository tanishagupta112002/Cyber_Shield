import threading
import time
import json
import os
import sqlite3
import random
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response, stream_with_context
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from extensions import db
from routes.logs import logs_bp, api_bp
from log_tamper_detector import predict_logs_dataframe

# --- Import agent monitoring ---
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "log_agent")))
from agent import start_multi_log_monitoring  # ✅ Agent function

# --- Import Models ---
from models import LogEvent, LogFile   # ✅ FIXED

# ------------------- App Setup -------------------
app = Flask(
    __name__,
    template_folder="webapp/templates",
    static_folder="webapp/static"
)
app.secret_key = "supersecretkey"

DB_NAME = "users.db"

# ------------ Mail Config -------------
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = "tanishagupta12345678@gmail.com"
app.config['MAIL_PASSWORD'] = "gnej rsia ajih axqn"  # ideally load from env var
mail = Mail(app)

# ------------ SQLAlchemy Config (Logs DB) -------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://loguser:Tanisha123@127.0.0.1:3308/log_detection'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# ------------ User DB (manual SQLite) -------------
def init_user_db():
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
init_user_db()

# ------------------- Blueprint Registration -------------------
app.register_blueprint(logs_bp)
app.register_blueprint(api_bp)
print(">>> Blueprints registered: logs_bp + api_bp")

# ------------------- Utility: Hash Check -------------------
import hashlib
def check_hash(row, prev_hash):
    """verify hash chain + recompute sha256"""
    data = f"{row['Filename']},{row['Event_ID']},{row['Level']},{row['Service']},{row['Timestamp']},{row['TimeGenerated']}"
    computed_hash = hashlib.sha256(data.encode()).hexdigest()
    if row["Hash"] != computed_hash:
        return "Tampered"
    if prev_hash and row["Prev_Hash"] != prev_hash:
        return "Tampered"
    return "Normal"

# ------------------- Final Classify -------------------
def classify_row(row, prev_hash):
    ai_class = row.get("Class", "Normal")
    hash_status = check_hash(row, prev_hash)
    if hash_status == "Tampered":
        return "Tampered"
    elif ai_class == "Suspicious":
        return "Suspicious"
    else:
        return "Normal"

# ------------------- AI Log Analyzer -------------------
TAMPER_KEYWORDS = ["TAMPER", "MODIFIED", "CORRUPT", "DELETED"]
SUSPICIOUS_KEYWORDS = ["WARN", "SUSPICIOUS", "FAILED", "MALFORMED"]
NORMAL_KEYWORDS = ["INFO", "OK", "SUCCESS", "START"]

def analyze_line(line: str):
    line_upper = line.upper()
    status = "Normal"
    confidence = 95

    # Count keyword matches
    tamper_count = sum(k in line_upper for k in TAMPER_KEYWORDS)
    suspicious_count = sum(k in line_upper for k in SUSPICIOUS_KEYWORDS)
    normal_count = sum(k in line_upper for k in NORMAL_KEYWORDS)

    # Determine status based on matches
    if tamper_count > 0:
        status = "Tampered"
        confidence = 90 + tamper_count * 2  # 90–99% depending on number of keywords
        confidence = min(confidence, 99)
    elif suspicious_count > 0:
        status = "Suspicious"
        confidence = 60 + suspicious_count * 5  # 60–85% depending on keywords
        confidence = min(confidence, 85)
    elif normal_count > 0:
        status = "Normal"
        confidence = 90 + normal_count * 2  # 90–98%
        confidence = min(confidence, 98)
    else:
        # If no keyword matches, mark as Suspicious with lower confidence
        status = "Suspicious"
        confidence = 50

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "filename": line[:30] + ("..." if len(line) > 30 else ""),
        "status": status,
        "confidence": f"{confidence}%",
        "timestamp": timestamp
    }


# ------------------- Routes -------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST" and "step" not in request.form:
        email = request.form["email"]
        password = request.form["password"]
        otp = str(random.randint(100000, 999999))
        session["signup_data"] = {
            "email": email,
            "password": generate_password_hash(password),
            "otp": otp
        }
        try:
            msg = Message("Your OTP Code", sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f"Your OTP is: {otp}"
            mail.send(msg)
            flash("OTP sent to your email. Please verify.", "info")
        except Exception as e:
            flash(f"Error sending email: {e}", "danger")
        return render_template("signup.html", step="otp")

    if request.method == "POST" and request.form.get("step") == "otp":
        user_otp = request.form["otp"]
        data = session.get("signup_data")
        if data and user_otp == data["otp"]:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            try:
                c.execute(
                    "INSERT INTO users (email, password) VALUES (?, ?)", 
                    (data["email"], data["password"])
                )
                conn.commit()
                flash("Account created successfully! Please login.", "success")
                session.pop("signup_data", None)
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Email already registered!", "danger")
            finally:
                conn.close()
        else:
            flash("Invalid OTP, try again.", "danger")
        return render_template("signup.html", step="otp")

    return render_template("signup.html", step="signup")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE email = ?", (email,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[1], password):
            session["user_id"] = user[0]
            session["email"] = email
            flash("Login successful!", "success")
            return redirect(url_for("profile"))
        else:
            flash("Invalid email or password.", "danger")
    return render_template("login.html")

@app.route("/profile")
def profile():
    if "user_id" not in session:
        flash("Please login first.", "danger")
        return redirect(url_for("login"))
    return render_template("profile.html", email=session["email"])

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/connect")
def connect():
    return render_template("connect.html")

@app.route("/protect")
def protect():
    return render_template("protect.html")

@app.route("/forensics")
def forensics():
    return render_template("forensics.html")

@app.route("/awareness")
def awareness():
    return render_template("awareness.html")

# ------------------- API Routes -------------------

@app.route("/api/analyze-log", methods=["POST"])
def analyze_log():
    file = request.files.get("logfile")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    content = file.read().decode("utf-8", errors="ignore")
    analyzed_logs = [analyze_line(line) for line in content.splitlines() if line.strip()]
    return jsonify(analyzed_logs)

@app.route("/ingest_logs", methods=["POST"])
def ingest_logs():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "No JSON received"}), 400
    df = pd.DataFrame(payload)
    required = {"Filename", "Event_ID", "Level", "Service", "Timestamp", "TimeGenerated", "Hash", "Class"}
    if not required.issubset(set(df.columns)):
        return jsonify({"error": "Missing required fields"}), 400

    df_with_preds = predict_logs_dataframe(df)
    cleaned_results = []
    prev_hash = None
    for row in df_with_preds:
        status = classify_row(row, prev_hash)
        prev_hash = row.get("Hash", "")
        confidence = round(float(row.get('Confidence', 0)), 2)
        cleaned_results.append({
            "filename": row.get("Filename"),
            "eventid": row.get("Event_ID"),
            "level": row.get("Level"),
            "source_name": row.get("Service"),
            "time_generated": row.get("TimeGenerated"),
            "hash": row.get("Hash"),
            "ai_class": row.get("Class"),
            "hash_check": check_hash(row, prev_hash),
            "display_status": status,
            "confidence": confidence
        })
        print(row.get("Filename"), row.get("Class"), "->", status)
    return jsonify(cleaned_results), 200

@app.route("/logs/<system_id>")
def logs_page(system_id):
    return render_template("logs.html", system_id=system_id)

@app.route("/api/log-events/<system_id>", methods=["GET"])
def get_log_events(system_id):
    logs = LogEvent.query.join(LogFile).filter(LogFile.system_id == system_id).order_by(LogEvent.id.desc()).all()
    result = []
    for log in logs:
        result.append({
            "id": log.id,
            "filename": log.log_file.filename,
            "seq_no": log.seq_no,
            "hash": log.hash,
            "prev_hash": log.prev_hash,
            "status": log.status,
            "tamper_reason": log.tamper_reason,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "eventid": log.eventid,
            "level": log.level,
            "source_name": log.source_name,
            "time_generated": log.time_generated,
            "display_status": log.status or "Normal"  # ✅ added for frontend
        })
    return jsonify(result)

@app.route("/api/verify-hashchain/<system_id>", methods=["POST"])
def verify_hashchain(system_id):
    logs = LogEvent.query.join(LogFile).filter(LogFile.system_id == system_id).order_by(LogEvent.seq_no).all()
    tampered_rows = []
    prev_hash = None
    for log in logs:
        if prev_hash and log.prev_hash != prev_hash:
            log.status = "Tampered"
            log.tamper_reason = "Hash chain broken"
            tampered_rows.append(log.id)
        prev_hash = log.hash
    db.session.commit()
    return jsonify({"status": "ok", "tampered_rows": tampered_rows})

# ------------------- SSE Stream -------------------
@app.route("/stream/<system_id>")
def stream_logs(system_id):
    def event_stream():
        last_id = 0
        while True:
            with app.app_context():
                events = (
                    LogEvent.query.join(LogFile)
                    .filter(LogFile.system_id == system_id, LogEvent.id > last_id)
                    .order_by(LogEvent.id)
                    .all()
                )
                for ev in events:
                    last_id = ev.id
                    data = {
                        "filename": ev.log_file.filename,
                        "seq_no": ev.seq_no,
                        "status": ev.status,
                        "hash": ev.hash,
                        "prev_hash": ev.prev_hash,
                        "timestamp": ev.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "eventid": ev.eventid,
                        "level": ev.level,
                        "source_name": ev.source_name,
                        "time_generated": ev.time_generated,
                        "tamper_reason": ev.tamper_reason,
                        "display_status": ev.status or "Normal"
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")
    

# ------------------- Run App -------------------
if __name__ == "__main__":
    print("Flask app is running from:", __file__)
    with app.app_context():
        db.create_all()
        t = threading.Thread(target=start_multi_log_monitoring, daemon=True)
        t.start()
        print("[INFO] Agent monitoring started in background")
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
