from flask import Blueprint, request, jsonify, current_app, render_template
from extensions import db
from models import LogFile, LogEvent
from datetime import datetime
from hashlib import sha256
import pandas as pd
from log_tamper_detector import predict_logs_dataframe

# -----------------------------
# Blueprint definitions
# -----------------------------
logs_bp = Blueprint("logs", __name__)                        # Frontend routes
api_bp = Blueprint("logs_api", __name__, url_prefix="/api")  # API routes for agent

# -----------------------------
# Frontend: Monitoring dashboard
# -----------------------------
@logs_bp.route("/logs/<system_id>")
def logs_dashboard(system_id):
    return render_template("logs.html", system_id=system_id)

# -----------------------------
# Canonicalization helper (MUST MATCH AGENT)
# -----------------------------
def canonicalize_event(seq_no, time_generated, eventid, level, source_name, message):
    fields = [
        str(seq_no),
        str(time_generated or ""),
        str(eventid or ""),
        str(level or ""),
        str(source_name or ""),
        str(message or "")
    ]
    return "||".join(fields)

# -----------------------------
# API: Receive new log event
# -----------------------------
@api_bp.route("/log-event", methods=["POST"])
def log_event():
    try:
        data = request.get_json()
        filename = data.get("filename")
        seq_no = int(data.get("seq_no", 0))
        system_id = data.get("system_id")
        user_id = int(data.get("user_id", 1))
        eventid = data.get("eventid")
        level = data.get("level")
        source_name = data.get("source_name")
        time_generated = data.get("time_generated")
        message = data.get("message", "")

        if not filename or not system_id:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        with current_app.app_context():
            # --- Get or create LogFile ---
            log_file = LogFile.query.filter_by(filename=filename, system_id=system_id).first()
            if not log_file:
                log_file = LogFile(
                    filename=filename,
                    system_id=system_id,
                    user_id=user_id,
                    latest_hash="",
                    last_seen_at=datetime.utcnow()
                )
                db.session.add(log_file)
                db.session.commit()

            # --- Determine prev_hash ---
            prev_hash = data.get("prev_hash", "")
            if not prev_hash:
                last_event = LogEvent.query.filter_by(log_file_id=log_file.id)\
                            .order_by(LogEvent.seq_no.desc()).first()
                prev_hash = last_event.hash if last_event else ""

            # --- Canonical payload + compute hash ---
            payload = canonicalize_event(seq_no, time_generated, eventid, level, source_name, message)
            computed_hash = sha256(((prev_hash or "") + payload).encode("utf-8")).hexdigest()

            # --- AI Detection ---
            df_pred = pd.DataFrame([{
                "Filename": filename,
                "Event_ID": eventid or 0,
                "Level": level or 0,
                "Service": source_name or "Unknown",
                "Timestamp": str(datetime.utcnow()),
                "TimeGenerated": str(time_generated or datetime.utcnow()),
                "Hash": computed_hash,
                "Seq_No": seq_no,
                "Prev_Hash": prev_hash,
                "Message": message
            }])
            results = predict_logs_dataframe(df_pred)
            if results and len(results) == 1:
                ai_status = results[0].get("Predicted", "ok")
                confidence = results[0].get("Confidence")
            else:
                ai_status = "ok"
                confidence = None

            # --- Tamper check & final status ---
            tamper_reason = None
            incoming_status = "Normal"  # default

            # 1️⃣ Check last stored event hash
            last_stored_event = LogEvent.query.filter_by(log_file_id=log_file.id)\
                                .order_by(LogEvent.seq_no.desc()).first()
            if last_stored_event:
                expected_last_hash = sha256(((last_stored_event.prev_hash or "") +
                                             canonicalize_event(last_stored_event.seq_no,
                                                                last_stored_event.time_generated,
                                                                last_stored_event.eventid,
                                                                last_stored_event.level,
                                                                last_stored_event.source_name,
                                                                last_stored_event.message)).encode("utf-8")).hexdigest()
                if last_stored_event.hash.strip().lower() != expected_last_hash.strip().lower():
                    last_stored_event.status = "Tampered"
                    last_stored_event.tamper_reason = "Incoming hash differs from stored"
                    db.session.add(last_stored_event)

            # 2️⃣ Check incoming event hash vs prev_hash
            expected_hash = sha256(((prev_hash or "") + payload).encode("utf-8")).hexdigest()
            if computed_hash != expected_hash:
                incoming_status = "Tampered"
                tamper_reason = f"Hash mismatch: expected {expected_hash[:12]}..., got {computed_hash[:12]}..."
            else:
                # 3️⃣ Hash is fine → rely on AI
                if ai_status.lower() == "ok":
                    incoming_status = "Normal"
                else:
                    incoming_status = "Suspicious"

            # --- Save new event ---
            log_event_entry = LogEvent(
                log_file_id=log_file.id,
                hash=computed_hash,
                prev_hash=prev_hash,
                seq_no=seq_no,
                status=incoming_status,
                timestamp=datetime.utcnow(),
                eventid=eventid,
                level=level,
                source_name=source_name,
                time_generated=time_generated,
                message=message,
                tamper_reason=tamper_reason
            )
            db.session.add(log_event_entry)

            # --- Update LogFile ---
            log_file.latest_hash = computed_hash
            log_file.last_seen_at = datetime.utcnow()
            db.session.commit()

        return jsonify({
            "status": "ok",
            "message": "Log event recorded",
            "predicted_status": incoming_status,
            "confidence": confidence,
            "computed_hash": computed_hash
        })

    except Exception as e:
        print("ERROR in log_event:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------
# API: Verify hash chain
# -----------------------------
@api_bp.route("/verify-hashchain/<system_id>", methods=["POST"])
def verify_hashchain(system_id):
    try:
        events = LogEvent.query.join(LogFile)\
                    .filter(LogFile.system_id == system_id)\
                    .order_by(LogEvent.seq_no.asc()).all()

        prev_hash = ""
        tampered_ids = []

        for e in events:
            payload = canonicalize_event(e.seq_no, e.time_generated, e.eventid, e.level, e.source_name, e.message)
            expected_hash = sha256(((prev_hash or "") + payload).encode("utf-8")).hexdigest()
            actual_hash = (e.hash or "").strip().lower()

            if actual_hash != expected_hash:
                e.status = "Tampered"
                e.tamper_reason = f"Hash mismatch (expected {expected_hash[:12]}... got {actual_hash[:12]}...)"
                tampered_ids.append(e.id)
                prev_hash = actual_hash or prev_hash
            else:
                if e.status != "Tampered":
                    e.status = e.status or "Normal"
                e.tamper_reason = None
                prev_hash = actual_hash

        db.session.commit()
        return jsonify({"status": "ok", "tampered_rows": tampered_ids})

    except Exception as ex:
        print("ERROR in verify_hashchain:", ex)
        return jsonify({"status": "error", "message": str(ex)}), 500


# # logs.py
# from flask import Blueprint, request, jsonify, current_app, render_template
# from extensions import db
# from models import LogFile, LogEvent
# from datetime import datetime
# from hashlib import sha256
# import pandas as pd
# from log_tamper_detector import predict_logs_dataframe

# # -----------------------------
# # Blueprint definitions
# # -----------------------------
# logs_bp = Blueprint("logs", __name__)                        # Frontend routes
# api_bp = Blueprint("logs_api", __name__, url_prefix="/api")  # API routes for agent

# # -----------------------------
# # Frontend: Monitoring dashboard
# # -----------------------------
# @logs_bp.route("/logs/<system_id>")
# def logs_dashboard(system_id):
#     return render_template("logs.html", system_id=system_id)

# # -----------------------------
# # Canonicalization helper (MUST MATCH AGENT)
# # -----------------------------
# def canonicalize_event(seq_no, time_generated, eventid, level, source_name, message):
#     fields = [
#         str(seq_no),
#         str(time_generated or ""),
#         str(eventid or ""),
#         str(level or ""),
#         str(source_name or ""),
#         str(message or "")
#     ]
#     return "||".join(fields)

# # -----------------------------
# # API: Receive new log event
# # -----------------------------
# @api_bp.route("/log-event", methods=["POST"])
# def log_event():
#     try:
#         data = request.get_json()
#         filename = data.get("filename")
#         seq_no = int(data.get("seq_no", 0))
#         system_id = data.get("system_id")
#         user_id = int(data.get("user_id", 1))
#         eventid = data.get("eventid")
#         level = data.get("level")
#         source_name = data.get("source_name")
#         time_generated = data.get("time_generated")
#         message = data.get("message", "")

#         if not filename or not system_id:
#             return jsonify({"status": "error", "message": "Missing required fields"}), 400

#         with current_app.app_context():
#             # --- Get or create LogFile ---
#             log_file = LogFile.query.filter_by(filename=filename, system_id=system_id).first()
#             if not log_file:
#                 log_file = LogFile(
#                     filename=filename,
#                     system_id=system_id,
#                     user_id=user_id,
#                     latest_hash="",
#                     last_seen_at=datetime.utcnow()
#                 )
#                 db.session.add(log_file)
#                 db.session.commit()

#             # --- Determine prev_hash ---
#             prev_hash = data.get("prev_hash", "")
#             if not prev_hash:
#                 last_event = LogEvent.query.filter_by(log_file_id=log_file.id)\
#                             .order_by(LogEvent.seq_no.desc()).first()
#                 prev_hash = last_event.hash if last_event else ""

#             # --- Canonical payload + compute hash ---
#             payload = canonicalize_event(seq_no, time_generated, eventid, level, source_name, message)
#             computed_hash = sha256(((prev_hash or "") + payload).encode("utf-8")).hexdigest()

#             # --- AI Detection ---
#             df_pred = pd.DataFrame([{
#                 "Filename": filename,
#                 "Event_ID": eventid or 0,
#                 "Level": level or 0,
#                 "Service": source_name or "Unknown",
#                 "Timestamp": str(datetime.utcnow()),
#                 "TimeGenerated": str(time_generated or datetime.utcnow()),
#                 "Hash": computed_hash,
#                 "Seq_No": seq_no,
#                 "Prev_Hash": prev_hash,
#                 "Message": message
#             }])
#             results = predict_logs_dataframe(df_pred)
#             if results and len(results) == 1:
#                 ai_status = results[0].get("Predicted", "ok")
#                 confidence = results[0].get("Confidence")
#             else:
#                 ai_status = "ok"
#                 confidence = None

#             # --- Tamper check & final status ---
#             tamper_reason = None
#             incoming_status = "Normal"  # default

#             # 1️⃣ Check if last stored event hash was modified
#             last_stored_event = LogEvent.query.filter_by(log_file_id=log_file.id)\
#                                 .order_by(LogEvent.seq_no.desc()).first()
#             if last_stored_event:
#                 expected_last_hash = sha256(((last_stored_event.prev_hash or "") +
#                                              canonicalize_event(last_stored_event.seq_no,
#                                                                 last_stored_event.time_generated,
#                                                                 last_stored_event.eventid,
#                                                                 last_stored_event.level,
#                                                                 last_stored_event.source_name,
#                                                                 last_stored_event.message)).encode("utf-8")).hexdigest()
#                 if last_stored_event.hash.strip().lower() != expected_last_hash.strip().lower():
#                     last_stored_event.status = "Tampered"
#                     last_stored_event.tamper_reason = "Stored hash mismatch"
#                     db.session.add(last_stored_event)

#             # 2️⃣ Check incoming event hash chain
#             expected_hash = sha256(((prev_hash or "") + payload).encode("utf-8")).hexdigest()
#             if computed_hash != expected_hash:
#                 incoming_status = "Tampered"
#                 tamper_reason = f"Hash mismatch: expected {expected_hash[:12]}..., got {computed_hash[:12]}..."
#             else:
#                 # 3️⃣ If hash is fine, check AI prediction
#                 incoming_status = "Normal" if ai_status.lower() == "ok" else "Suspicious"

#             # --- Save new event ---
#             log_event_entry = LogEvent(
#                 log_file_id=log_file.id,
#                 hash=computed_hash,
#                 prev_hash=prev_hash,
#                 seq_no=seq_no,
#                 status=incoming_status,
#                 timestamp=datetime.utcnow(),
#                 eventid=eventid,
#                 level=level,
#                 source_name=source_name,
#                 time_generated=time_generated,
#                 message=message,
#                 tamper_reason=tamper_reason
#             )
#             db.session.add(log_event_entry)

#             # --- Update LogFile ---
#             log_file.latest_hash = computed_hash
#             log_file.last_seen_at = datetime.utcnow()
#             db.session.commit()

#         return jsonify({
#             "status": "ok",
#             "message": "Log event recorded",
#             "predicted_status": incoming_status,
#             "confidence": confidence,
#             "computed_hash": computed_hash
#         })

#     except Exception as e:
#         print("ERROR in log_event:", e)
#         return jsonify({"status": "error", "message": str(e)}), 500

# # -----------------------------
# # API: Verify hash chain
# # -----------------------------
# @api_bp.route("/verify-hashchain/<system_id>", methods=["POST"])
# def verify_hashchain(system_id):
#     try:
#         events = LogEvent.query.join(LogFile)\
#                     .filter(LogFile.system_id == system_id)\
#                     .order_by(LogEvent.seq_no.asc()).all()

#         prev_hash = ""
#         tampered_ids = []

#         for e in events:
#             payload = canonicalize_event(e.seq_no, e.time_generated, e.eventid, e.level, e.source_name, e.message)
#             expected_hash = sha256(((prev_hash or "") + payload).encode("utf-8")).hexdigest()
#             actual_hash = (e.hash or "").strip().lower()

#             if actual_hash != expected_hash:
#                 e.status = "Tampered"
#                 e.tamper_reason = f"Hash mismatch (expected {expected_hash[:12]}... got {actual_hash[:12]}...)"
#                 tampered_ids.append(e.id)
#                 prev_hash = actual_hash or prev_hash
#             else:
#                 if e.status != "Tampered":
#                     e.status = e.status or "Normal"
#                 e.tamper_reason = None
#                 prev_hash = actual_hash

#         db.session.commit()
#         return jsonify({"status": "ok", "tampered_rows": tampered_ids})

#     except Exception as ex:
#         print("ERROR in verify_hashchain:", ex)
#         return jsonify({"status": "error", "message": str(ex)}), 500


