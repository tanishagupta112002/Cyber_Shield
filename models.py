from datetime import datetime
from extensions import db

class LogFile(db.Model):
    __tablename__ = "log_files"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)
    system_id = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    latest_hash = db.Column(db.String(64), nullable=True)
    last_seen_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint("system_id", "filename", name="_system_file_uc"),)

    events = db.relationship(
        "LogEvent",
        backref="log_file",
        lazy=True,
        cascade="all, delete-orphan"
    )


class LogEvent(db.Model):
    __tablename__ = "log_events"

    id = db.Column(db.Integer, primary_key=True)
    log_file_id = db.Column(db.Integer, db.ForeignKey("log_files.id"), nullable=False)

    hash = db.Column(db.String(64), nullable=False)
    prev_hash = db.Column(db.String(64), nullable=True)
    seq_no = db.Column(db.Integer, nullable=False, default=0)

    status = db.Column(db.String(20), default="ok")
    tamper_reason = db.Column(db.Text, nullable=True)

    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    time_generated = db.Column(db.String(50), nullable=True)

    eventid = db.Column(db.Integer, nullable=True)
    level = db.Column(db.String(50), nullable=True)
    source_name = db.Column(db.String(255), nullable=True)
    message = db.Column(db.Text, nullable=True)

    confidence = db.Column(db.Float, nullable=True)
