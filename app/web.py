"""Flask 应用工厂与路由。"""
from __future__ import annotations

import io
import os
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from . import config, export, services
from .db import get_db, init_db, recover_interrupted_batches


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

    init_db()
    recover_interrupted_batches()

    @app.get("/")
    def index():
        return render_template("index.html", batches=services.list_batches())

    @app.post("/upload")
    def upload():
        files = request.files.getlist("dicom_files")
        if not files or all(not f.filename for f in files):
            flash("请至少选择一个 DICOM 文件", "error")
            return redirect(url_for("index"))
        try:
            batch_id, saved = services.save_uploads(files)
        except Exception as exc:  # noqa: BLE001
            flash(f"上传暂存失败: {exc}", "error")
            return redirect(url_for("index"))
        try:
            services.run_batch(batch_id, saved)
        except services.BatchError as exc:
            flash(str(exc), "error")
            return redirect(url_for("batch_detail", batch_id=batch_id))
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("批次 %s 处理异常", batch_id)
            flash(f"处理失败: {exc}", "error")
            return redirect(url_for("batch_detail", batch_id=batch_id))
        flash(f"批次 {batch_id} 处理完成", "ok")
        return redirect(url_for("batch_detail", batch_id=batch_id))

    @app.get("/batches/<batch_id>")
    def batch_detail(batch_id: str):
        batch = services.get_batch(batch_id)
        if batch is None:
            abort(404)
        conn = get_db()
        try:
            files = conn.execute(
                "SELECT * FROM batch_files WHERE batch_id=? ORDER BY orig_name",
                (batch_id,),
            ).fetchall()
        finally:
            conn.close()
        return render_template("batch.html", batch=batch, files=files)

    @app.get("/batches/<batch_id>/download")
    def download(batch_id: str):
        batch = services.get_batch(batch_id)
        if batch is None:
            abort(404)
        if batch["status"] != "ready" or not batch["zip_path"]:
            # 半成品绝不提供下载
            abort(409, description="批次尚未处理成功，无可下载文件包")
        zip_path = Path(batch["zip_path"])
        if not zip_path.exists():
            abort(404, description="文件包缺失")
        return send_file(
            zip_path,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"batch_{batch_id}.zip",
        )

    @app.get("/export")
    def export_page():
        return render_template(
            "export.html", batches=export.list_batches_with_sr_count()
        )

    @app.post("/export")
    def export_run():
        batch_ids = request.form.getlist("batch_ids")
        if not batch_ids:
            flash("请至少勾选一个处理成功的批次", "error")
            return redirect(url_for("export_page"))
        try:
            csv_bytes, stats = export.export_measurements(batch_ids)
        except export.ExportError as exc:
            # 整次拒绝：不产出任何下载内容，回到导出页说明原因
            flash(f"导出被拒绝：{exc}", "error")
            return redirect(url_for("export_page"))
        if stats["measurement_count"] == 0:
            flash("所选批次的 SR 中没有 NUM 数值测量项，未生成导出文件", "error")
            return redirect(url_for("export_page"))
        from datetime import datetime

        filename = f"sr_measurements_{datetime.now():%Y%m%d_%H%M%S}.csv"
        flash(
            f"导出完成：{stats['sr_count']} 份 SR、"
            f"{stats['measurement_count']} 项测量，"
            f"其中 {stats['linked_count']} 项已关联唯一影像证据",
            "ok",
        )
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv; charset=utf-8",
            as_attachment=True,
            download_name=filename,
        )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.errorhandler(413)
    def too_large(_err):
        flash("上传内容超过服务端大小限制", "error")
        return redirect(url_for("index"))

    return app
