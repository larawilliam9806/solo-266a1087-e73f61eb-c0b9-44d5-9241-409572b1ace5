"""已发布 SR 数值测量导出。

业务规则（见需求）：
- 范围仅限用户勾选的 **ready（成功）批次** 所交付的实例；失败批次遗留对象、
  未勾选批次的对象都不算交付证据。
- 仅展开 SR 文档（SOP Class 属于 1.2.840.10008.5.1.4.1.1.88.* 家族）
  ContentSequence 中的 NUM 节点，递归遍历整棵内容树。
- 同一原 SOP 的 SR 即便被多个勾选批次复用，也只展开一次。
- 每个 NUM 只关联“同一父 ContentSequence 直接子节点”中的 IMAGE 证据：
  0 个 -> no_evidence；多于 1 个 -> multiple_evidence；恰有 1 个再校验
  引用 UID 是否由所选成功批次交付、对象文件是否存在、引用类 UID 是否与
  交付实例的 SOP Class UID 相符，分别给出 linked / missing / class_mismatch。
- 导出是按需实时计算、不缓存：补传影像并形成新的成功批次后重新导出，
  关联结果自然更新；既有批次的 zip 下载不受影响。
- 任一 SR 对象缺失、解析失败，或任一 NUM 必需字段不全，整次导出拒绝
  （不产出半成品 CSV）。
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom

from . import config
from .db import get_db

# SR 文档存储类全部位于该弧下（Enhanced/Comprehensive/Basic Text/Key Object
# Selection/CAD/Procedure Log 等，见 pydicom.uid 中 88.* 家族）
SR_CLASS_PREFIX = "1.2.840.10008.5.1.4.1.1.88."

# 证据状态机内码（写入 CSV，稳定供数据湖消费）
STATUS_LINKED = "linked"               # 唯一证据且交付、类相符
STATUS_MISSING = "missing"             # 缺件：未交付/对象缺失/引用不完整
STATUS_CLASS_MISMATCH = "class_mismatch"  # 类不符
STATUS_NO_EVIDENCE = "no_evidence"     # 同组无 IMAGE
STATUS_MULTIPLE = "multiple_evidence"  # 同组多于一个 IMAGE

STATUS_LABELS = {
    STATUS_LINKED: "已关联",
    STATUS_MISSING: "缺件",
    STATUS_CLASS_MISMATCH: "类不符",
    STATUS_NO_EVIDENCE: "无证据",
    STATUS_MULTIPLE: "多个证据",
}

CSV_COLUMNS = [
    "anonymized_patient_id",
    "source_sr_sop_uid",
    "content_tree_path",
    "container_code_scheme",
    "container_code_value",
    "container_code_meaning",
    "concept_code_scheme",
    "concept_code_value",
    "concept_code_meaning",
    "numeric_value",
    "unit_code_scheme",
    "unit_code_value",
    "unit_code_meaning",
    "evidence_status",
    "evidence_status_label",
    "evidence_reason",
    "evidence_image_count",
    "evidence_image_sop_uid",
    "evidence_sop_class_uid",
]


class ExportError(Exception):
    """整次导出拒绝类错误。"""


@dataclass
class _Delivery:
    """所选成功批次交付集合的快查视图。"""

    sop_uids: set[str]                              # 原 SOP UID 集合
    by_new_sop: dict[str, sqlite3_row]              # 新 SOP UID -> instances 行
    objects_dir: Path


# 轻量类型别名，避免在运行时强依赖 sqlite3.Row 做注解
sqlite3_row = Any


# ---------------------------------------------------------------- 基础工具
def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _code_triple(item: pydicom.Dataset, seq_name: str) -> tuple[str, str, str]:
    """取 ConceptNameCodeSequence 等编码序列首项的 (方案, 值, 含义)。"""
    seq = getattr(item, seq_name, None)
    if not seq:
        return "", "", ""
    code = seq[0]
    return (
        _text(getattr(code, "CodingSchemeDesignator", "")),
        _text(getattr(code, "CodeValue", "")),
        _text(getattr(code, "CodeMeaning", "")),
    )


def _is_sr_class(sop_class_uid: str) -> bool:
    return bool(sop_class_uid) and sop_class_uid.startswith(SR_CLASS_PREFIX)


def _backfill_sop_class(conn, row: sqlite3_row) -> str:
    """旧库行没有 sop_class_uid 时从对象文件懒读取并回填。

    新部署的实例在入库时即带类 UID，本函数只服务历史数据升级。
    """
    class_uid = row["sop_class_uid"] or ""
    if class_uid:
        return class_uid
    obj_path = config.get_objects_dir() / row["object_path"]
    try:
        ds = pydicom.dcmread(str(obj_path), stop_before_pixels=True)
        class_uid = _text(getattr(ds, "SOPClassUID", ""))
    except Exception:  # noqa: BLE001 - 读不出来就留空，证据校验时按缺件处理
        class_uid = ""
    if class_uid:
        conn.execute(
            "UPDATE instances SET sop_class_uid=? WHERE sop_uid=?",
            (class_uid, row["sop_uid"]),
        )
    return class_uid


# ---------------------------------------------------------------- 范围装载
def _load_delivery(conn, batch_ids: list[str]) -> _Delivery:
    if not batch_ids:
        raise ExportError("未选择任何批次")

    placeholders = ",".join("?" * len(batch_ids))
    selected = conn.execute(
        f"SELECT id, status FROM batches WHERE id IN ({placeholders})",
        tuple(batch_ids),
    ).fetchall()
    found = {r["id"]: r["status"] for r in selected}
    missing = [b for b in batch_ids if b not in found]
    if missing:
        raise ExportError(f"批次不存在: {', '.join(missing)}")
    not_ready = [b for b, st in found.items() if st != "ready"]
    if not_ready:
        # 失败/处理中的批次绝不进入交付集合
        raise ExportError(
            "只能选择处理成功（ready）的批次，以下批次不符合: "
            + ", ".join(not_ready)
        )

    delivered_rows = conn.execute(
        f"""
        SELECT DISTINCT i.sop_uid, i.new_sop_uid, i.object_path, i.sop_class_uid
        FROM batch_files bf
        JOIN instances i ON i.sop_uid = bf.sop_uid
        WHERE bf.batch_id IN ({placeholders})
        ORDER BY i.new_sop_uid
        """,
        tuple(batch_ids),
    ).fetchall()

    sop_uids: set[str] = set()
    by_new_sop: dict[str, sqlite3_row] = {}
    for row in delivered_rows:
        sop_uids.add(row["sop_uid"])
        if not row["sop_class_uid"]:
            # sqlite3.Row 不支持直接赋值更新，借助可变包装字典承载回填值
            row = _mutable(row)
            row["sop_class_uid"] = _backfill_sop_class(conn, row)
        by_new_sop[row["new_sop_uid"]] = row
    return _Delivery(
        sop_uids=sop_uids,
        by_new_sop=by_new_sop,
        objects_dir=config.get_objects_dir(),
    )


def _mutable(row: sqlite3_row) -> dict:
    return dict(row)


# ---------------------------------------------------------------- 证据判定
def _image_refs(image_item: pydicom.Dataset) -> list[tuple[str, str]]:
    """取 IMAGE 内容项 ReferencedSOPSequence 的 (类 UID, 实例 UID) 对。"""
    refs = getattr(image_item, "ReferencedSOPSequence", None) or []
    pairs: list[tuple[str, str]] = []
    for ref in refs:
        pairs.append((
            _text(getattr(ref, "ReferencedSOPClassUID", "")),
            _text(getattr(ref, "ReferencedSOPInstanceUID", "")),
        ))
    return pairs


def _evaluate_evidence(
    image_item: pydicom.Dataset, delivery: _Delivery
) -> dict[str, Any]:
    refs = _image_refs(image_item)
    # 引用本身不完整（无序列/缺实例 UID）按缺件处理，并在 reason 中说明
    if not refs or any(not inst for _, inst in refs):
        return {
            "status": STATUS_MISSING,
            "reason": "IMAGE 证据缺少完整的 ReferencedSOPSequence/ReferencedSOPInstanceUID",
            "image_sop_uid": "",
            "sop_class_uid": refs[0][0] if refs else "",
        }
    distinct_inst = {inst for _, inst in refs}
    if len(distinct_inst) > 1:
        # 单个 IMAGE 节点引用了多个不同 SOP，证据不唯一
        return {
            "status": STATUS_MULTIPLE,
            "reason": "单个 IMAGE 节点引用了多个不同 SOP，证据不唯一",
            "image_sop_uid": "",
            "sop_class_uid": "",
        }
    ref_class, ref_inst = refs[0]

    row = delivery.by_new_sop.get(ref_inst)
    if row is None:
        return {
            "status": STATUS_MISSING,
            "reason": "引用影像未由所选成功批次交付（系统中无对应实例）",
            "image_sop_uid": ref_inst,
            "sop_class_uid": ref_class,
        }
    if row["sop_uid"] not in delivery.sop_uids:
        # 理论上 JOIN 已保证，防御性保留：失败批次/未勾选批次遗留对象不算证据
        return {
            "status": STATUS_MISSING,
            "reason": "引用影像不属于所选的成功批次",
            "image_sop_uid": ref_inst,
            "sop_class_uid": ref_class,
        }
    if not (delivery.objects_dir / row["object_path"]).exists():
        return {
            "status": STATUS_MISSING,
            "reason": "引用影像已登记但对象文件缺失",
            "image_sop_uid": ref_inst,
            "sop_class_uid": ref_class,
        }
    delivered_class = row["sop_class_uid"] or ""
    if ref_class and delivered_class and ref_class != delivered_class:
        return {
            "status": STATUS_CLASS_MISMATCH,
            "reason": (
                f"引用类 UID {ref_class} 与交付实例 SOP Class UID "
                f"{delivered_class} 不符"
            ),
            "image_sop_uid": ref_inst,
            "sop_class_uid": ref_class,
        }
    if not ref_class:
        return {
            "status": STATUS_MISSING,
            "reason": "IMAGE 引用缺少 ReferencedSOPClassUID，无法核验类相符性",
            "image_sop_uid": ref_inst,
            "sop_class_uid": "",
        }
    return {
        "status": STATUS_LINKED,
        "reason": "",
        "image_sop_uid": ref_inst,
        "sop_class_uid": ref_class,
    }


# ---------------------------------------------------------------- 内容树遍历
def _segment(index: int, item: pydicom.Dataset) -> str:
    value_type = _text(getattr(item, "ValueType", "")) or "?"
    scheme, code, _ = _code_triple(item, "ConceptNameCodeSequence")
    concept = f"{scheme}/{code}" if scheme or code else "-"
    return f"{index}:{value_type}[{concept}]"


def _required_num_fields(
    item: pydicom.Dataset, sr_uid: str, path: str
) -> tuple[str, tuple[str, str, str], tuple[str, str, str]]:
    """校验并提取 NUM 必需字段；不全即抛 ExportError（整次拒绝）。"""
    scheme, code, meaning = _code_triple(item, "ConceptNameCodeSequence")
    if not scheme or not code:
        raise ExportError(
            f"SR {sr_uid} 路径 {path} 的 NUM 节点缺少测量概念编码 "
            "(ConceptNameCodeSequence.CodeValue/CodingSchemeDesignator)"
        )
    mvs = getattr(item, "MeasuredValueSequence", None)
    if not mvs:
        raise ExportError(
            f"SR {sr_uid} 路径 {path} 的 NUM 节点缺少 MeasuredValueSequence"
        )
    measured = mvs[0]
    numeric = _text(getattr(measured, "NumericValue", ""))
    if numeric == "":
        raise ExportError(
            f"SR {sr_uid} 路径 {path} 的 NUM 节点缺少 NumericValue"
        )
    unit_scheme, unit_code, unit_meaning = _code_triple(
        measured, "MeasurementUnitsCodeSequence"
    )
    if not unit_scheme or not unit_code:
        raise ExportError(
            f"SR {sr_uid} 路径 {path} 的 NUM 节点缺少计量单位编码 "
            "(MeasurementUnitsCodeSequence.CodeValue/CodingSchemeDesignator)"
        )
    return numeric, (scheme, code, meaning), (unit_scheme, unit_code, unit_meaning)


def _walk_content(
    seq,
    delivery: _Delivery,
    sr_uid: str,
    patient_id: str,
    path_prefix: list[str],
    container_stack: list[tuple[str, str, str]],
    rows: list[dict[str, Any]],
) -> None:
    for index, item in enumerate(seq, start=1):
        value_type = _text(getattr(item, "ValueType", ""))
        segment = _segment(index, item)
        path = ".".join(path_prefix + [segment])

        if value_type == "NUM":
            numeric, concept, unit = _required_num_fields(item, sr_uid, path)
            # 最近的 CONTAINER 祖先（NUM 自身不是 CONTAINER）
            container = container_stack[-1] if container_stack else ("", "", "")

            # 仅同一父序列的直接子节点中的 IMAGE 才算该 NUM 的候选证据
            sibling_images = [
                sib for sib in seq
                if _text(getattr(sib, "ValueType", "")) == "IMAGE"
            ]
            image_count = len(sibling_images)
            if image_count == 0:
                evidence = {
                    "status": STATUS_NO_EVIDENCE,
                    "reason": "同组直接子节点中没有 IMAGE 证据",
                    "image_sop_uid": "",
                    "sop_class_uid": "",
                }
            elif image_count > 1:
                evidence = {
                    "status": STATUS_MULTIPLE,
                    "reason": f"同组直接子节点中有 {image_count} 个 IMAGE，证据不唯一",
                    "image_sop_uid": "",
                    "sop_class_uid": "",
                }
            else:
                evidence = _evaluate_evidence(sibling_images[0], delivery)

            rows.append({
                "anonymized_patient_id": patient_id,
                "source_sr_sop_uid": sr_uid,
                "content_tree_path": path,
                "container_code_scheme": container[0],
                "container_code_value": container[1],
                "container_code_meaning": container[2],
                "concept_code_scheme": concept[0],
                "concept_code_value": concept[1],
                "concept_code_meaning": concept[2],
                "numeric_value": numeric,
                "unit_code_scheme": unit[0],
                "unit_code_value": unit[1],
                "unit_code_meaning": unit[2],
                "evidence_status": evidence["status"],
                "evidence_status_label": STATUS_LABELS[evidence["status"]],
                "evidence_reason": evidence["reason"],
                "evidence_image_count": image_count,
                "evidence_image_sop_uid": evidence["image_sop_uid"],
                "evidence_sop_class_uid": evidence["sop_class_uid"],
            })

        # 递归子树；CONTAINER 祖先栈仅压入 CONTAINER 节点
        child_seq = getattr(item, "ContentSequence", None)
        if child_seq:
            next_stack = container_stack
            if value_type == "CONTAINER":
                next_stack = container_stack + [_code_triple(item, "ConceptNameCodeSequence")]
            _walk_content(
                child_seq, delivery, sr_uid, patient_id,
                path_prefix + [segment], next_stack, rows,
            )


# ---------------------------------------------------------------- 对外入口
def export_measurements(batch_ids: list[str]) -> tuple[bytes, dict[str, int]]:
    """按所选成功批次导出 SR NUM 测量 CSV。

    返回 (utf-8-sig 编码的 CSV 字节, 统计信息)；任何硬性错误抛 ExportError。
    """
    conn = get_db()
    try:
        delivery = _load_delivery(conn, batch_ids)

        # 交付集合中的 SR（按新 SOP UID 去重排序；同原 SOP 跨批复用天然只出现一次）
        sr_rows = [
            row for row in delivery.by_new_sop.values()
            if _is_sr_class(row["sop_class_uid"] or "")
        ]
        sr_rows.sort(key=lambda r: r["new_sop_uid"])
        if not sr_rows:
            raise ExportError("所选成功批次中没有可导出的 SR 文档")

        rows: list[dict[str, Any]] = []
        seen_sr: set[str] = set()
        for sr_row in sr_rows:
            orig_sop = sr_row["sop_uid"]
            if orig_sop in seen_sr:  # 跨批复用的 SR 只计一次
                continue
            seen_sr.add(orig_sop)
            obj_path = delivery.objects_dir / sr_row["object_path"]
            if not obj_path.exists():
                raise ExportError(f"SR 对象文件缺失: {sr_row['new_sop_uid']}")
            try:
                ds = pydicom.dcmread(str(obj_path))
            except Exception as exc:  # noqa: BLE001
                raise ExportError(
                    f"SR 解析失败 {sr_row['new_sop_uid']}: {exc}"
                ) from exc

            patient_id = _text(getattr(ds, "PatientID", ""))
            top_seq = getattr(ds, "ContentSequence", None)
            if not top_seq:
                # 没有内容树就没有 NUM；不是错误，但记录为空遍历
                continue
            _walk_content(
                top_seq, delivery, sr_row["new_sop_uid"], patient_id,
                [], [], rows,
            )

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stats = {
            "sr_count": len(seen_sr),
            "measurement_count": len(rows),
            "linked_count": sum(
                1 for r in rows if r["evidence_status"] == STATUS_LINKED),
        }
        return buf.getvalue().encode("utf-8-sig"), stats
    finally:
        conn.close()


def list_batches_with_sr_count() -> list[sqlite3_row]:
    """批次列表附 SR 交付数（去重），供导出页勾选。"""
    conn = get_db()
    try:
        return conn.execute(
            """
            SELECT b.*,
                   COUNT(DISTINCT CASE
                       WHEN i.sop_class_uid LIKE ? THEN bf.sop_uid
                   END) AS sr_count
            FROM batches b
            LEFT JOIN batch_files bf ON bf.batch_id = b.id
            LEFT JOIN instances i ON i.sop_uid = bf.sop_uid
            GROUP BY b.id
            ORDER BY b.created_at DESC, b.id DESC
            """,
            (SR_CLASS_PREFIX + "%",),
        ).fetchall()
    finally:
        conn.close()
