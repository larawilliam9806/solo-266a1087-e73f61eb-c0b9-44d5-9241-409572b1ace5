"""SR 数值测量导出功能测试。

覆盖：
- NUM 展开、内容树路径、最近 CONTAINER/概念编码、数值与单位、匿名字段；
- IMAGE 证据五态：linked / missing / class_mismatch / no_evidence / multiple_evidence；
- 证据仅限所选成功批次；失败批次遗留对象不算证据；
- 跨批复用 SR 只计一次；补传影像后重查更新关联；
- SR 缺失/解析失败/NUM 必需字段不全 -> 整次拒绝；
- 导出不改变原批次下载。
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pydicom
from werkzeug.datastructures import MultiDict

from samples import generate_samples as gen


# ---------------------------------------------------------------- 辅助
def _upload(client, paths):
    data = MultiDict()
    handles = []
    for p in paths:
        fh = open(p, "rb")
        handles.append(fh)
        data.add("dicom_files", (fh, Path(p).name, "application/dicom"))
    resp = client.post("/upload", data=data, content_type="multipart/form-data",
                       follow_redirects=False)
    for fh in handles:
        fh.close()
    assert resp.status_code == 302
    return resp.headers["Location"].rstrip("/").split("/")[-1]


def _export_csv(client, batch_ids):
    resp = client.post("/export", data=MultiDict(
        [("batch_ids", b) for b in batch_ids]))
    return resp


def _rows(resp):
    assert resp.status_code == 200, resp.data[:200]
    text = resp.data.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _find(rows, code_value: str):
    matched = [r for r in rows if r["concept_code_value"] == code_value]
    assert len(matched) == 1
    return matched[0]


def _sr_ds(sample_files: Path) -> pydicom.Dataset:
    return pydicom.dcmread(str(sample_files / "sr_001.dcm"))


def _write_sr(ds: pydicom.Dataset, path: Path, new_sop: str | None = None) -> Path:
    if new_sop is not None:
        ds.SOPInstanceUID = new_sop
        ds.file_meta.MediaStorageSOPInstanceUID = new_sop
    ds.save_as(str(path), write_like_original=False)
    return path


def _set_image_refs(group_item, refs):
    """把测量组内的 IMAGE 节点替换为给定 (类UID, 实例UID) 引用列表。"""
    images = [it for it in group_item.ContentSequence
              if str(it.ValueType) == "IMAGE"]
    assert images
    image = images[0]
    seq = []
    for cls, inst in refs:
        ref = pydicom.Dataset()
        ref.ReferencedSOPClassUID = cls
        ref.ReferencedSOPInstanceUID = inst
        seq.append(ref)
    image.ReferencedSOPSequence = pydicom.Sequence(seq)


def _measurement_group(ds):
    root = ds.ContentSequence[0]
    groups = [it for it in root.ContentSequence
              if str(it.ValueType) == "CONTAINER"]
    assert len(groups) == 1
    return groups[0]


# ---------------------------------------------------------------- 页面
def test_export_page_lists_batches(client, sample_files):
    _upload(client, sorted(sample_files.glob("*.dcm")))
    resp = client.get("/export")
    assert resp.status_code == 200
    assert "SR 测量导出".encode() in resp.data
    assert b'batch_ids' in resp.data


def test_export_requires_selection(client, sample_files):
    _upload(client, sorted(sample_files.glob("*.dcm")))
    resp = client.post("/export", data=MultiDict([]), follow_redirects=True)
    assert "至少勾选".encode() in resp.data


def test_batch_without_sr_rejected(client, sample_files):
    bid = _upload(client, [sample_files / "ct_001.dcm"])
    resp = _export_csv(client, [bid])
    assert resp.status_code == 302
    followed = client.get(resp.headers["Location"])
    assert "没有可导出的 SR".encode() in followed.data


# ---------------------------------------------------------------- 基本展开
def test_full_export_structure_and_linked_evidence(client, sample_files):
    bid = _upload(client, sorted(sample_files.glob("*.dcm")))
    rows = _rows(_export_csv(client, [bid]))
    assert len(rows) == 2  # 样例 SR 含两个 NUM

    lesion = _find(rows, "G-A185")
    # 匿名 PatientID（P+12 位数字），原 ID 不得泄露
    assert lesion["anonymized_patient_id"].startswith("P")
    assert len(lesion["anonymized_patient_id"]) == 13
    assert gen.PATIENT_ID not in lesion.values()
    # 来源 SR 是新 SOP UID，原 SR UID 不得出现
    assert lesion["source_sr_sop_uid"] != gen.SOP_SR
    assert gen.SOP_SR not in str(rows)
    # 路径、最近 CONTAINER、概念、数值、单位
    assert "CONTAINER[DCM/112000]" in lesion["content_tree_path"]
    assert "NUM[SRT/G-A185]" in lesion["content_tree_path"]
    assert lesion["container_code_value"] == "125007"
    assert lesion["container_code_scheme"] == "DCM"
    assert lesion["concept_code_scheme"] == "SRT"
    assert lesion["concept_code_meaning"] == "Length of Target Lesion"
    assert lesion["numeric_value"] == "31.4"
    assert lesion["unit_code_value"] == "mm"
    assert lesion["unit_code_scheme"] == "UCUM"
    # 证据：唯一 IMAGE -> CT1，由本成功批次交付且类相符
    assert lesion["evidence_status"] == "linked"
    assert lesion["evidence_image_count"] == "1"
    assert lesion["evidence_sop_class_uid"] == str(pydicom.uid.CTImageStorage)
    # 证据 UID 必须是换号后的新 CT1 UID（原 UID 不得泄露）
    assert gen.SOP_CT_1 not in lesion["evidence_image_sop_uid"]
    assert lesion["evidence_image_sop_uid"] != ""

    volume = _find(rows, "118565006")
    assert volume["evidence_status"] == "no_evidence"
    assert volume["evidence_image_count"] == "0"
    assert volume["container_code_value"] == "112000"  # 最近 CONTAINER 是根
    assert volume["numeric_value"] == "2.75"
    assert volume["unit_code_value"] == "cm3"


# ---------------------------------------------------------------- 缺件 + 补传更新
def test_missing_evidence_then_resolved_after_supplementary_upload(client, sample_files):
    # 批次1：仅 SR，引用的 CT1 未交付 -> missing
    sr_only = _upload(client, [sample_files / "sr_001.dcm"])
    rows = _rows(_export_csv(client, [sr_only]))
    lesion = _find(rows, "G-A185")
    assert lesion["evidence_status"] == "missing"
    assert "未由所选成功批次交付" in lesion["evidence_reason"]

    # 批次2：补传 CT1 成功
    ct_batch = _upload(client, [sample_files / "ct_001.dcm"])

    # 仍只选批次1：证据集合不含 CT1，依旧 missing
    rows = _rows(_export_csv(client, [sr_only]))
    assert _find(rows, "G-A185")["evidence_status"] == "missing"

    # 同时选两个成功批次重查：关联更新为 linked
    rows = _rows(_export_csv(client, [sr_only, ct_batch]))
    lesion = _find(rows, "G-A185")
    assert lesion["evidence_status"] == "linked"

    # 原批次下载照常可用
    assert client.get(f"/batches/{sr_only}/download").status_code == 200
    assert client.get(f"/batches/{ct_batch}/download").status_code == 200


# ---------------------------------------------------------------- 类不符
def test_class_mismatch_marked(client, sample_files, tmp_path):
    ds = _sr_ds(sample_files)
    group = _measurement_group(ds)
    # 引用 CT1 的实例 UID，但类 UID 故意写成 MR Image Storage
    _set_image_refs(group, [
        (str(pydicom.uid.MRImageStorage), gen.SOP_CT_1),
    ])
    sr_path = _write_sr(ds, tmp_path / "sr_mismatch.dcm",
                        new_sop=gen.SOP_SR + ".99")
    bid = _upload(client, [sr_path, sample_files / "ct_001.dcm"])
    rows = _rows(_export_csv(client, [bid]))
    lesion = next(r for r in rows if r["concept_code_value"] == "G-A185")
    assert lesion["evidence_status"] == "class_mismatch"
    assert "不符" in lesion["evidence_reason"]


# ---------------------------------------------------------------- 多个证据
def test_multiple_evidence_marked(client, sample_files, tmp_path):
    ds = _sr_ds(sample_files)
    group = _measurement_group(ds)
    _set_image_refs(group, [
        (str(pydicom.uid.CTImageStorage), gen.SOP_CT_1),
        (str(pydicom.uid.CTImageStorage), gen.SOP_CT_2),
    ])
    sr_path = _write_sr(ds, tmp_path / "sr_multi.dcm",
                        new_sop=gen.SOP_SR + ".98")
    # 两张 CT 都交付，仍应判 multiple_evidence
    bid = _upload(client, [
        sr_path,
        sample_files / "ct_001.dcm",
        sample_files / "ct_002.dcm",
    ])
    rows = _rows(_export_csv(client, [bid]))
    lesion = next(r for r in rows if r["concept_code_value"] == "G-A185")
    assert lesion["evidence_status"] == "multiple_evidence"
    assert lesion["evidence_image_sop_uid"] == ""


def test_two_image_siblings_marked_multiple(client, sample_files, tmp_path):
    """同组存在两个 IMAGE 兄弟节点（即使引用同一 SOP）也判多个证据。"""
    ds = _sr_ds(sample_files)
    group = _measurement_group(ds)
    extra = pydicom.Dataset()
    extra.ValueType = "IMAGE"
    extra.ConceptNameCodeSequence = [pydicom.Dataset()]
    extra.ConceptNameCodeSequence[0].CodeValue = "121112"
    extra.ConceptNameCodeSequence[0].CodingSchemeDesignator = "DCM"
    extra.ConceptNameCodeSequence[0].CodeMeaning = "Measurement image"
    ref = pydicom.Dataset()
    ref.ReferencedSOPClassUID = str(pydicom.uid.CTImageStorage)
    ref.ReferencedSOPInstanceUID = gen.SOP_CT_1
    extra.ReferencedSOPSequence = [ref]
    group.ContentSequence.append(extra)
    sr_path = _write_sr(ds, tmp_path / "sr_two_img.dcm",
                        new_sop=gen.SOP_SR + ".97")
    bid = _upload(client, [sr_path, sample_files / "ct_001.dcm"])
    rows = _rows(_export_csv(client, [bid]))
    lesion = next(r for r in rows if r["concept_code_value"] == "G-A185")
    assert lesion["evidence_status"] == "multiple_evidence"
    assert lesion["evidence_image_count"] == "2"


# ---------------------------------------------------------------- 跨批复用只计一次
def test_sr_reused_across_batches_counted_once(client, sample_files):
    b1 = _upload(client, sorted(sample_files.glob("*.dcm")))
    b2 = _upload(client, [sample_files / "sr_001.dcm", sample_files / "ct_002.dcm"])
    rows = _rows(_export_csv(client, [b1, b2]))
    # 同一 SR 只展开一次 -> 仍然只有 2 行测量
    assert len(rows) == 2
    sr_uids = {r["source_sr_sop_uid"] for r in rows}
    assert len(sr_uids) == 1


# ---------------------------------------------------------------- 失败批次不算证据
def test_failed_batch_objects_not_counted(client, sample_files, tmp_path):
    # 先制造一个成功的 CT1 批次
    good = _upload(client, [sample_files / "ct_001.dcm"])

    # 再制造一个失败批次：同 SOP 不同字节（整批拒绝），其中顺带带 SR
    evil_dir = tmp_path / "evil"
    evil_dir.mkdir()
    ds = pydicom.dcmread(str(sample_files / "ct_001.dcm"))
    ds.PixelData = bytes([0xCC]) * len(ds.PixelData)
    evil_ct = evil_dir / "ct_evil.dcm"
    ds.save_as(str(evil_ct), write_like_original=False)
    failed_resp = _upload(client, [evil_ct, sample_files / "sr_001.dcm"])
    from app import services
    assert services.get_batch(failed_resp)["status"] == "failed"

    # 只选失败批次：SR 未由成功批次交付，页面上无法勾选；直接调接口应被拒绝
    resp = _export_csv(client, [failed_resp])
    assert resp.status_code == 302
    followed = client.get(resp.headers["Location"])
    assert "只能选择处理成功".encode() in followed.data

    # 选成功 CT 批次 + 失败批次：失败批次被整体拒绝（不静默忽略）
    resp = _export_csv(client, [good, failed_resp])
    assert resp.status_code == 302

    # 成功批次里没有 SR -> 单独选它会提示无 SR
    resp = _export_csv(client, [good])
    followed = client.get(resp.headers["Location"])
    assert "没有可导出的 SR".encode() in followed.data


# ---------------------------------------------------------------- 整次拒绝
def test_export_rejected_when_num_required_field_incomplete(client, sample_files, tmp_path):
    ds = _sr_ds(sample_files)
    group = _measurement_group(ds)
    num = next(it for it in group.ContentSequence if str(it.ValueType) == "NUM")
    # 删掉 NumericValue -> NUM 必需字段不全
    del num.MeasuredValueSequence[0].NumericValue
    sr_path = _write_sr(ds, tmp_path / "sr_bad_num.dcm",
                        new_sop=gen.SOP_SR + ".96")
    bid = _upload(client, [sr_path])
    resp = _export_csv(client, [bid])
    assert resp.status_code == 302
    followed = client.get(resp.headers["Location"])
    assert "导出被拒绝".encode() in followed.data
    assert "NumericValue".encode() in followed.data


def test_export_rejected_when_sr_object_missing(client, sample_files):
    bid = _upload(client, [sample_files / "sr_001.dcm"])
    from app import config
    from app.db import get_db
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT object_path FROM instances WHERE sop_uid=?", (gen.SOP_SR,)
        ).fetchone()
    finally:
        conn.close()
    obj = config.get_objects_dir() / row["object_path"]
    obj.unlink()
    resp = _export_csv(client, [bid])
    followed = client.get(resp.headers["Location"])
    assert "SR 对象文件缺失".encode() in followed.data


def test_unknown_batch_id_rejected(client, sample_files):
    _upload(client, [sample_files / "sr_001.dcm"])
    resp = _export_csv(client, ["nonexistent-batch-id"])
    followed = client.get(resp.headers["Location"])
    assert "批次不存在".encode() in followed.data


# ---------------------------------------------------------------- 原标识不泄露
def test_export_contains_no_original_identifiers(client, sample_files):
    bid = _upload(client, sorted(sample_files.glob("*.dcm")))
    resp = _export_csv(client, [bid])
    blob = resp.data.decode("utf-8-sig")
    for secret in (gen.PATIENT_ID, gen.SOP_SR, gen.SOP_CT_1, gen.SOP_CT_2,
                   gen.STUDY_UID):
        assert secret not in blob, f"原标识泄露: {secret}"
