"""生成含跨文件引用的合成 DICOM 样例（未压缩 Explicit VR Little Endian）。

产物（输出到 samples/data/）：
  ct_001.dcm, ct_002.dcm   —— 两张 CT 图像，共享同一 Study/Series，带像素数据
  rs_001.dcm               —— RT Structure Set，ReferencedFrameOfReferenceSequence
                              中引用两张 CT 的 SOPInstanceUID
  sr_001.dcm               —— SR 文档，ReferencedStudySequence/ReferencedSeriesSequence/
                              CurrentRequestedProcedureEvidenceSequence 嵌套引用
                              Study/Series/SOP 三级 UID

运行：python samples/generate_samples.py [输出目录]
脚本与输出目录分离；输出目录会被清空重建。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    RTStructureSetStorage,
    EnhancedSRStorage,
    generate_uid,
)

PATIENT_ID = "PID-2026-0001"
PATIENT_NAME = "ZHANG^SAN"
STUDY_UID = "1.2.826.0.1.9999.7777.1001"
SERIES_CT_UID = "1.2.826.0.1.9999.7777.2001"
FRAME_OF_REF_UID = "1.2.826.0.1.9999.7777.3001"
SOP_CT_1 = "1.2.826.0.1.9999.7777.4001"
SOP_CT_2 = "1.2.826.0.1.9999.7777.4002"
SOP_RT = "1.2.826.0.1.9999.7777.4003"
SOP_SR = "1.2.826.0.1.9999.7777.4004"
SERIES_RT_UID = "1.2.826.0.1.9999.7777.2002"
SERIES_SR_UID = "1.2.826.0.1.9999.7777.2003"


def _file_meta(sop_class: str, sop_uid: str) -> FileMetaDataset:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    meta.ImplementationVersionName = "SYNTHETIC-SAMPLE"
    return meta


def _patient_study(ds: Dataset) -> None:
    ds.PatientName = PATIENT_NAME
    ds.PatientID = PATIENT_ID
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "O"
    ds.StudyInstanceUID = STUDY_UID
    ds.StudyDate = "20261118"
    ds.StudyTime = "090000"
    ds.AccessionNumber = "ACC-0001"
    ds.ReferringPhysicianName = "LI^YI"
    ds.Modality = "CT"
    ds.StudyDescription = "Synthetic cross-reference study"


def _write(ds: FileDataset, path: Path) -> None:
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path), write_like_original=False)


def make_ct(path: Path, sop_uid: str, instance_number: int, pixel_seed: int) -> None:
    ds = FileDataset(str(path), Dataset(), file_meta=_file_meta(CTImageStorage, sop_uid),
                     preamble=b"\0" * 128)
    _patient_study(ds)
    ds.Modality = "CT"
    ds.SeriesInstanceUID = SERIES_CT_UID
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.SeriesNumber = 1
    ds.InstanceNumber = instance_number
    ds.FrameOfReferenceUID = FRAME_OF_REF_UID
    ds.ImagePositionPatient = [0.0, 0.0, float(instance_number) * 5.0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceThickness = 5.0
    ds.PixelSpacing = [0.7, 0.7]
    rows, cols = 64, 64
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    rng = np.random.default_rng(pixel_seed)
    pixels = (rng.integers(0, 2000, size=(rows, cols), dtype=np.uint16)
              + np.tile(np.arange(cols, dtype=np.uint16), (rows, 1)))
    ds.PixelData = pixels.astype("<u2").tobytes()
    _write(ds, path)


def _rt_referenced_study() -> Dataset:
    # RT Structure Set 的三层嵌套引用：Study -> Series -> 每一张 CT SOP
    ref_study = Dataset()
    ref_study.StudyInstanceUID = STUDY_UID

    ref_series = Dataset()
    ref_series.SeriesInstanceUID = SERIES_CT_UID
    contour_seq = []
    for sop_uid in (SOP_CT_1, SOP_CT_2):
        ref_image = Dataset()
        ref_image.ReferencedSOPClassUID = CTImageStorage
        ref_image.ReferencedSOPInstanceUID = sop_uid  # 跨文件 SOP 引用
        ref_image.ReferencedFrameNumber = 1
        contour_seq.append(ref_image)
    ref_series.ContourImageSequence = contour_seq
    ref_study.RTReferencedSeriesSequence = [ref_series]
    return ref_study


def make_rtstruct(path: Path) -> None:
    ds = FileDataset(str(path), Dataset(),
                     file_meta=_file_meta(RTStructureSetStorage, SOP_RT),
                     preamble=b"\0" * 128)
    _patient_study(ds)
    ds.Modality = "RTSTRUCT"
    ds.SeriesInstanceUID = SERIES_RT_UID
    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = SOP_RT
    ds.SeriesNumber = 2
    ds.InstanceNumber = 1
    ds.FrameOfReferenceUID = FRAME_OF_REF_UID

    frame_ref = Dataset()
    frame_ref.FrameOfReferenceUID = FRAME_OF_REF_UID
    frame_ref.RTReferencedStudySequence = [_rt_referenced_study()]
    ds.ReferencedFrameOfReferenceSequence = [frame_ref]

    structure_set_roi = Dataset()
    structure_set_roi.ROINumber = 1
    structure_set_roi.ROIName = "SyntheticTarget"
    structure_set_roi.ROIGenerationAlgorithm = "MANUAL"
    structure_set_roi.ReferencedFrameOfReferenceUID = FRAME_OF_REF_UID
    ds.StructureSetROISequence = [structure_set_roi]
    ds.StructureSetLabel = "SYNTH"
    _write(ds, path)


def make_sr(path: Path) -> None:
    ds = FileDataset(str(path), Dataset(),
                     file_meta=_file_meta(EnhancedSRStorage, SOP_SR),
                     preamble=b"\0" * 128)
    _patient_study(ds)
    ds.Modality = "SR"
    ds.SeriesInstanceUID = SERIES_SR_UID
    ds.SOPClassUID = EnhancedSRStorage
    ds.SOPInstanceUID = SOP_SR
    ds.SeriesNumber = 3
    ds.InstanceNumber = 1

    # 1) ReferencedStudySequence：注意该序列里的 ReferencedSOPInstanceUID 按 DICOM
    #    约定承载的是 StudyInstanceUID —— 用于验证“身份优先于标签提示”的映射规则
    ref_study = Dataset()
    ref_study.ReferencedSOPClassUID = "1.2.840.10008.3.1.2.3.1"  # Study Root 类 UID（不换号）
    ref_study.ReferencedSOPInstanceUID = STUDY_UID
    ds.ReferencedStudySequence = [ref_study]

    # 2) ReferencedSeriesSequence -> 嵌套引用 CT SOP
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = SERIES_CT_UID
    ref_image = Dataset()
    ref_image.ReferencedSOPClassUID = CTImageStorage  # 类 UID 不换号
    ref_image.ReferencedSOPInstanceUID = SOP_CT_1     # 跨文件 SOP 引用
    ref_series.ReferencedImageSequence = [ref_image]
    ds.ReferencedSeriesSequence = [ref_series]

    # 3) CurrentRequestedProcedureEvidenceSequence：Study -> Series -> SOP 三级
    evidence_study = Dataset()
    evidence_study.StudyInstanceUID = STUDY_UID
    evidence_series = Dataset()
    evidence_series.SeriesInstanceUID = SERIES_CT_UID
    ev_refs = []
    for sop_uid in (SOP_CT_1, SOP_CT_2):
        item = Dataset()
        item.ReferencedSOPClassUID = CTImageStorage
        item.ReferencedSOPInstanceUID = sop_uid
        ev_refs.append(item)
    evidence_series.ReferencedSOPSequence = ev_refs
    evidence_study.ReferencedSeriesSequence = [evidence_series]
    ds.CurrentRequestedProcedureEvidenceSequence = [evidence_study]

    # 最简 SR 内容树
    root = Dataset()
    root.ValueType = "CONTAINER"
    root.ConceptNameCodeSequence = [Dataset()]
    root.ConceptNameCodeSequence[0].CodeValue = "112000"
    root.ConceptNameCodeSequence[0].CodingSchemeDesignator = "DCM"
    root.ConceptNameCodeSequence[0].CodeMeaning = "Imaging Measurement Report"
    root.ContinuityOfContent = "CONTINUOUS"
    text_item = Dataset()
    text_item.ValueType = "TEXT"
    text_item.ConceptNameCodeSequence = [Dataset()]
    text_item.ConceptNameCodeSequence[0].CodeValue = "112039"
    text_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = "DCM"
    text_item.ConceptNameCodeSequence[0].CodeMeaning = "Recommended Follow-up"
    text_item.TextValue = "Synthetic report referencing the CT series."

    # 测量组 CONTAINER：内含 NUM + 同组唯一 IMAGE 证据（引用 CT1）
    group = Dataset()
    group.ValueType = "CONTAINER"
    group.ConceptNameCodeSequence = [Dataset()]
    group.ConceptNameCodeSequence[0].CodeValue = "125007"
    group.ConceptNameCodeSequence[0].CodingSchemeDesignator = "DCM"
    group.ConceptNameCodeSequence[0].CodeMeaning = "Measurement Group"

    num_lesion = Dataset()
    num_lesion.ValueType = "NUM"
    num_lesion.ConceptNameCodeSequence = [Dataset()]
    num_lesion.ConceptNameCodeSequence[0].CodeValue = "G-A185"
    num_lesion.ConceptNameCodeSequence[0].CodingSchemeDesignator = "SRT"
    num_lesion.ConceptNameCodeSequence[0].CodeMeaning = "Length of Target Lesion"
    measured = Dataset()
    measured.NumericValue = "31.4"
    unit_code = Dataset()
    unit_code.CodeValue = "mm"
    unit_code.CodingSchemeDesignator = "UCUM"
    unit_code.CodeMeaning = "millimeter"
    measured.MeasurementUnitsCodeSequence = [unit_code]
    num_lesion.MeasuredValueSequence = [measured]

    image_ref = Dataset()
    image_ref.ValueType = "IMAGE"
    image_ref.ConceptNameCodeSequence = [Dataset()]
    image_ref.ConceptNameCodeSequence[0].CodeValue = "121112"
    image_ref.ConceptNameCodeSequence[0].CodingSchemeDesignator = "DCM"
    image_ref.ConceptNameCodeSequence[0].CodeMeaning = "Measurement image"
    ref_sop = Dataset()
    ref_sop.ReferencedSOPClassUID = CTImageStorage
    ref_sop.ReferencedSOPInstanceUID = SOP_CT_1
    image_ref.ReferencedSOPSequence = [ref_sop]

    group.ContentSequence = [num_lesion, image_ref]

    # 第二个 NUM：同组无 IMAGE，用于演示 no_evidence 状态
    num_followup = Dataset()
    num_followup.ValueType = "NUM"
    num_followup.ConceptNameCodeSequence = [Dataset()]
    num_followup.ConceptNameCodeSequence[0].CodeValue = "118565006"
    num_followup.ConceptNameCodeSequence[0].CodingSchemeDesignator = "SCT"
    num_followup.ConceptNameCodeSequence[0].CodeMeaning = "Volume"
    measured2 = Dataset()
    measured2.NumericValue = "2.75"
    unit_code2 = Dataset()
    unit_code2.CodeValue = "cm3"
    unit_code2.CodingSchemeDesignator = "UCUM"
    unit_code2.CodeMeaning = "cubic centimeter"
    measured2.MeasurementUnitsCodeSequence = [unit_code2]
    num_followup.MeasuredValueSequence = [measured2]

    root.ContentSequence = [text_item, group, num_followup]
    ds.ContentSequence = [root]
    _write(ds, path)


def main(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    make_ct(out_dir / "ct_001.dcm", SOP_CT_1, 1, pixel_seed=11)
    make_ct(out_dir / "ct_002.dcm", SOP_CT_2, 2, pixel_seed=22)
    make_rtstruct(out_dir / "rs_001.dcm")
    make_sr(out_dir / "sr_001.dcm")
    print(f"已生成 4 个合成 DICOM 样例（含跨文件引用）到 {out_dir}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    main(target)
