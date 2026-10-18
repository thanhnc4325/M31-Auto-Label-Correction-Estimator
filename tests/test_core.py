"""Kiểm tra logic lõi (không cần Streamlit). Chạy: python -m unittest discover tests"""
import unittest

import numpy as np
import pandas as pd

from m31.calibration import (
    MIN_NONZERO_FOR_FIT,
    MIN_ROWS_FOR_FIT,
    BaselineTimeEstimator,
    evaluate,
    priority_vs_correction,
)
from m31.estimator import estimate
from m31.schema import validate_correction_log, validate_prelabels


def _pre(n=3, task="detection"):
    return pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)], "task": [task] * n, "model": ["m"] * n,
        "mean_confidence": np.linspace(0.3, 0.9, n), "num_predictions": [10] * n,
        "low_confidence_count": list(range(n)), "image_width": [1280] * n, "image_height": [720] * n,
    })


class TestSchema(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(validate_prelabels(_pre()).ok)

    def test_missing_column(self):
        res = validate_prelabels(_pre().drop(columns=["num_predictions"]))
        self.assertFalse(res.ok)

    def test_ground_truth_columns_are_dropped(self):
        res = validate_prelabels(_pre().assign(gt_boxes=1, map_50=0.5))
        self.assertTrue(res.ok)
        self.assertNotIn("gt_boxes", res.df.columns)
        self.assertNotIn("map_50", res.df.columns)
        self.assertTrue(any("ground truth" in w for w in res.warnings))

    def test_task_alias_and_bad_task(self):
        self.assertEqual(validate_prelabels(_pre(task="SEG")).df["task"].iloc[0], "segmentation")
        self.assertFalse(validate_prelabels(_pre(task="classification")).ok)

    def test_low_conf_exceeds_predictions(self):
        df = _pre()
        df.loc[0, "low_confidence_count"] = 99
        self.assertFalse(validate_prelabels(df).ok)

    def test_zero_predictions_allow_empty_confidence(self):
        df = _pre()
        df.loc[0, ["num_predictions", "low_confidence_count", "mean_confidence"]] = [0, 0, np.nan]
        self.assertTrue(validate_prelabels(df).ok)


class TestEstimate(unittest.TestCase):
    def test_outputs(self):
        out = estimate(validate_prelabels(_pre()).df, BaselineTimeEstimator())
        for c in ["predicted_quality_score", "estimated_correction_time_sec", "priority", "reason"]:
            self.assertIn(c, out.columns)
        self.assertTrue(out["predicted_quality_score"].between(0, 100).all())
        self.assertTrue(set(out["priority"]) <= {"thấp", "vừa", "cao"})
        # độ tin cậy thấp hơn + nhiều dự đoán kém hơn -> điểm thấp hơn
        self.assertLess(out["predicted_quality_score"].iloc[0], out["predicted_quality_score"].iloc[-1])

    def test_zero_predictions_high_priority(self):
        df = _pre()
        df.loc[0, ["num_predictions", "low_confidence_count", "mean_confidence"]] = [0, 0, np.nan]
        out = estimate(validate_prelabels(df).df, BaselineTimeEstimator())
        self.assertEqual(out["priority"].iloc[0], "cao")


class TestCalibration(unittest.TestCase):
    def _log(self, pre, fn):
        return pd.DataFrame({
            "image_id": pre["image_id"], "task": pre["task"], "model": pre["model"],
            "correction_time_sec": fn(pre), "edits_added": 0,
            "edits_removed": pre["low_confidence_count"], "edits_changed": 0,
        })

    def test_too_few_rows_no_mae(self):
        pre = validate_prelabels(_pre(3)).df
        rep = evaluate(pre, validate_correction_log(self._log(pre, lambda p: p["num_predictions"] * 1.0)).df)
        self.assertEqual(rep.baseline_mae, {})
        self.assertIsNone(rep.estimator)

    def test_calibration_recovers_linear_relation(self):
        n = MIN_ROWS_FOR_FIT + 10
        rng = np.random.default_rng(0)
        pre = _pre(n)
        pre["num_predictions"] = rng.integers(1, 40, n)
        pre["low_confidence_count"] = (pre["num_predictions"] * rng.uniform(0, 1, n)).astype(int)
        pre = validate_prelabels(pre).df
        log = validate_correction_log(
            self._log(pre, lambda p: 5 + 1.0 * p["num_predictions"] + 4.0 * p["low_confidence_count"])
        ).df
        rep = evaluate(pre, log)
        self.assertIsNotNone(rep.estimator)
        self.assertLess(rep.calibrated_cv_mae["detection"], rep.baseline_mae["detection"])

    def test_unmatched_rows_reported(self):
        pre = validate_prelabels(_pre(6)).df
        log = self._log(pre, lambda p: p["num_predictions"] * 1.0)
        log.loc[0, "image_id"] = "khong_ton_tai"
        rep = evaluate(pre, validate_correction_log(log).df)
        self.assertEqual(rep.n_unmatched_log, 1)



class TestLeakageGuards(unittest.TestCase):
    def test_post_correction_columns_dropped_from_prelabels(self):
        df = _pre().assign(human_correction_time_sec=5.0, action_delete=1, status="OK", notes="x")
        res = validate_prelabels(df)
        self.assertTrue(res.ok)
        for c in ["human_correction_time_sec", "action_delete", "status", "notes"]:
            self.assertNotIn(c, res.df.columns)
        self.assertTrue(any("sau khi con người sửa" in w for w in res.warnings))

    def test_optional_feature_columns_kept(self):
        df = _pre().assign(min_confidence=0.2, small_box_count=1, confidence_source="yolov8n")
        res = validate_prelabels(df)
        self.assertTrue(res.ok)
        self.assertIn("small_box_count", res.df.columns)
        self.assertIn("confidence_source", res.df.columns)

    def test_estimates_do_not_depend_on_correction_log(self):
        pre = validate_prelabels(_pre(6)).df
        a = estimate(pre, BaselineTimeEstimator())
        # Ghép log vào rồi dự đoán lại: kết quả phải y hệt (log không phải đầu vào).
        log = pre[["image_id", "task", "model"]].assign(
            correction_time_sec=999.0, edits_added=9, edits_removed=9, edits_changed=9)
        b = estimate(pre.merge(log, on=["image_id", "task", "model"]), BaselineTimeEstimator())
        for c in ["predicted_quality_score", "estimated_correction_time_sec", "priority"]:
            pd.testing.assert_series_equal(a[c], b[c])


class TestRealDataGuards(unittest.TestCase):
    def _sparse(self, n_nonzero):
        n = MIN_ROWS_FOR_FIT + 20
        pre = _pre(n)
        pre = validate_prelabels(pre).df
        t = np.zeros(n)
        t[:n_nonzero] = 8.0
        log = pre[["image_id", "task", "model"]].assign(
            correction_time_sec=t, edits_added=0, edits_removed=(t > 0).astype(int), edits_changed=0)
        return pre, validate_correction_log(log).df

    def test_calibrated_model_disabled_when_few_nonzero(self):
        pre, log = self._sparse(MIN_NONZERO_FOR_FIT - 1)
        rep = evaluate(pre, log)
        self.assertIsNone(rep.estimator)
        self.assertIn("detection", rep.constant_cv_mae)  # vẫn báo MAE để tham khảo

    def test_priority_vs_correction_counts(self):
        pre, log = self._sparse(5)
        est = estimate(pre, BaselineTimeEstimator())
        pv = priority_vs_correction(est, log)
        self.assertEqual(int(pv["so_anh"].sum()), len(pre))
        self.assertEqual(int(pv["so_anh_can_sua"].sum()), 5)


class TestPrepareRealData(unittest.TestCase):
    """Chạy scripts/prepare_real_data.build trên một ZIP nhỏ tự tạo (không cần dữ liệu thật)."""

    def test_build_from_tiny_zip(self):
        import io
        import sys
        import tempfile
        import zipfile
        from pathlib import Path

        from PIL import Image

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import prepare_real_data as prd

        def png(arr):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()

        mask = np.zeros((20, 40), dtype=np.uint8)
        mask[0:10, 0:10] = 255  # 100 px trong box 10x10 -> fill = 1.0
        def csv(*lines):
            return "\n".join(lines) + "\n"

        log_header = ("image_id,task_type,model_name,num_predictions,inference_time_sec,"
                      "human_correction_time_sec,action_add,action_delete,action_modify,status,notes")
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "pilot.zip"
            with zipfile.ZipFile(zp, "w") as z:
                for i in ["a", "b"]:
                    z.writestr(f"yolo_overlays/{i}_overlay.png", png(np.zeros((20, 40, 3), dtype=np.uint8)))
                z.writestr("yolo_predictions_100.csv", csv(
                    "image_id,class_name,confidence,x_min,y_min,x_max,y_max",
                    "a,car,0.4,0,0,10,10"))
                z.writestr("sam_predictions_100.csv", csv(
                    "image_id,task,model,class_name,prompt_source,x1,y1,x2,y2,confidence,mask_path",
                    "a,seg,sam_b,car,yolo_box,0,0,10,10,0.4,sam_masks/a_box_00.png"))
                z.writestr("sam_masks/a_box_00.png", png(mask))
                z.writestr("correction_log_yolo_100.csv", csv(
                    log_header,
                    "a,detection,yolov8n,1,0.1,7.5,0,1,0,NEED_CORRECTION,x",
                    "b,detection,yolov8n,0,0.1,0,0,0,0,OK,"))
                z.writestr("correction_log_sam_100.csv", csv(
                    log_header,
                    "a,segmentation,sam_b,1,0.1,9.0,0,0,1,NEED_CORRECTION,y",
                    "b,segmentation,sam_b,0,0.1,0,0,0,0,OK,"))
            pre, log, manifest = prd.build(zp)

        self.assertEqual(len(pre), 4)
        self.assertEqual(manifest["num_predictions_mismatch_vs_log"], 0)
        leaked = {"correction_time_sec", "human_correction_time_sec", "status", "notes", "edits_removed"}
        self.assertFalse(leaked & set(pre.columns))
        a_det = pre[(pre.image_id == "a") & (pre.task == "detection")].iloc[0]
        self.assertEqual(a_det["low_confidence_count"], 1)
        self.assertEqual(a_det["image_width"], 40)
        a_seg = pre[(pre.image_id == "a") & (pre.task == "segmentation")].iloc[0]
        self.assertAlmostEqual(a_seg["mean_mask_fill_ratio"], 1.0)
        b_det = pre[(pre.image_id == "b") & (pre.task == "detection")].iloc[0]
        self.assertEqual(b_det["num_predictions"], 0)
        self.assertTrue(validate_prelabels(pre).ok)
        self.assertTrue(validate_correction_log(log).ok)
        self.assertEqual(log["correction_time_sec"].tolist(), [7.5, 0.0, 9.0, 0.0])


class TestFeatureScripts(unittest.TestCase):
    """scripts/build_features.py và scripts/analyze_features.py."""

    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

    def test_build_features_without_logs_or_ground_truth(self):
        """ZIP chỉ có dự đoán + mask: nếu script đọc log/GT, test sẽ lỗi."""
        import io
        import tempfile
        import zipfile
        from pathlib import Path

        from PIL import Image

        import build_features as bf

        def png(arr):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            return buf.getvalue()

        m0 = np.zeros((20, 40), dtype=np.uint8); m0[0:10, 0:10] = 255           # đúng box 0
        m1 = np.zeros((20, 40), dtype=np.uint8); m1[5:10, 5:20] = 255           # 25 px trong box 1, 50 px ngoài
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "p.zip"
            with zipfile.ZipFile(zp, "w") as z:
                for i in ["a", "b"]:
                    z.writestr(f"yolo_overlays/{i}_overlay.png", png(np.zeros((20, 40, 3), dtype=np.uint8)))
                z.writestr("yolo_predictions_100.csv", "image_id,class_name,confidence,x_min,y_min,x_max,y_max\n"
                           "a,car,0.2,0,0,10,10\na,car,0.8,5,5,10,20\n")
                z.writestr("sam_predictions_100.csv",
                           "image_id,task,model,class_name,prompt_source,x1,y1,x2,y2,confidence,mask_path\n"
                           "a,s,sam_b,car,yolo_box,0,0,10,10,0.2,sam_masks/a0.png\n"
                           "a,s,sam_b,car,yolo_box,5,5,10,20,0.8,sam_masks/a1.png\n")
                z.writestr("sam_masks/a0.png", png(m0))
                z.writestr("sam_masks/a1.png", png(m1))
            df = bf.build(zp)

        self.assertEqual(len(df), 4)
        forbidden = {"correction_time_sec", "edits_added", "edits_removed", "edits_changed", "status", "notes"}
        self.assertFalse(forbidden & set(df.columns))
        det = df[(df.image_id == "a") & (df.task == "detection")].iloc[0]
        self.assertEqual(det["n_boxes"], 2)
        self.assertEqual(det["conf_n_lt030"], 1)
        self.assertEqual(det["conf_n_ge070"], 1)
        self.assertAlmostEqual(det["max_pair_iou"], 25 / 150, places=5)  # giao 5x5, hợp 100 + 75 - 25
        self.assertEqual(det["n_boxes_touch_bottom"], 1)        # y2 = 20 = H
        seg = df[(df.image_id == "a") & (df.task == "segmentation")].iloc[0]
        self.assertAlmostEqual(seg["mask_fill_in_box_mean"], (1.0 + 25 / 75) / 2, places=5)
        self.assertAlmostEqual(seg["mask_outside_box_frac_mean"], (0 + 50 / 75) / 2, places=5)
        self.assertAlmostEqual(seg["mask_overlap_frac"], 25 / 150, places=5)  # m0 ∩ m1 = 5x5; hợp = 100 + 75 - 25
        empty = df[(df.image_id == "b") & (df.task == "segmentation")].iloc[0]
        self.assertEqual(empty["n_boxes"], 0)
        self.assertTrue(np.isnan(empty["conf_mean"]))

    def test_stratified_folds_spread_corrections(self):
        import analyze_features as af
        pos = np.array([True] * 13 + [False] * 87)
        folds = af.stratified_folds(pos, 5, np.random.default_rng(0))
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(100)))
        self.assertEqual(sorted(int(pos[f].sum()) for f in folds), [2, 2, 3, 3, 3])

    def test_auc(self):
        import analyze_features as af
        pos = np.array([True, True, False, False])
        self.assertEqual(af.auc(np.array([4.0, 3.0, 2.0, 1.0]), pos), 1.0)
        self.assertEqual(af.auc(np.array([1.0, 2.0, 3.0, 4.0]), pos), 0.0)
        self.assertEqual(af.auc(np.ones(4), pos), 0.5)


class TestTrainedRidge(unittest.TestCase):
    """m31/ridge_model.py: fold theo image_id, không lọt cột log, artifact đọc lại được."""

    def _merged(self, n=40, n_pos=8):
        import m31.ridge_model as rm
        rng = np.random.default_rng(1)
        ids = [f"img{i:03d}" for i in range(n)]
        rows_pre, rows_log = [], []
        for task, model in [("detection", "yolo"), ("segmentation", "sam")]:
            for i, iid in enumerate(ids):
                k = int(rng.integers(0, 10))
                rows_pre.append(dict(image_id=iid, task=task, model=model, mean_confidence=float(rng.uniform(0.3, 0.9)),
                                     num_predictions=k, low_confidence_count=int(rng.integers(0, k + 1)),
                                     image_width=1280, image_height=720))
                t = 8.0 + k if i < n_pos else 0.0
                rows_log.append(dict(image_id=iid, task=task, model=model, correction_time_sec=t, edits_added=0,
                                     edits_removed=int(t > 0), edits_changed=0, status="x", notes="ghi chú"))
        pre = validate_prelabels(pd.DataFrame(rows_pre)).df
        log = validate_correction_log(pd.DataFrame(rows_log)).df
        return rm, rm.merge_for_training(pre, log)

    def test_merge_drops_post_correction_columns(self):
        _, m = self._merged()
        self.assertFalse({"edits_added", "edits_removed", "edits_changed", "status", "notes"} & set(m.columns))
        self.assertIn("correction_time_sec", m.columns)

    def test_group_folds_keep_image_together_and_stratify(self):
        rm, m = self._merged()
        folds = rm.group_folds(m, 5, seed=42)
        all_ids = np.concatenate(folds)
        self.assertEqual(len(all_ids), len(set(all_ids)))            # mỗi ảnh đúng 1 fold
        self.assertEqual(set(all_ids), set(m["image_id"]))
        pos = set(m.loc[m.correction_time_sec > 0, "image_id"])
        self.assertEqual(sorted(len(pos & set(f)) for f in folds), [1, 1, 2, 2, 2])
        # Seed cố định -> tái lập được
        self.assertEqual([f.tolist() for f in folds], [f.tolist() for f in rm.group_folds(m, 5, seed=42)])

    def test_cv_reports_baselines_and_counts(self):
        rm, m = self._merged()
        summary, folds = rm.cross_validate(m, k=5, seed=42, repeats=2)
        for task in ["detection", "segmentation"]:
            self.assertEqual(summary[task]["n_samples"], 40)
            self.assertEqual(summary[task]["n_corrected"], 8)
            self.assertEqual(set(summary[task]["mae"]), {"ridge", "zero", "train_median"})
        self.assertEqual(int(folds.groupby("task")["n_corrected_test"].sum().iloc[0]), 8)

    def test_artifact_round_trip_without_log_rows(self):
        import json
        import tempfile
        from pathlib import Path
        rm, m = self._merged()
        coefs = rm.fit_final(m)
        summary, _ = rm.cross_validate(m, k=5, seed=42, repeats=1)
        art = rm.build_artifact(m, coefs, summary, {"seed": 42}, [{"file": "x.csv"}])
        text = json.dumps(art, ensure_ascii=False)
        self.assertNotIn("img000", text)          # không có image_id / dòng log
        self.assertNotIn("ghi chú", text)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.json"
            p.write_text(text, encoding="utf-8")
            est = rm.load_trained_ridge(p)
        pred = est.predict(m)
        self.assertTrue((pred >= 0).all())
        expected = np.clip(rm._design(m[m.task == "detection"]) @ coefs["detection"], 0, None)
        np.testing.assert_allclose(pred[m.task == "detection"].to_numpy(), expected.round(1), atol=1e-4)


class TestSimpleModels(unittest.TestCase):
    """m31/simple_models.py: hurdle và LAD cho dữ liệu nhiều số 0."""

    def _X(self, n):
        rng = np.random.default_rng(0)
        return np.column_stack([np.ones(n), rng.integers(0, 10, n), rng.integers(0, 5, n), rng.uniform(0, 1, n)])

    def test_lad_predicts_median_on_mostly_zero_target(self):
        from m31.simple_models import fit_predict_lad
        X = self._X(100)
        y = np.zeros(100); y[:13] = 9.0            # 87% số 0, không liên quan đặc trưng
        self.assertLess(fit_predict_lad(X, y, X).max(), 0.01)

    def test_lad_recovers_linear_relation(self):
        from m31.simple_models import fit_predict_lad
        X = self._X(60)
        y = 1.0 + 2.0 * X[:, 1]
        np.testing.assert_allclose(fit_predict_lad(X, y, X), y, atol=1e-3)

    def test_hurdle_uses_positive_median_when_separable(self):
        from m31.simple_models import fit_predict_hurdle
        X = self._X(80)
        y = np.where(X[:, 1] >= 7, 10.0, 0.0)       # cần sửa <=> nhiều box
        pred = fit_predict_hurdle(X, y, X)
        self.assertTrue(set(np.unique(pred)) <= {0.0, 10.0})
        self.assertGreater(((pred > 0) == (y > 0)).mean(), 0.9)


class TestModelVersionMetrics(unittest.TestCase):
    """Hàm thuần của scripts/compare_model_versions.py (không cần torch/ultralytics)."""

    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import compare_model_versions as cmv
        cls.cmv = cmv

    def test_greedy_match_one_to_one(self):
        pred = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], float)
        ref = np.array([[0, 0, 10, 10]], float)
        n, ious = self.cmv.greedy_match(pred, np.array([0.9, 0.8, 0.7]), ref)
        self.assertEqual(n, 1)                      # một box tham chiếu chỉ ghép được một lần
        self.assertAlmostEqual(ious[0], 1.0)
        self.assertEqual(self.cmv.greedy_match(pred, np.ones(3), np.zeros((0, 4)))[0], 0)

    def test_mask_scores_ignore_and_region(self):
        gt = np.zeros((4, 4), np.uint8)
        gt[0, :2] = 13                              # 2 pixel car
        gt[3, :] = 255                              # hàng bỏ qua
        pred = np.zeros((4, 4), bool)
        pred[0, 0] = True                           # đúng 1
        pred[3, :] = True                           # rơi vào vùng 255 -> không tính
        s = self.cmv.mask_scores(pred, gt)
        self.assertEqual((s["inter"], s["union"]), (1, 2))
        self.assertAlmostEqual(s["dice"], 2 / 3)
        region = np.zeros((4, 4), bool)
        region[0, 0] = True
        self.assertAlmostEqual(self.cmv.mask_scores(pred, gt, region)["iou"], 1.0)

    def test_extract_det_labels_only_car_for_requested_ids(self):
        import json
        import tempfile
        import zipfile
        from pathlib import Path
        data = [{"name": "a.jpg", "labels": [{"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
                                             {"category": "truck", "box2d": {"x1": 0, "y1": 0, "x2": 9, "y2": 9}}]},
                {"name": "b.jpg", "labels": [{"category": "car", "box2d": {"x1": 5, "y1": 5, "x2": 6, "y2": 6}}]}]
        with tempfile.TemporaryDirectory() as d:
            zp = Path(d) / "a.zip"
            with zipfile.ZipFile(zp, "w") as z:
                z.writestr(self.cmv.DET_LABELS_IN_ARCHIVE, json.dumps(data))
            with zipfile.ZipFile(zp) as z:
                out = self.cmv.extract_det_labels(z, ["a", "zzz"])
        self.assertEqual(out, {"a": [[1, 2, 3, 4]]})


class TestModelZooStats(unittest.TestCase):
    """Hàm thống kê của scripts/compare_model_zoo.py."""

    def test_ratio_and_bootstrap(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import compare_model_zoo as z
        num, den = np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])
        self.assertAlmostEqual(z.ratio(num, den)(np.arange(3)), 0.5)
        self.assertTrue(np.isnan(z.ratio(num, np.zeros(3))(np.arange(3))))
        lo, hi = z.boot(z.ratio(num, den), 3, np.random.default_rng(0))
        self.assertAlmostEqual(lo, 0.5)              # mọi ảnh cùng tỉ lệ -> CI suy biến tại 0.5
        self.assertAlmostEqual(hi, 0.5)


class TestSamEvalAudit(unittest.TestCase):
    """scripts/audit_sam_eval.py: tách tập 'có prompt' và tập 'IoU xác định', tái tạo đúng số cũ."""

    def test_sets_and_recompute(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        rows = []
        # img1, img2: có prompt; img3: không prompt nhưng có car; img4: không prompt, không car
        spec = {"sam_b": [(100, 80, 0.8), (50, 50, 0.6)], "sam2_b": [(100, 90, 0.9), (50, 40, 0.5)]}
        for m, vals in spec.items():
            for k, (G, P, d) in enumerate(vals, 1):
                rows.append({"model": m, "task": "segmentation", "image_id": f"img{k}", "status": "ok", "n_prompts": 1,
                             "gt_car_px": G, "pred_px": P, "dice": d, "iou": d / (2 - d), "iou_in_prompt_boxes": 0.9})
            rows.append({"model": m, "task": "segmentation", "image_id": "img3", "status": "no_prompt", "n_prompts": 0,
                         "gt_car_px": 30, "pred_px": 0, "dice": 0.0, "iou": 0.0, "iou_in_prompt_boxes": None})
            rows.append({"model": m, "task": "segmentation", "image_id": "img4", "status": "no_prompt", "n_prompts": 0,
                         "gt_car_px": 0, "pred_px": 0, "dice": None, "iou": None, "iou_in_prompt_boxes": None})
        df = pd.DataFrame(rows)
        P, G, d = df.pred_px.astype(float), df.gt_car_px.astype(float), df.dice.fillna(0.0)
        df["I"], df["U"] = d * (P + G) / 2, P + G - d * (P + G) / 2
        summ = [{"model": m, "mean_sec": 0.1, "pooled_iou": float(g.I.sum() / g.U.sum()),
                 "mean_iou_per_image": float(g.iou.mean())} for m, g in df.groupby("model", sort=False)]
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            df.drop(columns=["I", "U"]).to_csv(run / "per_image_results.csv", index=False)
            (run / "segmentation_summary.json").write_text(json.dumps(summ), encoding="utf-8")
            r = subprocess.run([sys.executable, str(root / "scripts" / "audit_sam_eval.py"), "--run", str(run)],
                               capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(r.returncode, 0, r.stderr)
            audit = json.loads((run / "sam_eval_audit.json").read_text(encoding="utf-8"))
            rec = pd.read_csv(run / "sam_eval_recomputed.csv")
        self.assertEqual(audit["set_A_92_prompted"], 2)
        self.assertEqual(audit["set_B_98_iou_defined"], 3)
        self.assertEqual(audit["set_B_minus_A"], ["img3"])
        self.assertEqual(audit["not_scored"], ["img4"])
        self.assertLess(audit["max_abs_diff_pooled_iou"], 1e-9)
        a = rec[(rec.eval_set == "A_92_prompted")].set_index("model")
        self.assertEqual(int(a.loc["sam_b", "n_images"]), 2)


class TestModelBenchmarkLoader(unittest.TestCase):
    """m31/model_benchmark.py: đọc đúng file, dùng tập 92 ảnh cho SAM, báo thiếu dữ liệu thay vì lỗi."""

    def _write_run(self, run, with_recomputed=True, with_det=True):
        import json
        run.mkdir(parents=True)
        (run / "run_config.json").write_text(json.dumps({"n_images": 100, "hardware": {"gpu": "GPU-X"}}), encoding="utf-8")
        if with_det:
            (run / "detection_summary.json").write_text(json.dumps([
                {"model": "yolov8n", "n_images_ok": 100, "mean_sec": 0.014, "total_car_boxes": 557, "gt_images": 14,
                 "gt_boxes": 126, "tp": 77, "fp": 13, "fn": 49, "precision": 0.856, "precision_ci95": [0.79, 0.93],
                 "recall": 0.611, "recall_ci95": [0.54, 0.71]},
                {"model": "yolo11m", "reference": "yolov8n", "n_images_ok": 100, "mean_sec": 0.022, "total_car_boxes": 676,
                 "gt_images": 14, "gt_boxes": 126, "tp": 88, "fp": 8, "fn": 38, "precision": 0.917,
                 "precision_ci95": [0.87, 0.96], "recall": 0.698, "recall_ci95": [0.62, 0.79],
                 "precision_diff_vs_ref_ci95": [-0.004, 0.137], "recall_diff_vs_ref_ci95": [0.033, 0.137]}]), encoding="utf-8")
        (run / "segmentation_summary.json").write_text(json.dumps([
            {"model": "sam_b", "n_images_ok": 92, "n_images_with_prompt": 92, "mean_sec": 0.30, "total_sec": 27.6,
             "pooled_iou_in_prompt_boxes": 0.894},
            {"model": "sam2_t", "reference": "sam_b", "n_images_ok": 92, "n_images_with_prompt": 92, "mean_sec": 0.12,
             "total_sec": 11.0, "pooled_iou_in_prompt_boxes": 0.911}]), encoding="utf-8")
        if with_recomputed:
            rows = []
            for es, n, b, t in [("A_92_prompted", 92, 0.775, 0.790), ("B_98_iou_defined", 98, 0.771, 0.786)]:
                rows.append({"eval_set": es, "n_images": n, "model": "sam_b", "reference": "sam_b", "mean_sec": 0.3,
                             "pooled_iou": b, "mean_iou_per_image": 0.76})
                rows.append({"eval_set": es, "n_images": n, "model": "sam2_t", "reference": "sam_b", "mean_sec": 0.12,
                             "pooled_iou": t, "mean_iou_per_image": 0.77, "pooled_iou_diff_vs_ref": t - b,
                             "n_images_better": 46, "n_images_worse": 41, "n_images_tie": 5})
            pd.DataFrame(rows).to_csv(run / "sam_eval_recomputed.csv", index=False)

    def test_loads_main_sam_set_and_detection(self):
        import tempfile
        from pathlib import Path
        from m31.model_benchmark import load_benchmark
        with tempfile.TemporaryDirectory() as d:
            self._write_run(Path(d) / "r1")
            b = load_benchmark(Path(d))
        self.assertEqual(b.problems, [])
        self.assertEqual(set(b.detection.model), {"yolov8n", "yolo11m"})
        self.assertTrue(b.detection.set_index("model").loc["yolov8n", "is_reference"])
        self.assertEqual(set(b.segmentation.n_images), {92})                 # chỉ tập 92, không lấy 98
        s = b.segmentation.set_index("model")
        self.assertAlmostEqual(s.loc["sam2_t", "pooled_iou"], 0.790)
        self.assertAlmostEqual(s.loc["sam2_t", "speedup_vs_ref"], 0.30 / 0.12)

    def test_missing_files_reported_not_faked(self):
        import tempfile
        from pathlib import Path
        from m31.model_benchmark import load_benchmark
        with tempfile.TemporaryDirectory() as d:
            self._write_run(Path(d) / "r1", with_recomputed=False, with_det=False)
            b = load_benchmark(Path(d))
            none = load_benchmark(Path(d) / "khong_ton_tai")
        self.assertTrue(b.detection.empty and b.segmentation.empty)       # không thay bằng tập 98
        self.assertEqual(len(b.problems), 2)
        self.assertFalse(none.available)
        self.assertEqual(len(none.problems), 1)

    def test_display_names(self):
        from m31.model_benchmark import display_name, yolo_sort_key
        self.assertEqual([display_name(m) for m in ["yolov8n", "yolo11m", "sam2.1_t", "sam_b", "mobile_sam"]],
                         ["YOLOv8n", "YOLO11m", "SAM2.1-T", "SAM-B", "MobileSAM"])
        self.assertEqual(sorted(["yolo11m", "yolov8x", "yolov8n", "yolo11n"], key=yolo_sort_key),
                         ["yolov8n", "yolov8x", "yolo11n", "yolo11m"])


class TestPilotImages(unittest.TestCase):
    """data/real/pilot_images/: đúng 100 ảnh pilot × 2 task, kèm giấy phép, không file thừa, dung lượng hợp lý."""

    def test_previews_match_pilot_ids(self):
        from pathlib import Path

        from PIL import Image
        root = Path(__file__).resolve().parent.parent
        ids = set(pd.read_csv(root / "data/real/processed/pilot100_prelabels.csv")["image_id"])
        self.assertEqual(len(ids), 100)
        base = root / "data/real/pilot_images"
        self.assertTrue((base / "LICENSE_BDD100K.md").is_file())
        total = 0
        for task in ("detection", "segmentation"):
            files = sorted((base / task).glob("*"))
            self.assertEqual({f.stem for f in files}, ids, task)               # đúng 100 ID, không thừa
            self.assertTrue(all(f.suffix == ".jpg" for f in files))
            for f in files:
                total += f.stat().st_size
                self.assertLess(f.stat().st_size, 100 * 1024 * 1024)
            with Image.open(files[0]) as im:
                self.assertEqual(im.size, (1280, 720))
        self.assertEqual({p.name for p in base.iterdir()}, {"detection", "segmentation", "LICENSE_BDD100K.md"})
        self.assertLess(total, 200 * 1000 * 1000)


if __name__ == "__main__":
    unittest.main()
