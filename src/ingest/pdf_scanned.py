"""Scanned/image adapter: render -> OpenCV preprocess -> PaddleOCR -> shared
regionize.build_regions (same typing as native) -> optional vision-verify of
low-confidence regions. Vision is text-cleanup only, never the coordinate source."""
from __future__ import annotations

import os

os.environ.setdefault("FLAGS_use_mkldnn", "0")  # paddle 3.x oneDNN CPU kernel is buggy

import fitz
import numpy as np

from src.canonical.model import BBox, Document, Page, Token
from src.config import get_settings
from src.ingest.base import FormatAdapter, register
from src.ingest.regionize import Line, build_regions
from src.ingest.resolver import ResolvedDoc

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# faster/smaller vs accurate model pairs (PaddleOCR 3.x)
_MODEL_TIERS = {
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
}

# module-level singleton: load the OCR model once, reuse across adapter instances
# (adapter is re-created per comparison; the model must not reload each time)
_OCR_CACHE: dict = {}


def _get_ocr(cfg):
    import paddle
    from paddleocr import PaddleOCR

    use_gpu = cfg.device == "gpu" or (cfg.device == "auto" and paddle.is_compiled_with_cuda())
    device = "gpu" if use_gpu else "cpu"
    key = (cfg.model_tier, device, cfg.cpu_threads, cfg.use_textline_orientation)
    if key in _OCR_CACHE:
        return _OCR_CACHE[key]

    kwargs = dict(
        lang="en", device=device,
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=cfg.use_textline_orientation,
        enable_mkldnn=False,
        cpu_threads=int(cfg.cpu_threads),
    )
    det_model, rec_model = _MODEL_TIERS.get(cfg.model_tier, (None, None))
    if det_model:
        kwargs["text_detection_model_name"] = det_model
        kwargs["text_recognition_model_name"] = rec_model
    try:
        ocr = PaddleOCR(**kwargs)
    except Exception:  # unknown model name / arg on this paddleocr build -> safe default
        ocr = PaddleOCR(lang="en", device=device,
                        use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=cfg.use_textline_orientation,
                        enable_mkldnn=False)
    _OCR_CACHE[key] = ocr
    return ocr


@register("scanned_pdf")
class ScannedPdfAdapter(FormatAdapter):
    def __init__(self):
        self.cfg = get_settings().ingest.scanned
        self._ocr = None  # lazy: model init is heavy

    def parse(self, resolved: ResolvedDoc) -> Document:
        images = self._render(resolved)
        pages: list[Page] = []
        for pindex, img in enumerate(images):
            pages.append(self._parse_page(resolved.pid, img, pindex))
        return Document(
            pid=resolved.pid, doc_family=resolved.doc_family, rev_label=resolved.rev_label,
            source_format="scanned_pdf", page_count=len(pages),
            metadata={"path": resolved.path, "dpi": self.cfg.dpi}, pages=pages,
        )

    # --- 1. render ---
    def _render(self, resolved: ResolvedDoc) -> list[np.ndarray]:
        from pathlib import Path
        if Path(resolved.path).suffix.lower() in _IMAGE_EXTS:
            import cv2
            arr = cv2.imdecode(np.frombuffer(resolved.raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            return [arr]
        imgs = []
        with fitz.open(stream=resolved.raw_bytes, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=self.cfg.dpi)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    arr = arr[:, :, :3]
                imgs.append(arr[:, :, ::-1].copy())  # RGB -> BGR for cv2
        return imgs

    # --- 2. preprocess ---
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not self.cfg.deskew:
            return gray
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ink = np.column_stack(np.where(binary > 0))
        angle = cv2.minAreaRect(ink)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if 0.3 < abs(angle) < 10:  # conservative: skip tiny/implausible skews
            h, w = gray.shape
            rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            gray = cv2.warpAffine(gray, rot_matrix, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return gray

    # --- 3-5. OCR -> regions -> verify ---
    def _parse_page(self, pid: str, img: np.ndarray, pindex: int) -> Page:
        proc = self._preprocess(img)
        detections = self._ocr_lines(proc)  # [(text, (x0,y0,x1,y1), conf)]
        ph, pw = proc.shape[:2]
        lines = [Line(text, bbox, [Token(text, BBox(*bbox), conf)])
                 for (text, bbox, conf) in detections]
        regions = build_regions(pid, pindex, lines, float(pw), float(ph), provenance="ocr")
        if self.cfg.vision_verify:
            self._vision_verify(regions, img)
        return Page(page_index=pindex, width=float(pw), height=float(ph), rotation=0,
                    regions=regions, raw_text=" ".join(d[0] for d in detections),
                    metadata={"ocr_engine": self.cfg.ocr_engine, "dpi": self.cfg.dpi})

    def _ocr_lines(self, img: np.ndarray):
        if img.ndim == 2:  # PaddleOCR expects 3-channel
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self._ocr is None:
            self._ocr = _get_ocr(self.cfg)
        result = self._ocr.predict(img)
        if not result:
            return []
        page = result[0]
        texts, scores = page["rec_texts"], page["rec_scores"]
        boxes = page["rec_boxes"] if len(page.get("rec_boxes", [])) else page["rec_polys"]
        detections = []
        for text, score, box in zip(texts, scores, boxes):
            if not text.strip():
                continue
            box_arr = np.array(box)
            if box_arr.ndim == 2:  # polygon -> axis-aligned
                x0, y0 = box_arr[:, 0].min(), box_arr[:, 1].min()
                x1, y1 = box_arr[:, 0].max(), box_arr[:, 1].max()
            else:                  # already [x0,y0,x1,y1]
                x0, y0, x1, y1 = box_arr.tolist()
            detections.append((text.strip(), (float(x0), float(y0), float(x1), float(y1)), float(score)))
        return detections

    def _vision_verify(self, regions, img) -> None:
        """Re-transcribe the N lowest-confidence regions with the vision model
        (text only, box kept). No-op without an API key."""
        import base64
        import os

        threshold = self.cfg.ocr_conf_verify_below
        max_calls = int(getattr(self.cfg, "vision_max_calls", 30))
        low_conf = sorted([r for r in regions if r.confidence < threshold and r.text.strip()],
                          key=lambda r: r.confidence)[:max_calls]
        if not low_conf or not os.environ.get("OPENAI_API_KEY"):
            return

        import cv2
        from langchain_core.messages import HumanMessage

        from src.chat.llm import build_vision_model
        from src.config import get_settings
        from src.observability.tracing import record_llm

        model = build_vision_model()
        vmodel_name = get_settings().llm.reason_model
        vprice = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.60)}.get(vmodel_name, (2.5, 10.0))
        h, w = img.shape[:2]
        prompt = ("This is a small crop from an engineering drawing (P&ID). "
                  "Transcribe the text EXACTLY as printed — same characters, tags, "
                  "numbers, punctuation. Output only the transcription, no commentary.")

        pad = 4
        for region in low_conf:
            box = region.bbox
            x0, y0 = max(0, int(box.x0) - pad), max(0, int(box.y0) - pad)
            x1, y1 = min(w, int(box.x1) + pad), min(h, int(box.y1) + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            ok, buf = cv2.imencode(".png", img[y0:y1, x0:x1])
            if not ok:
                continue
            data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
            try:
                msg = HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ])
                resp = model.invoke([msg])
                text = (resp.content or "").strip()
                usage = getattr(resp, "usage_metadata", None) or {}
                tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                record_llm(vmodel_name, tokens_in, tokens_out,
                           (tokens_in * vprice[0] + tokens_out * vprice[1]) / 1e6)
            except Exception:
                continue  # keep the OCR text on failure
            if text:
                region.text = text
                region.attrs["text"] = text
                region.provenance = "ocr+vision"
