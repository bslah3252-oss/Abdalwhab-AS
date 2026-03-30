import os
import io
import base64
import threading
import numpy as np
import cv2
import urllib.request

os.environ["KIVY_NO_CONSOLELOG"] = "0"

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.image import Image as KivyImage
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.colorpicker import ColorPicker
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.progressbar import ProgressBar
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
from PIL import Image, ImageEnhance
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

Window.clearcolor = (0.06, 0.07, 0.09, 1)

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
INNER_MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
CONTOUR_CONNECTIONS = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291),
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]


def get_model_path():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_dir, MODEL_FILENAME)


def download_model_if_needed(progress_callback=None):
    path = get_model_path()
    if os.path.exists(path):
        return path

    def _reporthook(count, block_size, total_size):
        if progress_callback and total_size > 0:
            pct = min(1.0, count * block_size / total_size)
            progress_callback(pct)

    urllib.request.urlretrieve(MODEL_URL, path, reporthook=_reporthook)
    return path


def lm_to_px(lm, h, w):
    return int(lm.x * w), int(lm.y * h)


def get_polygon(landmarks, h, w, indices):
    pts = [lm_to_px(landmarks[i], h, w) for i in indices]
    return np.array(pts, dtype=np.int32)


def whiten_teeth(img, mask, level=0.7):
    if not np.any(mask):
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.where(mask > 0, hsv[:, :, 1] * (1.0 - level * 0.85), hsv[:, :, 1])
    hsv[:, :, 2] = np.where(mask > 0, np.clip(hsv[:, :, 2] + level * 65, 0, 255), hsv[:, :, 2])
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_lip_color(img, lip_mask, color_bgr, opacity=0.4):
    overlay = img.copy()
    overlay[lip_mask > 0] = color_bgr
    return cv2.addWeighted(img, 1 - opacity, overlay, opacity, 0)


def apply_skin_smoothing(img, face_mask, smoothing=0.5):
    if smoothing <= 0:
        return img
    ksize = max(3, int(smoothing * 18) * 2 + 1)
    blurred = cv2.bilateralFilter(img, ksize, 75, 75)
    alpha = np.zeros(img.shape, dtype=np.float32)
    alpha[face_mask > 0] = smoothing
    return (blurred.astype(np.float32) * alpha + img.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def draw_contours(img, landmarks, h, w):
    result = img.copy()
    for (a, b) in CONTOUR_CONNECTIONS:
        if a < len(landmarks) and b < len(landmarks):
            cv2.line(result, lm_to_px(landmarks[a], h, w), lm_to_px(landmarks[b], h, w),
                     (0, 255, 128), 1, cv2.LINE_AA)
    return result


def draw_mesh(img, landmarks, h, w):
    result = img.copy()
    for lm in landmarks:
        cv2.circle(result, lm_to_px(lm, h, w), 1, (0, 200, 255), -1, cv2.LINE_AA)
    return result


def apply_dsd(image_bgr, params, model_path):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionTaskRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return image_bgr, "No face detected. Use a clear front-facing photo."

    lms = result.face_landmarks[0]
    out = image_bgr.copy()

    inner_mouth_pts = get_polygon(lms, h, w, INNER_MOUTH)
    teeth_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(teeth_mask, [inner_mouth_pts], 255)

    upper_pts = get_polygon(lms, h, w, UPPER_LIP_OUTER + list(reversed(UPPER_LIP_INNER)))
    lower_pts = get_polygon(lms, h, w, LOWER_LIP_OUTER + list(reversed(LOWER_LIP_INNER)))
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [upper_pts, lower_pts], 255)
    lip_mask = cv2.subtract(lip_mask, teeth_mask)

    face_pts = get_polygon(lms, h, w, FACE_OVAL)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [face_pts], 255)

    whitening = float(params.get("whitening", 0.5))
    if whitening > 0:
        out = whiten_teeth(out, teeth_mask, whitening)

    lip_color = params.get("lip_color", (0, 68, 204))
    lip_opacity = float(params.get("lip_opacity", 0.0))
    if lip_opacity > 0:
        out = apply_lip_color(out, lip_mask, lip_color, lip_opacity)

    smoothing = float(params.get("smoothing", 0.0))
    if smoothing > 0:
        out = apply_skin_smoothing(out, face_mask, smoothing)

    brightness = float(params.get("brightness", 1.0))
    if brightness != 1.0:
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Brightness(pil).enhance(max(0.1, brightness))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    contrast = float(params.get("contrast", 1.0))
    if contrast != 1.0:
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        pil = ImageEnhance.Contrast(pil).enhance(max(0.1, contrast))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    if params.get("show_mesh"):
        out = draw_mesh(out, lms, h, w)
    if params.get("show_landmarks"):
        out = draw_contours(out, lms, h, w)

    return out, f"Done — {len(lms)} landmarks detected"


def bgr_to_kivy_texture(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    flipped = cv2.flip(rgb, 0)
    h, w = flipped.shape[:2]
    texture = Texture.create(size=(w, h), colorfmt="rgb")
    texture.blit_buffer(flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    return texture


def save_image(bgr_img, path):
    cv2.imwrite(path, bgr_img)


class LabeledSlider(BoxLayout):
    def __init__(self, label, min_val, max_val, default, fmt="{:.0f}%", **kwargs):
        super().__init__(orientation="vertical", size_hint_y=None, height=dp(56), **kwargs)
        self.fmt = fmt
        self._value = default

        header = BoxLayout(size_hint_y=None, height=dp(20))
        self.name_lbl = Label(
            text=label, halign="left", valign="middle",
            color=get_color_from_hex("#CBD5E1"), font_size=dp(12),
        )
        self.name_lbl.bind(size=self.name_lbl.setter("text_size"))
        self.val_lbl = Label(
            text=self._format(default), halign="right", valign="middle",
            color=get_color_from_hex("#60A5FA"), font_size=dp(12), bold=True,
        )
        self.val_lbl.bind(size=self.val_lbl.setter("text_size"))
        header.add_widget(self.name_lbl)
        header.add_widget(self.val_lbl)
        self.add_widget(header)

        self.slider = Slider(min=min_val, max=max_val, value=default,
                             cursor_size=(dp(18), dp(18)))
        self.slider.bind(value=self._on_value)
        self.add_widget(self.slider)

    def _format(self, v):
        return self.fmt.format(v)

    def _on_value(self, instance, value):
        self._value = value
        self.val_lbl.text = self._format(value)

    @property
    def value(self):
        return self.slider.value

    def reset(self, default):
        self.slider.value = default


class DSDRoot(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)
        self._original_bgr = None
        self._result_bgr = None
        self._model_path = None
        self._processing = False
        self._lip_color_bgr = (0, 68, 204)
        self._show_landmarks = False
        self._show_mesh = False

        self._build_ui()
        Clock.schedule_once(self._init_model, 0.5)

    def _build_ui(self):
        # ── Top bar ──
        top = BoxLayout(size_hint_y=None, height=dp(50),
                        padding=[dp(14), 0], spacing=dp(10))
        with top.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(0.1, 0.11, 0.17, 1)
            self._top_rect = Rectangle(pos=top.pos, size=top.size)
        top.bind(pos=lambda i, v: setattr(self._top_rect, "pos", v),
                 size=lambda i, v: setattr(self._top_rect, "size", v))

        top.add_widget(Label(
            text="[b]Digital Smile Design[/b]",
            markup=True, font_size=dp(16),
            color=get_color_from_hex("#E2E8F0"),
            size_hint_x=None, width=dp(220),
        ))
        self._status_lbl = Label(
            text="Loading model...",
            font_size=dp(11),
            color=get_color_from_hex("#8892B0"),
        )
        top.add_widget(self._status_lbl)
        self.add_widget(top)

        # ── Main split ──
        main = BoxLayout(orientation="horizontal", spacing=dp(10), padding=dp(10))
        self.add_widget(main)

        # ── Left: preview ──
        preview_col = BoxLayout(orientation="vertical", spacing=dp(6))
        main.add_widget(preview_col)

        view_bar = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
        for label, view in [("Original", "original"), ("Result", "result"), ("Split", "split")]:
            btn = ToggleButton(
                text=label, group="view",
                font_size=dp(12),
                background_normal="",
                background_down="",
                background_color=get_color_from_hex("#1E2235"),
                color=get_color_from_hex("#8892B0"),
                state="down" if view == "split" else "normal",
            )
            btn.bind(on_press=lambda x, v=view: self._set_view(v))
            view_bar.add_widget(btn)
        preview_col.add_widget(view_bar)

        self._split_box = BoxLayout(orientation="horizontal", spacing=dp(6))
        self._img_original = KivyImage(allow_stretch=True, keep_ratio=True)
        self._img_result = KivyImage(allow_stretch=True, keep_ratio=True)
        self._split_box.add_widget(self._img_original)
        self._split_box.add_widget(self._img_result)
        preview_col.add_widget(self._split_box)

        self._img_single = KivyImage(allow_stretch=True, keep_ratio=True)
        self._img_single.opacity = 0
        preview_col.add_widget(self._img_single)

        btn_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
        self._upload_btn = Button(
            text="Upload Photo", font_size=dp(13),
            background_normal="", background_color=get_color_from_hex("#1E40AF"),
        )
        self._upload_btn.bind(on_press=self._open_filechooser)
        self._save_btn = Button(
            text="Save Result", font_size=dp(13),
            background_normal="", background_color=get_color_from_hex("#065F46"),
            disabled=True,
        )
        self._save_btn.bind(on_press=self._save_result)
        btn_row.add_widget(self._upload_btn)
        btn_row.add_widget(self._save_btn)
        preview_col.add_widget(btn_row)

        # ── Right: controls ──
        ctrl_scroll = ScrollView(size_hint_x=None, width=dp(260))
        ctrl_col = BoxLayout(orientation="vertical", spacing=dp(10),
                             padding=dp(10), size_hint_y=None)
        ctrl_col.bind(minimum_height=ctrl_col.setter("height"))
        ctrl_scroll.add_widget(ctrl_col)
        main.add_widget(ctrl_scroll)

        def section(title):
            lbl = Label(
                text=title, font_size=dp(11), bold=True,
                color=get_color_from_hex("#4A5568"),
                size_hint_y=None, height=dp(22),
                halign="left",
            )
            lbl.bind(size=lbl.setter("text_size"))
            ctrl_col.add_widget(lbl)

        section("SMILE DESIGN")
        self._sl_whitening = LabeledSlider("Teeth Whitening", 0, 100, 50)
        ctrl_col.add_widget(self._sl_whitening)

        section("LIP COLOR")
        self._sl_lip_opacity = LabeledSlider("Lip Opacity", 0, 100, 0)
        ctrl_col.add_widget(self._sl_lip_opacity)
        lip_color_btn = Button(
            text="Choose Lip Color", font_size=dp(12),
            size_hint_y=None, height=dp(36),
            background_normal="", background_color=get_color_from_hex("#7C1D1D"),
        )
        lip_color_btn.bind(on_press=self._open_color_picker)
        ctrl_col.add_widget(lip_color_btn)
        self._lip_preview = Label(
            text="Current: #CC4444",
            font_size=dp(11),
            color=get_color_from_hex("#CC4444"),
            size_hint_y=None, height=dp(18),
        )
        ctrl_col.add_widget(self._lip_preview)

        section("SKIN")
        self._sl_smoothing = LabeledSlider("Smoothing", 0, 100, 0)
        ctrl_col.add_widget(self._sl_smoothing)

        section("ADJUSTMENTS")
        self._sl_brightness = LabeledSlider("Brightness", -50, 50, 0, fmt="{:+.0f}")
        self._sl_contrast = LabeledSlider("Contrast", -50, 50, 0, fmt="{:+.0f}")
        ctrl_col.add_widget(self._sl_brightness)
        ctrl_col.add_widget(self._sl_contrast)

        section("OVERLAYS")
        lm_btn = ToggleButton(
            text="Face Contours", font_size=dp(12),
            size_hint_y=None, height=dp(34),
            background_normal="", background_down="",
            background_color=get_color_from_hex("#1E2235"),
        )
        mesh_btn = ToggleButton(
            text="Full Mesh", font_size=dp(12),
            size_hint_y=None, height=dp(34),
            background_normal="", background_down="",
            background_color=get_color_from_hex("#1E2235"),
        )
        lm_btn.bind(state=lambda i, s: setattr(self, "_show_landmarks", s == "down"))
        mesh_btn.bind(state=lambda i, s: setattr(self, "_show_mesh", s == "down"))
        ctrl_col.add_widget(lm_btn)
        ctrl_col.add_widget(mesh_btn)

        self._apply_btn = Button(
            text="Apply Design", font_size=dp(14), bold=True,
            size_hint_y=None, height=dp(46),
            background_normal="", background_color=get_color_from_hex("#2563EB"),
            disabled=True,
        )
        self._apply_btn.bind(on_press=self._apply_dsd)
        ctrl_col.add_widget(self._apply_btn)

        reset_btn = Button(
            text="Reset All", font_size=dp(12),
            size_hint_y=None, height=dp(36),
            background_normal="", background_color=get_color_from_hex("#374151"),
        )
        reset_btn.bind(on_press=self._reset)
        ctrl_col.add_widget(reset_btn)

        self._progress = ProgressBar(max=100, value=0, size_hint_y=None, height=dp(6))
        ctrl_col.add_widget(self._progress)

        self._set_view("split")

    def _set_view(self, view):
        self._current_view = view
        if view == "split":
            self._split_box.opacity = 1
            self._img_single.opacity = 0
            self._split_box.size_hint_y = 1
            self._img_single.size_hint_y = None
            self._img_single.height = 0
        else:
            self._split_box.opacity = 0
            self._img_single.opacity = 1
            self._split_box.size_hint_y = None
            self._split_box.height = 0
            self._img_single.size_hint_y = 1
            if view == "original" and self._original_bgr is not None:
                self._img_single.texture = bgr_to_kivy_texture(self._original_bgr)
            elif view == "result" and self._result_bgr is not None:
                self._img_single.texture = bgr_to_kivy_texture(self._result_bgr)

    def _init_model(self, dt):
        def _download():
            try:
                def _progress(pct):
                    Clock.schedule_once(lambda dt: setattr(self._progress, "value", pct * 100), 0)
                    Clock.schedule_once(lambda dt: setattr(self._status_lbl, "text",
                                                           f"Downloading model {pct*100:.0f}%..."), 0)
                path = download_model_if_needed(progress_callback=_progress)
                self._model_path = path
                Clock.schedule_once(self._on_model_ready, 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._set_status(f"Model error: {e}", error=True), 0)

        threading.Thread(target=_download, daemon=True).start()

    def _on_model_ready(self, dt):
        self._set_status("Ready — upload a photo to begin")
        self._progress.value = 0

    def _set_status(self, msg, error=False):
        self._status_lbl.text = msg
        self._status_lbl.color = get_color_from_hex("#EF4444") if error else get_color_from_hex("#8892B0")

    def _open_filechooser(self, instance):
        content = BoxLayout(orientation="vertical")
        fc = FileChooserListView(filters=["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"])
        content.add_widget(fc)
        btns = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8), padding=dp(6))

        popup = Popup(title="Select an image", content=content,
                      size_hint=(0.9, 0.85))

        def _load(inst):
            if fc.selection:
                self._load_image(fc.selection[0])
                popup.dismiss()

        load_btn = Button(text="Open", background_normal="",
                          background_color=get_color_from_hex("#2563EB"))
        load_btn.bind(on_press=_load)
        cancel_btn = Button(text="Cancel", background_normal="",
                            background_color=get_color_from_hex("#374151"))
        cancel_btn.bind(on_press=popup.dismiss)
        btns.add_widget(load_btn)
        btns.add_widget(cancel_btn)
        content.add_widget(btns)
        popup.open()

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            self._set_status("Failed to load image.", error=True)
            return
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        self._original_bgr = img
        self._result_bgr = None
        self._img_original.texture = bgr_to_kivy_texture(img)
        self._img_result.texture = None
        self._apply_btn.disabled = self._model_path is None
        self._save_btn.disabled = True
        self._set_status("Photo loaded — press Apply Design")

    def _open_color_picker(self, instance):
        content = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))
        cp = ColorPicker()
        content.add_widget(cp)
        btns = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        popup = Popup(title="Lip Color", content=content, size_hint=(0.8, 0.8))

        def _ok(inst):
            r, g, b, a = cp.color
            self._lip_color_bgr = (int(b * 255), int(g * 255), int(r * 255))
            hex_r = int(r * 255)
            hex_g = int(g * 255)
            hex_b = int(b * 255)
            hex_str = f"#{hex_r:02X}{hex_g:02X}{hex_b:02X}"
            self._lip_preview.text = f"Current: {hex_str}"
            self._lip_preview.color = (r, g, b, 1)
            popup.dismiss()

        ok_btn = Button(text="OK", background_normal="",
                        background_color=get_color_from_hex("#2563EB"))
        ok_btn.bind(on_press=_ok)
        cancel_btn = Button(text="Cancel", background_normal="",
                            background_color=get_color_from_hex("#374151"))
        cancel_btn.bind(on_press=popup.dismiss)
        btns.add_widget(ok_btn)
        btns.add_widget(cancel_btn)
        content.add_widget(btns)
        popup.open()

    def _apply_dsd(self, instance):
        if self._original_bgr is None or self._model_path is None or self._processing:
            return
        self._processing = True
        self._apply_btn.disabled = True
        self._set_status("Processing...")

        params = {
            "whitening": self._sl_whitening.value / 100.0,
            "lip_color": self._lip_color_bgr,
            "lip_opacity": self._sl_lip_opacity.value / 100.0,
            "smoothing": self._sl_smoothing.value / 100.0,
            "brightness": 1.0 + self._sl_brightness.value / 100.0,
            "contrast": 1.0 + self._sl_contrast.value / 100.0,
            "show_landmarks": self._show_landmarks,
            "show_mesh": self._show_mesh,
        }
        img = self._original_bgr.copy()
        model_path = self._model_path

        def _run():
            try:
                result_img, message = apply_dsd(img, params, model_path)
                Clock.schedule_once(lambda dt: self._on_result(result_img, message), 0)
            except Exception as e:
                Clock.schedule_once(lambda dt: self._on_error(str(e)), 0)

        threading.Thread(target=_run, daemon=True).start()

    def _on_result(self, result_img, message):
        self._result_bgr = result_img
        self._img_result.texture = bgr_to_kivy_texture(result_img)
        if self._current_view == "result":
            self._img_single.texture = bgr_to_kivy_texture(result_img)
        self._save_btn.disabled = False
        self._apply_btn.disabled = False
        self._processing = False
        error = "No face" in message or "error" in message.lower()
        self._set_status(message, error=error)

    def _on_error(self, msg):
        self._apply_btn.disabled = False
        self._processing = False
        self._set_status(f"Error: {msg}", error=True)

    def _save_result(self, instance):
        if self._result_bgr is None:
            return
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.path.expanduser("~"), "Pictures") \
            if os.path.exists(os.path.join(os.path.expanduser("~"), "Pictures")) \
            else os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(save_dir, f"dsd_result_{ts}.jpg")
        save_image(self._result_bgr, path)
        self._set_status(f"Saved: {path}")

    def _reset(self, instance):
        self._sl_whitening.reset(50)
        self._sl_lip_opacity.reset(0)
        self._sl_smoothing.reset(0)
        self._sl_brightness.reset(0)
        self._sl_contrast.reset(0)
        self._show_landmarks = False
        self._show_mesh = False


class DSDApp(App):
    def build(self):
        self.title = "Digital Smile Design"
        return DSDRoot()


if __name__ == "__main__":
    DSDApp().run()
