import os
import sys
import traceback
from datetime import datetime
from collections import deque

import fitz  # PyMuPDF

from PySide6.QtCore import (
    Qt, QRectF, QRect, QPointF, QSize, QElapsedTimer, QEvent
)
from PySide6.QtGui import (
    QAction, QImage, QPainter, QPen, QPixmap, QColor, QTabletEvent,
    QKeySequence, QShortcut, QIcon
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSizePolicy, QDialog
)

# ---------------------------------------------------------------------
# Carga de recursos del paquete (iconos, etc.)
# ---------------------------------------------------------------------
try:
    # Python 3.9+: importlib.resources.files
    from importlib.resources import files as pkg_files
    def pkg_asset(rel_path: str) -> str:
        # ejemplo: pkg_asset("assets/icon.ico")
        return str(pkg_files("quicksignpdf").joinpath(rel_path))
except Exception:
    # Fallback: relativo al cwd del repo
    def pkg_asset(rel_path: str) -> str:
        return os.path.join(os.path.abspath("."), "src", "quicksignpdf", rel_path)

# ---------------------------------------------------------------------
# Recortar imagen por la bounding box de alfa (firma pequeña se adapta)
# ---------------------------------------------------------------------
def crop_alpha_bbox(qimg: QImage, padding: int = 16) -> QImage:
    """Recorta el QImage a la caja mínima que contiene píxeles con alfa>0."""
    w, h = qimg.width(), qimg.height()
    minx, miny, maxx, maxy = w, h, -1, -1
    for y in range(h):
        for x in range(w):
            if qimg.pixelColor(x, y).alpha() > 0:
                if x < minx: minx = x
                if y < miny: miny = y
                if x > maxx: maxx = x
                if y > maxy: maxy = y
    if maxx < 0:
        # Imagen totalmente transparente (nada dibujado)
        return qimg

    minx = max(0, minx - padding)
    miny = max(0, miny - padding)
    maxx = min(w - 1, maxx + padding)
    maxy = min(h - 1, maxy + padding)

    rect = QRect(minx, miny, maxx - minx + 1, maxy - miny + 1)
    return qimg.copy(rect)

# ---------------------------------------------------------------------
# Inserción de firma (PNG bytes) en PDF
# ---------------------------------------------------------------------
def insert_signature_png(pdf_path, page_index, rect_pt, png_bytes):
    doc = fitz.open(pdf_path)
    page_index = max(0, min(page_index, len(doc) - 1))
    page = doc[page_index]
    rect = fitz.Rect(*rect_pt)
    page.insert_image(rect, stream=png_bytes, keep_proportion=True, overlay=True)

    base, ext = os.path.splitext(pdf_path)
    out = f"{base}_firmado.pdf"
    if os.path.exists(out):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"{base}_firmado_{ts}.pdf"

    # Deflate + garbage para PDFs más limpios
    doc.save(out, deflate=True, garbage=4)
    doc.close()
    return out

# ---------------------------------------------------------------------
# Canvas de firma con suavizado + Catmull–Rom
# ---------------------------------------------------------------------
class SignatureCanvas(QWidget):
    """
    Anti-jitter:
    - Exponential smoothing (One-Euro aprox) + promedio móvil (ventana 8).
    - Spline Catmull–Rom con alta subdivisión.
    - Sin líneas fantasma; sólo pinta con presión/botón reales.
    """
    def __init__(self, size_px=QSize(2000, 900), pen_width=7, parent=None):
        super().__init__(parent)
        self.setMinimumSize(900, 300)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.pen_width = pen_width
        self.pen_color = QColor(0, 0, 0)

        # Umbrales
        self.min_start_dist = 6
        self.min_seg_len = max(1.2 * pen_width, 6)
        self.press_threshold = 0.10
        self._pressed = False

        # Buffer de dibujo
        self._img = QImage(size_px, QImage.Format.Format_ARGB32_Premultiplied)
        self._img.fill(Qt.GlobalColor.transparent)

        # Estado del trazo
        self._points = []
        self._raw_prev = None
        the_smooth_prev = None
        self._smooth_prev = the_smooth_prev

        # Exponential smoothing
        self._timer = QElapsedTimer(); self._timer.start()
        self._last_ms = self._timer.elapsed()
        self.smin = 0.14
        self.smax = 0.65
        self.vref = 2.8

        # Promedio móvil corto
        self._ma_win = deque(maxlen=8)

        pal = self.palette()
        pal.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(pal); self.setAutoFillBackground(True)

    def image(self) -> QImage:
        return self._img

    # ---- Suavizado ----
    def _alpha_for(self, v):
        t = max(0.0, min(1.0, v / self.vref))
        return self.smin + (self.smax - self.smin) * t

    def _ema(self, pt: QPointF) -> QPointF:
        now = self._timer.elapsed()
        dt = max(1, now - self._last_ms)
        self._last_ms = now
        if self._smooth_prev is None or self._raw_prev is None:
            self._raw_prev = QPointF(pt); self._smooth_prev = QPointF(pt)
            return QPointF(pt)
        dx = pt.x() - self._raw_prev.x(); dy = pt.y() - self._raw_prev.y()
        v = (dx*dx + dy*dy) ** 0.5 / dt
        a = self._alpha_for(v)
        sx = (1 - a) * self._smooth_prev.x() + a * pt.x()
        sy = (1 - a) * self._smooth_prev.y() + a * pt.y()
        self._raw_prev = QPointF(pt); self._smooth_prev = QPointF(sx, sy)
        return QPointF(sx, sy)

    def _moving_avg(self, pt: QPointF) -> QPointF:
        self._ma_win.append((pt.x(), pt.y()))
        n = len(self._ma_win)
        sx = sum(x for x, _ in self._ma_win) / n
        sy = sum(y for _, y in self._ma_win) / n
        return QPointF(sx, sy)

    # ---- Catmull–Rom ----
    @staticmethod
    def _catmull(p0, p1, p2, p3, t):
        t2, t3 = t * t, t * t * t
        x = 0.5 * (
            (2 * p1.x()) + (-p0.x() + p2.x()) * t +
            (2 * p0.x() - 5 * p1.x() + 4 * p2.x() - p3.x()) * t2 +
            (-p0.x() + 3 * p1.x() - 3 * p2.x() + p3.x()) * t3
        )
        y = 0.5 * (
            (2 * p1.y()) + (-p0.y() + p2.y()) * t +
            (2 * p0.y() - 5 * p1.y() + 4 * p2.y() - p3.y()) * t2 +
            (-p0.y() + 3 * p1.y() - 3 * p2.y() + p3.y()) * t3
        )
        return QPointF(x, y)

    def _draw_spline_segment(self, p0, p1, p2, p3):
        x_off = (self.width() - self._img.width()) / 2
        y_off = (self.height() - self._img.height()) / 2

        def to_buf(q): return QPointF(q.x() - x_off, q.y() - y_off)
        p0b, p1b, p2b, p3b = map(to_buf, (p0, p1, p2, p3))

        approx_len = (p2b - p1b).manhattanLength()
        steps = max(16, int(approx_len / 0.8))  # densidad alta

        painter = QPainter(self._img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(
            self.pen_color, self.pen_width,
            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
        )
        painter.setPen(pen)
        prev = self._catmull(p0b, p1b, p2b, p3b, 0.0)
        for i in range(1, steps + 1):
            t = i / steps
            cur = self._catmull(p0b, p1b, p2b, p3b, t)
            painter.drawLine(prev, cur)
            prev = cur
        painter.end()
        self.update()

    # ---- Pintado ----
    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pm = QPixmap.fromImage(self._img)
        x = (self.width() - pm.width()) // 2
        y = (self.height() - pm.height()) // 2
        p.drawPixmap(x, y, pm)

    def resizeEvent(self, e):
        super().resizeEvent(e); self.update()

    # ---- Flujo de trazo ----
    def _stroke_begin(self, pos_w: QPointF):
        self._points = [pos_w]
        self._raw_prev = None; self._smooth_prev = None
        self._ma_win.clear()

    def _stroke_add(self, pos_w: QPointF):
        s = self._ema(pos_w)
        s = self._moving_avg(s)

        if len(self._points) == 1:
            if (s - self._points[0]).manhattanLength() < self.min_seg_len:
                return
            self._points.append(s)
            # primer segmento lineal
            x_off = (self.width() - self._img.width()) / 2
            y_off = (self.height() - self._img.height()) / 2
            a, b = self._points[0], self._points[1]
            a = QPointF(a.x() - x_off, a.y() - y_off)
            b = QPointF(b.x() - x_off, b.y() - y_off)
            painter = QPainter(self._img)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(
                self.pen_color, self.pen_width,
                Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin
            )
            painter.setPen(pen); painter.drawLine(a, b); painter.end(); self.update()
            return

        self._points.append(s)
        if len(self._points) >= 4:
            p0, p1, p2, p3 = self._points[-4], self._points[-3], self._points[-2], self._points[-1]
            self._draw_spline_segment(p0, p1, p2, p3)

    def _stroke_end(self):
        self._points.clear()
        self._raw_prev = None; self._smooth_prev = None
        self._ma_win.clear()

    # ---- Pluma / Mouse ----
    def tabletEvent(self, ev: QTabletEvent):
        t = ev.type(); pos = ev.position()
        pressure = ev.pressure() if hasattr(ev, "pressure") else 0.0
        down = self._pressed or (pressure is not None and pressure > self.press_threshold) \
               or bool(ev.buttons() & Qt.MouseButton.LeftButton)

        if t == QTabletEvent.Type.TabletPress:
            self._pressed = True; self._stroke_begin(pos); ev.accept(); return
        if t == QTabletEvent.Type.TabletMove:
            if not down:
                self._pressed = False; self._stroke_end(); ev.accept(); return
            if not self._points:
                self._stroke_begin(pos)
            self._stroke_add(pos); ev.accept(); return
        if t == QTabletEvent.Type.TabletRelease:
            self._pressed = False; self._stroke_end(); ev.accept(); return
        super().tabletEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton: return
        self._stroke_begin(ev.position())

    def mouseMoveEvent(self, ev):
        if not (ev.buttons() & Qt.MouseButton.LeftButton): return
        self._stroke_add(ev.position())

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._stroke_end()

    def clear(self):
        self._img.fill(Qt.GlobalColor.transparent); self.update()

# ---------------------------------------------------------------------
# Diálogo de firma (modal + fullscreen) con recorte por alfa en accept()
# ---------------------------------------------------------------------
class SignaturePad(QDialog):
    def __init__(self, size_px=QSize(2000, 900), pen_width=7, parent=None, fullscreen=True):
        super().__init__(parent)
        self.setWindowTitle("Firmar")
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Lienzo
        self.canvas = SignatureCanvas(size_px=size_px, pen_width=pen_width, parent=self)

        # UI
        btn_clear = QPushButton("Limpiar"); btn_cancel = QPushButton("Cancelar")
        btn_ok = QPushButton("Confirmar"); btn_ok.setDefault(True)

        row = QHBoxLayout(); row.setContentsMargins(16, 8, 16, 16)
        row.addWidget(btn_clear); row.addStretch(); row.addWidget(btn_cancel); row.addWidget(btn_ok)

        lay = QVBoxLayout(self); lay.setContentsMargins(16, 16, 16, 8)
        lay.addWidget(self.canvas, 1); lay.addLayout(row)

        btn_clear.clicked.connect(self.canvas.clear)
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)

        # Atajos
        QShortcut(QKeySequence("Escape"), self, activated=self.reject)
        QShortcut(QKeySequence("Return"), self, activated=self.accept)
        QShortcut(QKeySequence("Enter"), self, activated=self.accept)

        # F11 toggle fullscreen
        def toggle_full():
            if self.windowState() & Qt.WindowState.WindowFullScreen:
                self.setWindowState(Qt.WindowState.WindowNoState)
                self.showMaximized()
            else:
                self.setWindowState(Qt.WindowState.WindowFullScreen)
        QShortcut(QKeySequence("F11"), self, activated=toggle_full)

        self._want_fullscreen = bool(fullscreen)
        self._shown_once = False

        if not self._want_fullscreen:
            self.resize(1200, 700)
            self.showMaximized()

        pal = self.palette()
        pal.setColor(self.backgroundRole(), Qt.GlobalColor.white)
        self.setPalette(pal); self.setAutoFillBackground(True)

        self._png_bytes = None

    def showEvent(self, ev: QEvent):
        super().showEvent(ev)
        if not self._shown_once:
            self._shown_once = True
            if self._want_fullscreen:
                self.setWindowState(Qt.WindowState.WindowFullScreen)

    def result_png_bytes(self): return self._png_bytes

    def accept(self):
        # Guardar la imagen recortada por alfa (para que escale bien al rectángulo)
        from PySide6.QtCore import QBuffer, QByteArray
        qimg = crop_alpha_bbox(self.canvas.image(), padding=16)
        qba = QByteArray(); buf = QBuffer(qba); buf.open(QBuffer.OpenModeFlag.WriteOnly)
        qimg.save(buf, "PNG"); buf.close()
        self._png_bytes = bytes(qba)
        super().accept()

    def reject(self):
        self._png_bytes = None
        super().reject()

# ---------------------------------------------------------------------
# Visor PDF con selección + preview de firma editable (mover/resize/zoom)
# ---------------------------------------------------------------------
class PdfViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc = None; self.page_index = 0
        self.render_scale = 1.0; self.pm_size = QSize(1, 1)

        self.canvas = QLabel()
        self.canvas.setFrameShape(QFrame.Shape.Panel)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setScaledContents(False)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(200, 200)

        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_sign = QPushButton("Firmar aquí")

        top = QHBoxLayout()
        top.addWidget(QLabel("Página:"))
        self.lbl_page = QLabel("-/-"); top.addWidget(self.lbl_page)
        top.addStretch(); top.addWidget(self.btn_prev); top.addWidget(self.btn_next)
        top.addStretch(); top.addWidget(self.btn_sign)

        # Botones de preview edición
        self.btn_apply = QPushButton("Aplicar")
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_apply.setVisible(False); self.btn_cancel.setVisible(False)
        top.addWidget(self.btn_cancel); top.addWidget(self.btn_apply)

        lay = QVBoxLayout(self); lay.addLayout(top); lay.addWidget(self.canvas)

        # Selección
        self.sel_start = None; self.sel_rect = None
        self.canvas.mousePressEvent = self._start_sel
        self.canvas.mouseMoveEvent = self._drag_sel
        self.canvas.mouseReleaseEvent = self._end_sel

        # Navegación / acción
        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)
        self.btn_sign.clicked.connect(self.sign_here)

        # Preview edición
        self.btn_apply.clicked.connect(self.apply_signature)
        self.btn_cancel.clicked.connect(self.cancel_signature)

        self._page_pix = None; self._page_pt_size = (1, 1)
        self._preview_active = False
        self._preview_img = None    # QImage
        self._preview_rect = None   # QRectF en coords de canvas
        self._dragging = False
        self._resizing = None
        self._drag_start = None
        self._orig_rect = None

        # Capturar rueda en modo preview
        self.canvas.wheelEvent = self._wheel_event

    # ---- PDF I/O ----
    def open_pdf(self, path):
        try:
            if self.doc: self.doc.close()
            self.doc = fitz.open(path); self.page_index = 0
            self.render_page()
        except Exception as e:
            QMessageBox.critical(self, "PDF", f"No pude abrir el PDF:\n{e}")

    def _target_rect(self): return self.canvas.contentsRect()

    def render_page(self):
        if not self.doc:
            self.canvas.clear(); self.lbl_page.setText("-/-"); return
        page = self.doc[self.page_index]; self._page_pt_size = (page.rect.width, page.rect.height)

        cr = self._target_rect(); W, H = cr.width(), cr.height()
        if W < 5 or H < 5: return
        s = max(0.1, min(W / self._page_pt_size[0], H / self._page_pt_size[1]))
        self.render_scale = s

        pix = page.get_pixmap(matrix=fitz.Matrix(s, s), alpha=False)
        self.pm_size = QSize(pix.width, pix.height)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
        self._page_pix = QPixmap.fromImage(img.copy())
        self._draw_canvas(); self.lbl_page.setText(f"{self.page_index+1}/{len(self.doc)}")
        self.sel_rect = None
        # Si cambias de página, cancela preview
        self.cancel_signature()

    def _draw_canvas(self):
        cr = self._target_rect()
        canvas_pm = QPixmap(cr.size()); canvas_pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(canvas_pm); p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        x_off = (cr.width() - self.pm_size.width()) // 2
        y_off = (cr.height() - self.pm_size.height()) // 2

        if self._page_pix:
            p.drawPixmap(x_off, y_off, self._page_pix)

        # Selección normal (rect dorado) si no hay preview
        if self.sel_rect and not self.sel_rect.isEmpty() and not self._preview_active:
            pen = QPen(QColor(255, 215, 0), 3); p.setPen(pen)
            pix_rect = QRectF(x_off, y_off, self.pm_size.width(), self.pm_size.height())
            r = self.sel_rect.intersected(pix_rect); p.drawRect(r)

        # Preview de firma (contain + handles)
        if self._preview_active and self._preview_img is not None and self._preview_rect is not None:
            r = QRectF(self._preview_rect)
            img_w = self._preview_img.width()
            img_h = self._preview_img.height()
            if img_w > 0 and img_h > 0:
                sx = r.width() / img_w
                sy = r.height() / img_h
                scale = min(sx, sy)  # contain
                draw_w = img_w * scale
                draw_h = img_h * scale
                draw_x = r.x() + (r.width() - draw_w) / 2.0
                draw_y = r.y() + (r.height() - draw_h) / 2.0

                # Caja de edición
                pen_box = QPen(QColor(255, 215, 0, 220), 2)
                p.setPen(pen_box); p.drawRect(r)

                # Firma
                p.drawImage(QRectF(draw_x, draw_y, draw_w, draw_h), self._preview_img)

                # Handles
                handle = 8.0
                for cx, cy in [(r.left(), r.top()), (r.right(), r.top()),
                               (r.left(), r.bottom()), (r.right(), r.bottom())]:
                    p.fillRect(QRectF(cx - handle/2, cy - handle/2, handle, handle), QColor(255, 215, 0))

        p.end(); self.canvas.setPixmap(canvas_pm)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.doc: self.render_page()

    # ---- Selección & Preview ----
    def _pos_local(self, pos: QPointF):
        cr = self._target_rect()
        return QPointF(pos.x() - cr.x(), pos.y() - cr.y()), cr

    def _hit_corner(self, pos: QPointF, r: QRectF, tol=12):
        corners = {
            'tl': QPointF(r.left(),  r.top()),
            'tr': QPointF(r.right(), r.top()),
            'bl': QPointF(r.left(),  r.bottom()),
            'br': QPointF(r.right(), r.bottom()),
        }
        for name, c in corners.items():
            if abs(pos.x() - c.x()) <= tol and abs(pos.y() - c.y()) <= tol:
                return name
        return None

    def _start_sel(self, ev):
        p, cr = self._pos_local(ev.position())
        if not QRectF(0, 0, cr.width(), cr.height()).contains(p):
            return

        if self._preview_active and self._preview_rect:
            corner = self._hit_corner(p, self._preview_rect)
            if corner:
                self._resizing = corner; self._dragging = False; self._drag_start = p; self._orig_rect = QRectF(self._preview_rect)
            elif self._preview_rect.contains(p):
                self._dragging = True; self._resizing = None; self._drag_start = p; self._orig_rect = QRectF(self._preview_rect)
            self._draw_canvas()
            return

        # Selección normal
        self.sel_start = p; self.sel_rect = QRectF(p, p); self._draw_canvas()

    def _drag_sel(self, ev):
        p, _ = self._pos_local(ev.position())
        if self._preview_active and self._preview_rect:
            cr = self._target_rect()
            pix_rect = QRectF(
                (cr.width() - self.pm_size.width()) / 2,
                (cr.height() - self.pm_size.height()) / 2,
                self.pm_size.width(), self.pm_size.height()
            )
            if self._dragging:
                delta = p - self._drag_start
                r = QRectF(self._orig_rect); r.translate(delta.x(), delta.y())
                r = r.intersected(pix_rect)
                self._preview_rect = r; self._draw_canvas()
                return
            if self._resizing:
                r = QRectF(self._orig_rect)
                dx = p.x() - self._drag_start.x()
                dy = p.y() - self._drag_start.y()
                if self._resizing == 'tl':
                    r.setTop(r.top() + dy); r.setLeft(r.left() + dx)
                elif self._resizing == 'tr':
                    r.setTop(r.top() + dy); r.setRight(r.right() + dx)
                elif self._resizing == 'bl':
                    r.setBottom(r.bottom() + dy); r.setLeft(r.left() + dx)
                elif self._resizing == 'br':
                    r.setBottom(r.bottom() + dy); r.setRight(r.right() + dx)
                if r.width() < 40: r.setRight(r.left() + 40)
                if r.height() < 20: r.setBottom(r.top() + 20)
                r = r.intersected(pix_rect)
                self._preview_rect = r; self._draw_canvas()
                return

        # Selección normal
        if self.sel_start is None: return
        self.sel_rect = QRectF(self.sel_start, p).normalized(); self._draw_canvas()

    def _end_sel(self, ev):
        self._dragging = False; self._resizing = None
        if self.sel_start is None: return
        self._draw_canvas(); self.sel_start = None

    def _wheel_event(self, ev):
        if not (self._preview_active and self._preview_rect):
            return  # aquí podrías implementar zoom de página si quieres
        p, _ = self._pos_local(ev.position())
        if not self._preview_rect.contains(p):
            return
        delta = ev.angleDelta().y()
        factor = 1.0 + (0.0015 * delta)  # ~15% por notch
        r = QRectF(self._preview_rect)
        cx, cy = r.center().x(), r.center().y()
        r.setWidth(r.width() * factor); r.setHeight(r.height() * factor)
        r.moveCenter(QPointF(cx, cy))
        cr = self._target_rect()
        pix_rect = QRectF(
            (cr.width() - self.pm_size.width()) / 2,
            (cr.height() - self.pm_size.height()) / 2,
            self.pm_size.width(), self.pm_size.height()
        )
        r = r.intersected(pix_rect)
        self._preview_rect = r; self._draw_canvas()

    def _rect_to_pdf_points(self):
        if not self.sel_rect or self.sel_rect.isEmpty(): return None
        cr = self._target_rect()
        x_off = (cr.width() - self.pm_size.width()) / 2
        y_off = (cr.height() - self.pm_size.height()) / 2
        r = QRectF(self.sel_rect); r.translate(-x_off, -y_off)
        r = r.intersected(QRectF(0, 0, self.pm_size.width(), self.pm_size.height()))
        if r.isEmpty(): return None
        return (r.left() / self.render_scale, r.top() / self.render_scale,
                r.right() / self.render_scale, r.bottom() / self.render_scale)

    def _canvas_rect_to_pdf_points(self, r: QRectF):
        cr = self._target_rect()
        x_off = (cr.width() - self.pm_size.width()) / 2
        y_off = (cr.height() - self.pm_size.height()) / 2
        r2 = QRectF(r); r2.translate(-x_off, -y_off)
        r2 = r2.intersected(QRectF(0, 0, self.pm_size.width(), self.pm_size.height()))
        if r2.isEmpty(): return None
        return (r2.left() / self.render_scale, r2.top() / self.render_scale,
                r2.right() / self.render_scale, r2.bottom() / self.render_scale)

    # ---- Flujo de firma ----
    def sign_here(self):
        if not self.doc:
            QMessageBox.warning(self, "Firmar", "Abre un PDF primero."); return
        rect_pt = self._rect_to_pdf_points()
        if not rect_pt:
            QMessageBox.warning(self, "Firmar", "Dibuja un rectángulo para la firma."); return

        # Tamaño del lienzo del pad ~ tamaño del rect en puntos a 300 dpi
        left, top, right, bottom = rect_pt
        width_pt = max(1.0, right - left)
        height_pt = max(1.0, bottom - top)
        dpi = 300
        width_px = int(width_pt / 72.0 * dpi)
        height_px = int(height_pt / 72.0 * dpi)
        width_px = max(600, min(width_px, 3000))
        height_px = max(300, min(height_px, 2000))

        pad = SignaturePad(size_px=QSize(width_px, height_px), pen_width=7, parent=self, fullscreen=True)
        if pad.exec() != QDialog.DialogCode.Accepted:
            return
        png_bytes = pad.result_png_bytes()
        if not png_bytes:
            return

        # Cargar la imagen para preview
        img = QImage()
        if not img.loadFromData(png_bytes, "PNG"):
            QMessageBox.critical(self, "Firma", "No se pudo leer la imagen de la firma.")
            return

        if not self.sel_rect or self.sel_rect.isEmpty():
            QMessageBox.warning(self, "Firma", "Selecciona el área objetivo en la página.")
            return

        self._preview_img = img
        self._preview_rect = QRectF(self.sel_rect)  # inicia sobre el rect elegido
        self._preview_active = True
        self.btn_apply.setVisible(True); self.btn_cancel.setVisible(True)
        self._draw_canvas()

    def apply_signature(self):
        if not (self._preview_active and self._preview_img is not None and self._preview_rect is not None):
            return
        rect_pt = self._canvas_rect_to_pdf_points(self._preview_rect)
        if not rect_pt:
            QMessageBox.warning(self, "Firmar", "La firma quedó fuera de la página."); return

        from PySide6.QtCore import QBuffer, QByteArray
        qba = QByteArray(); buf = QBuffer(qba); buf.open(QBuffer.OpenModeFlag.WriteOnly)
        self._preview_img.save(buf, "PNG"); buf.close()
        png_bytes = bytes(qba)

        try:
            out = insert_signature_png(self.doc.name, self.page_index, rect_pt, png_bytes)
            QMessageBox.information(self, "Firmar", f"PDF firmado guardado:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Firmar", f"Error al insertar firma:\n{e}")
        finally:
            self.cancel_signature()  # limpiar preview

    def cancel_signature(self):
        self._preview_active = False
        self._preview_img = None
        self._preview_rect = None
        self.btn_apply.setVisible(False); self.btn_cancel.setVisible(False)
        self._draw_canvas()

    # ---- Navegación ----
    def prev_page(self):
        if not self.doc: return
        if self.page_index > 0: self.page_index -= 1; self.render_page()

    def next_page(self):
        if not self.doc: return
        if self.page_index < len(self.doc) - 1: self.page_index += 1; self.render_page()

# ---------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QuickSign PDF")
        try:
            self.setWindowIcon(QIcon(pkg_asset("assets/icon.ico")))
        except Exception:
            pass

        self.resize(1200, 850)
        self.viewer = PdfViewer()
        self.setCentralWidget(self.viewer)

        open_act = QAction("Abrir PDF…", self); open_act.triggered.connect(self.open_pdf)
        self.menuBar().addAction(open_act)

        help_act = QAction("Ayuda (F1)", self); help_act.triggered.connect(self.show_help)
        self.menuBar().addAction(help_act)

        self.statusBar()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(url.toLocalFile().lower().endswith(".pdf") for url in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".pdf"):
                self.viewer.open_pdf(path); break

    def show_help(self):
        QMessageBox.information(
            self, "Atajos",
            "Esc: cancelar\nEnter: confirmar\nF11: pantalla completa (pad)\n"
            "Preview: arrastra para mover, esquinas para redimensionar, rueda para zoom.\n\n"
            "Nota: esto es una marca visual de firma (no firma digital PAdES)."
        )

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Elegir PDF", "", "PDF (*.pdf)")
        if path: self.viewer.open_pdf(path)

# ---------------------------------------------------------------------
# Arranque
# ---------------------------------------------------------------------
def main():
    try:
        import fitz  # asegurar import
    except Exception as e:
        print("ERROR importando PyMuPDF (fitz):", e)
        print("Instala con: pip install pymupdf")
        raise

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("QuickSign PDF")
        app.setOrganizationName("QuickSign")
        app.setApplicationDisplayName("QuickSign PDF")

        w = MainWindow()
        w.show()
        sys.exit(app.exec())
    except Exception:
        tb = traceback.format_exc()
        try:
            QMessageBox.critical(None, "Error crítico", tb)
        except Exception:
            pass
        with open("firmar_error.txt", "w", encoding="utf-8") as f:
            f.write(tb)
        print(tb)
        input("Hubo un error. Revisa firmar_error.txt. Pulsa Enter para salir...")

if __name__ == "__main__":
    main()
