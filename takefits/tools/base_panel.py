"""
Base class for tool panels with common functionality.

This module provides BaseToolPanel which extracts common patterns from all tool panels:
- Constructor with fits_viewer and subwindows
- Default window positioning
- AppState access
- Display update helpers
- Close event handling
"""
from PySide6.QtWidgets import QWidget, QMessageBox, QApplication
from PySide6.QtCore import Qt


def confirm_pending_close(
    parent,
    title: str,
    text: str,
    *,
    keep_label: str = "Keep and Close",
    discard_label: str | None = "Discard and Close",
) -> str:
    """
    Show a standard pending-changes dialog.

    Returns one of: "keep", "discard", "cancel".
    """
    app = QApplication.instance()
    # Do not block application shutdown with per-panel confirmation dialogs.
    if app is not None:
        try:
            if bool(app.property("takefits_app_closing")):
                return "keep"
        except Exception:
            pass
        try:
            if app.closingDown():
                return "keep"
        except Exception:
            pass

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    keep_btn = box.addButton(keep_label, QMessageBox.ButtonRole.AcceptRole)
    discard_btn = None
    if discard_label:
        discard_btn = box.addButton(discard_label, QMessageBox.ButtonRole.DestructiveRole)
    cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(keep_btn)
    box.exec()
    clicked = box.clickedButton()
    if clicked == keep_btn:
        return "keep"
    if discard_btn is not None and clicked == discard_btn:
        return "discard"
    if clicked == cancel_btn:
        return "cancel"
    return "cancel"


def _resolve_main_window(fits_viewer):
    return getattr(fits_viewer, "main_window", None) or fits_viewer


def action_history_records_up_to_cursor(fits_viewer):
    """Return action records up to the current cursor."""
    main_window = _resolve_main_window(fits_viewer)
    session = getattr(main_window, "action_session", None)
    if session is None:
        return []
    try:
        history = list(getattr(session, "history", []) or [])
    except Exception:
        return []
    try:
        cursor = int(getattr(session, "cursor", len(history)))
    except Exception:
        cursor = len(history)
    cursor = max(0, min(cursor, len(history)))
    return history[:cursor]


def has_action_record_tag(
    fits_viewer,
    tag: str,
    *,
    prefix: bool = False,
) -> bool:
    """Return True when action history contains a record tag."""
    target = str(tag or "").strip()
    if not target:
        return False
    for record in reversed(action_history_records_up_to_cursor(fits_viewer)):
        record_tag = str(getattr(record, "tag", "") or "").strip()
        if not record_tag:
            continue
        if prefix:
            if record_tag.startswith(target):
                return True
        elif record_tag == target:
            return True
    return False


def record_action_preview(
    fits_viewer,
    action_name: str,
    params: dict,
    *,
    replace_tag: str | None = None,
) -> bool:
    """
    Record a preview/temporary action into ActionSession history without re-executing.
    """
    main_window = _resolve_main_window(fits_viewer)
    recorder = getattr(main_window, "record_action", None)
    if callable(recorder):
        recorder(action_name, params=params, replace_tag=replace_tag)
        return True
    session = getattr(main_window, "action_session", None)
    if session is not None and hasattr(session, "record"):
        session.record(action_name, params=params, replace_tag=replace_tag)
        return True
    return False


def clear_action_preview_record(
    fits_viewer,
    replace_tag: str,
    *,
    action_name: str | None = None,
) -> bool:
    """
    Remove previously recorded preview actions.

    Priority:
    1) remove by replace_tag (current format)
    2) if nothing removed and action_name is provided, remove all matching
       action records (legacy entries recorded without tags)
    """
    main_window = _resolve_main_window(fits_viewer)
    session = getattr(main_window, "action_session", None)
    if session is None:
        clearer = getattr(main_window, "clear_recorded_action", None)
        if callable(clearer) and replace_tag:
            clearer(replace_tag=replace_tag)
            return True
        return False

    removed = False
    if replace_tag and hasattr(session, "remove_record_by_tag"):
        removed = bool(session.remove_record_by_tag(replace_tag))

    if (not removed) and action_name:
        try:
            history = list(getattr(session, "history", []) or [])
            cursor = int(getattr(session, "cursor", len(history)))
        except Exception:
            history = list(getattr(session, "history", []) or [])
            cursor = len(history)
        action_token = str(action_name or "").strip().lower()
        keep = []
        removed_before_cursor = 0
        for idx, record in enumerate(history):
            token = str(getattr(record, "action", "") or "").strip().lower()
            if token == action_token:
                removed = True
                if idx < cursor:
                    removed_before_cursor += 1
                continue
            keep.append(record)
        if removed:
            session.history = keep
            session._cursor = max(0, cursor - removed_before_cursor)

    if removed:
        refresher = getattr(main_window, "_refresh_undo_redo_actions", None)
        if callable(refresher):
            try:
                refresher()
            except Exception:
                pass
    return removed


def capture_preferred_cursor_snapshot(fits_viewer):
    """
    Capture the current shared cursor snapshot for replay-based reset.
    """
    main_window = _resolve_main_window(fits_viewer)
    capture = getattr(main_window, "_capture_shared_cursor_snapshot", None)
    if callable(capture):
        try:
            snapshot = capture()
            if isinstance(snapshot, dict) and snapshot:
                return dict(snapshot)
        except Exception:
            pass
    index_getter = getattr(fits_viewer, "current_channel_index", None)
    if callable(index_getter):
        try:
            return {"zpix": int(index_getter())}
        except Exception:
            pass
    return None


def replay_action_history_to_current_cursor(fits_viewer, *, preferred_cursor=None) -> bool:
    """
    Replay ActionSession to its current cursor and apply the state to viewers.
    """
    main_window = _resolve_main_window(fits_viewer)
    session = getattr(main_window, "action_session", None)
    if session is None:
        return False
    replay_to_cursor = getattr(session, "_replay_to_cursor", None)
    apply_state = getattr(main_window, "_apply_action_session_state_to_viewers", None)
    if not callable(replay_to_cursor) or not callable(apply_state):
        return False
    try:
        history = list(getattr(session, "history", []) or [])
        cursor = int(getattr(session, "cursor", len(history)))
        cursor = max(0, min(cursor, len(history)))
    except Exception:
        return False
    try:
        replay_to_cursor(cursor)
        if isinstance(preferred_cursor, dict) and preferred_cursor:
            apply_state(preferred_cursor=dict(preferred_cursor))
        else:
            apply_state()
        return True
    except Exception:
        return False


class BaseToolPanel(QWidget):
    """
    Base class for tool panels with common functionality.
    
    Subclasses should:
    1. Call super().__init__(fits_viewer, subwindows) first
    2. Override initUI() to create widgets
    3. Optionally override move_to_default_position() for custom placement
    """
    
    def __init__(self, fits_viewer, subwindows=None):
        """
        Initialize the tool panel.
        
        Args:
            fits_viewer: Main FITSViewer instance
            subwindows: List of subwindow viewers (optional)
        """
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.fits_viewer = fits_viewer
        self.subwindows = subwindows if subwindows is not None else []
        self._setup_panel()
    
    def _setup_panel(self):
        """Template method for panel setup. Called from __init__."""
        self.initUI()
        self.move_to_default_position()
    
    def initUI(self):
        """
        Create the UI for this panel. Override in subclass.
        
        Raises:
            NotImplementedError: Must be overridden by subclass
        """
        raise NotImplementedError("Subclass must implement initUI()")
    
    def move_to_default_position(self):
        """
        Position panel next to the control panel or main window.
        
        Default behavior: Position to the right of the control panel.
        Override in subclass for custom positioning.
        """
        if hasattr(self.fits_viewer, 'control_panel') and self.fits_viewer.control_panel:
            cp_geom = self.fits_viewer.control_panel.geometry()
            self.move(cp_geom.x() + cp_geom.width(), cp_geom.y())
        else:
            main_window = self.get_main_window()
            if main_window:
                geo = main_window.geometry()
                self.move(geo.right() + 10, geo.top())
    
    def get_main_window(self):
        """
        Get reference to the main window.
        
        Returns:
            MainWindow instance or None
        """
        return getattr(self.fits_viewer, 'main_window', None)

    # ------------------------------------------------------------------
    # Pending-close hooks
    def has_pending_changes(self) -> bool:
        """Override in subclasses when close should prompt for pending changes."""
        return False

    def pending_close_title(self) -> str:
        return f"Close {self.windowTitle() or 'Panel'}"

    def pending_close_text(self) -> str:
        return "There are unapplied changes."

    def pending_close_keep_label(self) -> str:
        return "Keep and Close"

    def pending_close_discard_label(self) -> str | None:
        return "Discard and Close"

    def discard_pending_changes(self) -> None:
        """Override in subclasses to rollback panel-local changes."""
        return

    def on_keep_pending_changes(self) -> None:
        """Override in subclasses if keep-action requires cleanup."""
        return

    def resync_after_workspace_restore(self) -> None:
        """Optional hook to rebuild panel-local state after workspace load."""
        return
    
    def get_app_state(self):
        """
        Get reference to the AppState for headless operations.
        
        Returns:
            AppState instance or None
        """
        main = self.get_main_window()
        return getattr(main, 'app_state', None)
    
    def update_all_displays(self):
        """
        Refresh all displays (main viewer and subwindows).
        
        Default behavior: Call canvas.draw() on each window.
        Override for custom update logic.
        """
        all_windows = [self.fits_viewer] + list(self.subwindows)
        for window in all_windows:
            if window and hasattr(window, 'canvas'):
                try:
                    window.canvas.draw()
                except Exception:
                    pass
    
    def closeEvent(self, event):
        """
        Handle window close event.
        
        Emits destroyed signal and calls parent close.
        """
        if self.has_pending_changes():
            choice = confirm_pending_close(
                self,
                self.pending_close_title(),
                self.pending_close_text(),
                keep_label=self.pending_close_keep_label(),
                discard_label=self.pending_close_discard_label(),
            )
            if choice == "cancel":
                event.ignore()
                return
            if choice == "discard":
                self.discard_pending_changes()
            else:
                self.on_keep_pending_changes()
        super().closeEvent(event)
