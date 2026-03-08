# core/ai_handler.py

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QTimer # Import QTimer for delayed execution
import json
import traceback
from matplotlib import colormaps

from takefits.core.color import ColorMode
from takefits.tools.color_scale import ColorSettingsPanel  # Only import the panel class, not the enum
# Import necessary window types if accessing their panels
from takefits.tools.integration import IntegResultWindow # Assuming this holds the panel
from takefits.tools.channel_map import ChannelMapWindow # Assuming this holds the panel


# API Key Setup is handled in control_panel.py

class AIHandler:
    """
    Handles interaction with the OpenAI API for interpreting user commands
    related to FITS image display settings (primarily color adjustments).
    It receives user input, sends it to the AI for interpretation, parses the
    structured JSON response, validates it, and dispatches the corresponding
    actions by calling methods on the appropriate UI panels (like ColorSettingsPanel).
    If the target panel (MAIN, Integ, Channel) is not open, it attempts to open it.
    """
    def __init__(self, client, control_panel):
        """
        Initializes the AIHandler.

        Args:
            client: The initialized OpenAI client instance. If None, AI features are disabled.
            control_panel: Reference to the main ControlPanel instance to access UI elements
                           and other application parts (like ColorSettingsPanel).
        """
        self.client = client
        self.control_panel = control_panel # Keep a reference to the control panel
        self.command_queue = []
        self.current_command_index = 0

    def get_command_interpretation(self, user_command):
        """
        Sends the user command to the configured OpenAI model via the client
        and attempts to parse the response into a structured JSON format
        representing the user's intent(s) and parameters. Includes target panel.
        """
        if self.client is None:
            QMessageBox.critical(None, "API Error", "OpenAI client not initialized. AI features disabled.\nPlease check API key setup and 'openai' library installation.")
            return None

        # --- Prompt Definition (Updated for Gamma and Target) ---
        prompt = f"""
        You are an AI assistant for a FITS data analysis GUI (takefits). Your task is to interpret user commands for adjusting image display settings.
        **It is absolutely critical that you identify ALL distinct intents within a single user command.** Commands often contain multiple actions.

        **Instructions:**
        1.  Analyze the user command carefully.
        2.  Identify every distinct action requested (e.g., setting color scale, changing colormap, setting gamma, inverting colormap, log scale).
        3.  Determine the target window/panel if specified (e.g., "main window", "channel map", "integration plot", "moment 1 plot"). Default to "main" if unspecified. Map "integration plot", "moment 1 plot", "moment 2 plot", "peak intensity plot" etc. to target "integ". Map "channel map" to target "channel". Map "main window" or unspecified to target "main".
        4.  For each action, determine the corresponding intent and extract its required parameters, including the target.
        5.  **Format the output STRICTLY as follows:**
            * If **only one** intent is found, return a **single JSON object**: `{{"intent": "...", "params": {{"target": "...", ...}}}}`
            * If **two or more** intents are found, return a **JSON array**: `[{{"intent": "...", "params": {{"target": "...", ...}}}}, ...]`
            * **DO NOT** add any text before or after the JSON object or array.

        **Possible Intents & Parameters:**
        * `set_colorscale`: Adjust color scale limits. Requires `vmin` (float/int), `vmax` (float/int). Optional: `target` (string: "main", "integ", "channel", default "main"). Use null for vmin/vmax for auto-scaling.
        * `set_colormap`: Change color map. Requires `colormap_name` (string). Optional: `target` (string, default "main"). Do not include "_r".
        * `invert_colormap`: Invert current colormap. Optional: `target` (string, default "main").
        * `log_scale_on`: Enable log scaling. Optional: `target` (string, default "main").
        * `log_scale_off`: Disable log scaling. Optional: `target` (string, default "main").
        * `set_gamma`: Adjust gamma correction. Requires `gamma_value` (float). Optional: `target` (string, default "main").
        * `unknown`: If the command is unclear or unrelated.

        **Examples:**

        * User: `set color scale 0 to 10 for main window`
            AI: `{{"intent": "set_colorscale", "params": {{"target": "main", "vmin": 0, "vmax": 10}}}}`
        * User: `チャンネルマップのカラーマップをviridisに反転して`
            AI: `[
                 {{"intent": "set_colormap", "params": {{"target": "channel", "colormap_name": "viridis"}}}},
                 {{"intent": "invert_colormap", "params": {{"target": "channel"}}}}
                ]`
        * User: `ガンマ値を1.5にして`
            AI: `{{"intent": "set_gamma", "params": {{"target": "main", "gamma_value": 1.5}}}}`
        * User: `積分強度図のカラースケールを0から50に、ログスケールをオフ`
            AI: `[
                 {{"intent": "set_colorscale", "params": {{"target": "integ", "vmin": 0, "vmax": 50}}}},
                 {{"intent": "log_scale_off", "params": {{"target": "integ"}}}}
                ]`
        * User: `Set gamma to 2 on the moment 1 plot`
            AI: `{{"intent": "set_gamma", "params": {{"target": "integ", "gamma_value": 2.0}}}}`


        **User command:** "{user_command}"
        **AI:**
        """
        # --- End Prompt Definition ---

        try:
            # --- Call OpenAI API ---
            response = self.client.chat.completions.create(
                model="gpt-4-turbo", # Recommend GPT-4 or newer
                messages=[
                    {"role": "system", "content": "You are an AI assistant interpreting commands for FITS image display adjustments. Identify all intents, targets (main, integ, channel - default main), and parameters. Return JSON or a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=400 # Increased slightly
            )
            result_text = response.choices[0].message.content.strip()
            # --- End API Call ---

            print(f"AI Prompt received: {user_command}")
            print(f"OpenAI Raw Response Content: {result_text}")

            # --- JSON Parsing and Basic Validation ---
            try:
                # Remove potential markdown formatting
                if result_text.startswith("```json"): result_text = result_text[7:]
                if result_text.endswith("```"): result_text = result_text[:-3]
                result_text = result_text.strip()
                parsed_result = json.loads(result_text)
                print(f"OpenAI Parsed JSON: {parsed_result}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                QMessageBox.critical(None, "API Error", f"Could not parse JSON response:\n{result_text[:100]}...")
                return {"intent": "unknown", "params": {"reason": "Invalid JSON format from AI"}}
            # --- End JSON Parsing ---

            # --- Content Structure Validation & Target Defaulting/Validation ---
            results_to_process = []
            if isinstance(parsed_result, dict): results_to_process = [parsed_result]
            elif isinstance(parsed_result, list): results_to_process = parsed_result
            else:
                print(f"Warning: OpenAI response is not dict or list: {type(parsed_result)}")
                return {"intent": "unknown", "params": {"reason": "Invalid response format (not dict or list)"}}

            valid_structure = True
            for item in results_to_process:
                if not isinstance(item, dict) or "intent" not in item or "params" not in item:
                    print(f"Warning: Invalid item structure: {item}")
                    valid_structure = False; break
                # Add Default Target if missing
                if "target" not in item["params"]:
                    item["params"]["target"] = "main"
                    print(f"Defaulting target to 'main' for intent '{item['intent']}'")
                # Validate Target Value
                elif item["params"]["target"] not in ["main", "integ", "channel"]:
                    print(f"Warning: Invalid target '{item['params']['target']}'. Defaulting to 'main'.")
                    item["params"]["target"] = "main"


            if not valid_structure:
                 return [{"intent": "unknown", "params": {"reason": "Invalid item structure"}}][0] if isinstance(parsed_result, list) else {"intent": "unknown", "params": {"reason": "Invalid JSON structure"}}

            # --- Colormap Name Validation (already included target) ---
            available_maps = {name.lower() for name in colormaps}
            for item in results_to_process:
                if item.get("intent") == "set_colormap":
                    cmap_name = item.get("params", {}).get("colormap_name")
                    if cmap_name and cmap_name.lower() not in available_maps:
                        print(f"Warning: Invalid colormap '{cmap_name}'. Treating as unknown.")
                        item["intent"] = "unknown"; item["params"]["reason"] = f"Invalid colormap: {cmap_name}"
                    elif cmap_name is None:
                         print("Warning: Missing colormap_name. Treating as unknown.")
                         item["intent"] = "unknown"; item["params"]["reason"] = "Missing colormap_name"
            # --- End Colormap Name Validation ---

            return parsed_result # Return validated dict or list

        except Exception as e: # Catch other errors
            print(f"Error during AI interpretation or validation: {e}")
            print(f"Detailed error: {traceback.format_exc()}")
            error_message = f"Could not contact AI assistant or process response: {e}"
            if "authentication" in str(e).lower(): error_message = "OpenAI API key invalid/missing."
            QMessageBox.critical(None,"API Error", error_message)
            return None

    def _get_target_panel(self, target_str):
        """
        Finds the appropriate ColorSettingsPanel instance and its parent window
        based on the target string.
        Returns the panel instance, the window instance, and its corresponding ColorMode.
        Panel or window can be None if not found.
        """
        target_panel = None
        target_window = None
        mode = ColorMode.MAIN # Default

        if target_str == "main":
            mode = ColorMode.MAIN
            # Get the panel instance from the control_panel
            target_panel = self.control_panel.color_settings_panel
            # The 'window' for the main panel is the main FITSViewer itself
            target_window = self.control_panel.fits_viewer
            if target_panel and target_panel.mode == mode:
                print("Target panel found: MAIN")
            else:
                # Panel might exist but is for a different mode (unlikely here) or doesn't exist yet
                print("MAIN ColorSettingsPanel not found or not in MAIN mode.")
                target_panel = None
            return target_panel, target_window, mode

        elif target_str == "integ":
            mode = ColorMode.INTEG
            integ_windows = []
            if hasattr(self.control_panel, 'integ_settings_panel') and \
               self.control_panel.integ_settings_panel and \
               hasattr(self.control_panel.integ_settings_panel, 'integ_result_windows'):
                 integ_windows = self.control_panel.integ_settings_panel.integ_result_windows
                 print(f"Found {len(integ_windows)} integ result windows tracked.")
            else:
                 print("Could not find list of integ result windows.")

            for win in integ_windows:
                if win and not win.isHidden():
                    target_window = win
                    print(f"Target window found: {type(target_window).__name__} (visible)")
                    if hasattr(win, 'color_settings_panel') and win.color_settings_panel:
                        if win.color_settings_panel.mode == mode:
                            target_panel = win.color_settings_panel
                            print("Target panel found: INTEG (visible and correct mode)")
                        else:
                            print(f"Warning: Found INTEG window {win}, but its panel has mode {win.color_settings_panel.mode}. Skipping.")
                            target_panel = None # Panel exists but wrong mode
                    else:
                        print(f"Found INTEG window {win}, but no color panel attached yet.")
                        target_panel = None # Panel doesn't exist yet
                    break # Found the first visible window

            if target_window is None: print("No visible INTEG window found.")
            # Return panel (None if not found/wrong mode), window (None if no visible window), mode
            return target_panel, target_window, mode

        elif target_str == "channel":
            mode = ColorMode.CHANNEL
            channel_map_windows = []
            if hasattr(self.control_panel, 'chmap_settings_panel') and \
               self.control_panel.chmap_settings_panel and \
               hasattr(self.control_panel.chmap_settings_panel, 'channel_result_windows'):
                 channel_map_windows = self.control_panel.chmap_settings_panel.channel_result_windows
                 print(f"Found {len(channel_map_windows)} channel map windows tracked.")
            else:
                 print("Could not find list of channel map windows.")

            for win in channel_map_windows:
                 if win and not win.isHidden():
                     target_window = win
                     print(f"Target window found: {type(target_window).__name__} (visible)")
                     if hasattr(win, 'color_settings_panel') and win.color_settings_panel:
                         if win.color_settings_panel.mode == mode:
                              target_panel = win.color_settings_panel
                              print("Target panel found: CHANNEL (visible and correct mode)")
                         else:
                              print(f"Warning: Found CHANNEL window {win}, but its panel has mode {win.color_settings_panel.mode}. Skipping.")
                              target_panel = None
                     else:
                          print(f"Found CHANNEL window {win}, but no color panel attached yet.")
                          target_panel = None
                     break

            if target_window is None: print("No visible CHANNEL window found.")
            return target_panel, target_window, mode

        else:
            print(f"Unknown target string: {target_str}")
            return None, None, ColorMode.MAIN

    def dispatch_ai_command(self, interpretation):
        """Initiates the processing of interpreted AI commands via a queue."""
        if interpretation is None: print("Dispatch error: Interpretation is None."); return
        commands_to_run = interpretation if isinstance(interpretation, list) else [interpretation]
        if isinstance(interpretation, list): print(f"Dispatching {len(commands_to_run)} AI commands")
        else: print("Dispatching single AI command.")

        self.command_queue = commands_to_run
        self.current_command_index = 0
        self._process_next_command() # Start queue processing

    def _process_next_command(self):
        """Processes the next command, handles panel finding and opening."""
        if self.current_command_index >= len(self.command_queue):
            print("All AI commands processed."); return

        command_dict = self.command_queue[self.current_command_index]
        print(f"Processing command {self.current_command_index + 1}/{len(self.command_queue)}: {command_dict}")

        intent = command_dict.get("intent", "unknown")
        params = command_dict.get("params", {})
        target_str = params.get("target", "main")

        target_panel, target_window, target_mode = self._get_target_panel(target_str)

        panel_dependent_intents = ["set_colorscale", "set_colormap", "invert_colormap", "log_scale_on", "log_scale_off", "set_gamma"]
        needs_panel_open = False
        can_auto_open = False

        if intent in panel_dependent_intents:
            if target_window is None:
                # Window doesn't exist or isn't visible (for integ/channel)
                # target_window should always exist for main, so this implies integ/channel window not found/visible
                QMessageBox.warning(None, "AI Command", f"Target window '{target_str}' is not open or not found. Command '{intent}' skipped.")
                self.current_command_index += 1
                QTimer.singleShot(10, self._process_next_command)
                return
            elif target_panel is None or target_panel.isHidden():
                needs_panel_open = True
                # Check if panel can be opened
                if target_mode == ColorMode.MAIN:
                    # MAIN panel opened via ControlPanel
                    can_auto_open = hasattr(self.control_panel, 'open_color_settings') and callable(getattr(self.control_panel, 'open_color_settings', None))
                else:
                    # integ/channel panels opened via their respective windows
                    can_auto_open = hasattr(target_window, 'open_color_settings') and callable(getattr(target_window, 'open_color_settings', None))

        # --- Attempt to Open Panel if Needed ---
        if needs_panel_open:
            if can_auto_open:
                print(f"'{target_str.upper()}' Panel needed for '{intent}', attempting to open...")
                # Call the appropriate open method
                if target_mode == ColorMode.MAIN:
                    self.control_panel.open_color_settings()
                else: # Integ or Channel
                    target_window.open_color_settings()

                QTimer.singleShot(150, self._process_current_command_after_delay) # Wait and retry
                return # Stop processing here, wait for the timer
            else:
                # This case should be rare if the window exists but cannot open its panel
                QMessageBox.warning(None, "AI Command", f"Cannot automatically open '{target_str.upper()}' Color Settings panel for '{intent}'. Command skipped.")
                self.current_command_index += 1
                QTimer.singleShot(10, self._process_next_command)
                return

        # --- Execute Command Immediately (Panel ready or not needed) ---
        # Ensure target_panel is passed even if it was None initially (for non-panel-dependent commands)
        success = self.dispatch_single_command(command_dict, target_panel, target_mode)
        if not success:
             QMessageBox.warning(None, "Command Failed", f"Command {self.current_command_index + 1} ('{intent}' on '{target_str}') failed or was skipped.")

        self.current_command_index += 1
        QTimer.singleShot(10, self._process_next_command) # Process next command

    def _process_current_command_after_delay(self):
        """Re-attempts the current command after waiting for a panel to open."""
        if self.current_command_index >= len(self.command_queue): return

        command_dict = self.command_queue[self.current_command_index]
        intent = command_dict.get("intent", "unknown")
        target_str = command_dict.get("params", {}).get("target", "main")

        # Re-fetch the panel, window, and mode
        target_panel, target_window, target_mode = self._get_target_panel(target_str)

        # Check if panel is now open and visible *and* belongs to the correct mode
        if target_panel is None or target_panel.isHidden() or target_panel.mode != target_mode:
            QMessageBox.warning(None, "AI Command Error", f"Failed to open or find correct '{target_str.upper()}' Color Settings panel for '{intent}'. Command skipped.")
            success = False
        else:
            print(f"'{target_str.upper()}' Panel is open, now executing command {self.current_command_index + 1}: {command_dict}")
            success = self.dispatch_single_command(command_dict, target_panel, target_mode) # Execute
            if not success:
                 QMessageBox.warning(None, "Command Failed", f"Command {self.current_command_index + 1} ('{intent}' on '{target_str}') failed after panel open attempt.")

        self.current_command_index += 1
        QTimer.singleShot(10, self._process_next_command) # Move to the next


    def dispatch_single_command(self, command_dict, target_panel, target_mode):
        """Processes an individual command for the specified target panel/mode."""
        if not isinstance(command_dict, dict): return False
        intent = command_dict.get("intent", "unknown")
        params = command_dict.get("params", {}) # Params already include 'target'
        print(f"Executing: {intent} on {target_mode.value}, Params: {params}")

        handler_map = {
            "set_colorscale": self.handle_set_colorscale,
            "set_colormap": self.handle_set_colormap,
            "invert_colormap": self.handle_invert_colormap,
            "log_scale_on": lambda p, panel, mode: self.handle_log_scale(p, panel, mode, enable=True),
            "log_scale_off": lambda p, panel, mode: self.handle_log_scale(p, panel, mode, enable=False),
            "set_gamma": self.handle_set_gamma,
        }
        handler = handler_map.get(intent)

        if handler:
             # Check if panel is required and available *before* calling handler
             panel_dependent = intent in ["set_colorscale", "set_colormap", "invert_colormap", "log_scale_on", "log_scale_off", "set_gamma"]
             if panel_dependent and target_panel is None:
                  # This case should ideally be caught earlier during the opening logic
                  print(f"Critical Error: Target panel for mode '{target_mode.value}' is None when calling panel-dependent handler for '{intent}'.")
                  return False # Command cannot proceed without the panel

             try:
                 # Call the handler, passing the panel and mode
                 if intent in ["set_colorscale", "set_colormap", "invert_colormap", "log_scale_on", "log_scale_off", "set_gamma"]:
                     success = handler(params, target_panel, target_mode)
                 else:
                     # Handle commands that might not need the panel (if any added later)
                     success = handler(params)

                 return success if success is not None else True # Assume success if handler returns None
             except Exception as e:
                  print(f"Error executing handler for intent '{intent}': {e}\n{traceback.format_exc()}")
                  QMessageBox.critical(None, "Execution Error", f"Error executing '{intent}':\n{e}")
                  return False
        elif intent == "unknown":
            reason = params.get('reason', "Could not understand the command")
            QMessageBox.warning(None, "AI Command Error", f"Command not understood.\nReason: {reason}")
            return False
        else:
            QMessageBox.warning(None, "Not Implemented", f"AI intent '{intent}' is recognized but not implemented yet.")
            return False


    # --- Handler Implementations ---
    # (Handlers remain the same as they receive the correct panel)
    # ... (handle_set_colorscale, handle_set_colormap, etc. remain the same) ...
    def handle_set_colorscale(self, params, panel, mode):
        """Handler for 'set_colorscale'. Operates on the provided panel."""
        vmin_in = params.get('vmin'); vmax_in = params.get('vmax')
        target_str = params.get('target', mode.value) # Use mode value if target missing
        print(f"Handling set_colorscale on '{target_str}'...")
        if panel is None: print("Error: Panel is None in handle_set_colorscale."); return False

        try:
            # If both vmin and vmax are None, trigger auto-scaling
            if vmin_in is None and vmax_in is None:
                # Assuming the panel has a method like set_min_max or auto_intensity
                if hasattr(panel, 'set_min_max'):
                    panel.set_min_max()
                    print(f"Auto-scaling triggered for '{target_str}'.")
                    return True
                elif hasattr(panel, 'auto_intensity'):
                    panel.auto_intensity()
                    print(f"Auto-scaling triggered for '{target_str}'.")
                    return True
                else:
                    print(f"Error: No auto-scale method found on panel for '{target_str}'.")
                    return False
            else:
                # Handle specific vmin/vmax values
                vmin_str = f"{float(vmin_in):.3g}" if vmin_in is not None else ""
                vmax_str = f"{float(vmax_in):.3g}" if vmax_in is not None else ""
                # Basic validation: vmin < vmax
                if vmin_in is not None and vmax_in is not None and float(vmin_in) >= float(vmax_in):
                    QMessageBox.warning(None, "Input Error", f"vmin ({vmin_in}) must be less than vmax ({vmax_in}) for '{target_str}'.")
                    return False
                # Update UI fields and trigger the update logic in the panel
                panel.intensity_min.setText(vmin_str)
                panel.intensity_max.setText(vmax_str)
                panel.update_intensity_range() # This method should apply the changes
                print(f"Color scale set to {vmin_str} - {vmax_str} for '{target_str}'.")
                return True
        except (ValueError, TypeError) as e:
            QMessageBox.critical(None, "Parameter Error", f"Invalid number format for vmin/vmax for '{target_str}': {e}")
            return False
        except Exception as e:
            print(f"Unexpected error in handle_set_colorscale for '{target_str}': {e}\n{traceback.format_exc()}")
            return False


    def handle_set_colormap(self, params, panel, mode):
        """Handler for 'set_colormap'. Operates on the provided panel."""
        cmap_name = params.get('colormap_name'); target_str = params.get('target', mode.value)
        if not cmap_name:
            QMessageBox.warning(None, "Parameter Error", f"Colormap name missing for '{target_str}'.")
            return False
        print(f"Handling set_colormap to '{cmap_name}' on '{target_str}'...")
        if panel is None: print("Error: Panel is None in handle_set_colormap."); return False

        base_cmap_name = cmap_name.replace('_r', '') # Ensure we use the base name for the combo box
        try:
            # Check if the colormap exists in the combo box (case-insensitive check might be better)
            items = [panel.colorscale_combo.itemText(i).lower() for i in range(panel.colorscale_combo.count())]
            if base_cmap_name.lower() not in items:
                print(f"Warning: Colormap '{base_cmap_name}' not found in UI list for '{target_str}'. Attempting to set anyway.")
                # Optionally, add it dynamically if desired, or just proceed

            # Set the combo box text, which should trigger its signal if connected correctly
            panel.colorscale_combo.setCurrentText(base_cmap_name)

            # Explicitly call the panel's method to apply the change, in case the signal doesn't cover everything
            if hasattr(panel, 'change_color_scale'):
                panel.change_color_scale()
                print(f"Colormap set and applied for '{target_str}'.")
                return True
            else:
                print(f"Warning: Panel for '{target_str}' does not have 'change_color_scale' method.")
                # Even if the method is missing, setting the combo box might have worked via signals
                return True # Assume success if setting text didn't fail

        except Exception as e:
            print(f"Error in handle_set_colormap for '{target_str}': {e}\n{traceback.format_exc()}")
            QMessageBox.critical(None, "Execution Error", f"Failed to set colormap for '{target_str}': {e}")
            return False

    def handle_invert_colormap(self, params, panel, mode):
        """Handler for 'invert_colormap'. Operates on the provided panel."""
        target_str = params.get('target', mode.value)
        print(f"Handling invert_colormap on '{target_str}'...")
        if panel is None: print("Error: Panel is None in handle_invert_colormap."); return False
        try:
            # Toggle the state of the checkbox
            new_state = not panel.invert_checkbox.isChecked()
            panel.invert_checkbox.setChecked(new_state)
            print(f"Colormap inversion toggled to {new_state} for '{target_str}'.")
            # Explicitly call the change method AFTER setting the checkbox state
            if hasattr(panel, 'change_color_scale'):
                panel.change_color_scale()
                return True
            else:
                print(f"Warning: Panel for '{target_str}' does not have 'change_color_scale' method.")
                return True # Assume checkbox toggle worked

        except Exception as e:
            print(f"Error in handle_invert_colormap for '{target_str}': {e}\n{traceback.format_exc()}")
            QMessageBox.critical(None, "Execution Error", f"Failed to invert colormap for '{target_str}': {e}")
            return False

    # Updated signature to accept enable parameter directly
    def handle_log_scale(self, params, panel, mode, enable):
        """Handler for 'log_scale_on'/'off'. Operates on the provided panel."""
        target_str = params.get('target', mode.value)
        action = 'On' if enable else 'Off'
        print(f"Handling log_scale {action} on '{target_str}'...")
        if panel is None: print(f"Error: Panel is None in handle_log_scale for '{target_str}'."); return False

        # Check prerequisites only if enabling log scale
        if enable:
            try:
                # Read current vmin/vmax from the panel's input fields
                vmin_text = panel.intensity_min.text()
                vmax_text = panel.intensity_max.text()
                if not vmin_text or not vmax_text:
                    raise ValueError("vmin or vmax field is empty.")
                vmin = float(vmin_text)
                vmax = float(vmax_text)
                # Log scale requires positive values
                if vmin <= 0 or vmax <= 0:
                    QMessageBox.warning(None, "Log Scale Error", f"Log scale for '{target_str}' requires positive vmin and vmax. Current: vmin={vmin_text}, vmax={vmax_text}.")
                    panel.log_checkbox.setChecked(False) # Ensure checkbox reflects failure
                    return False
            except ValueError as e:
                QMessageBox.warning(None, "Log Scale Error", f"Log scale for '{target_str}' requires valid numeric vmin/vmax. Error: {e}")
                panel.log_checkbox.setChecked(False) # Ensure checkbox reflects failure
                return False
            except Exception as e: # Catch other potential errors reading UI elements
                print(f"Error reading vmin/vmax for log scale check on '{target_str}': {e}")
                panel.log_checkbox.setChecked(False)
                return False

        # Update checkbox and trigger the panel's update logic
        try:
            panel.log_checkbox.setChecked(enable)
            # Explicitly call the panel's method to apply the change
            if hasattr(panel, 'toggle_log_scale'):
                panel.toggle_log_scale()
                print(f"Log scale toggled to {action} and applied for '{target_str}'.")
                return True
            else:
                print(f"Warning: Panel for '{target_str}' does not have 'toggle_log_scale' method.")
                return True # Assume checkbox toggle might have worked via signals

        except Exception as e:
            print(f"Error in handle_log_scale during apply for '{target_str}': {e}\n{traceback.format_exc()}")
            QMessageBox.critical(None, "Execution Error", f"Failed to toggle log scale for '{target_str}': {e}")
            # Attempt to revert checkbox state on error? Maybe too complex.
            return False

    def handle_set_gamma(self, params, panel, mode):
        """Handler for 'set_gamma'. Operates on the provided panel."""
        gamma_value_in = params.get('gamma_value')
        target_str = params.get('target', mode.value)
        if gamma_value_in is None:
            QMessageBox.warning(None, "Parameter Error", f"Gamma value missing for '{target_str}'.")
            return False
        print(f"Handling set_gamma to {gamma_value_in} on '{target_str}'...")
        if panel is None: print(f"Error: Panel is None in handle_set_gamma for '{target_str}'."); return False

        try:
            gamma_value = float(gamma_value_in)
            # Gamma must be positive
            if gamma_value <= 0:
                QMessageBox.warning(None, "Input Error", f"Gamma value ({gamma_value}) must be positive for '{target_str}'.")
                return False

            # Update UI spinbox
            panel.gamma_spinbox.setValue(gamma_value)

            # Explicitly call the panel's method to apply the gamma change
            if hasattr(panel, 'update_gamma_from_spinbox'):
                panel.update_gamma_from_spinbox()
                print(f"Gamma set to {gamma_value} and applied for '{target_str}'.")
                return True
            else:
                print(f"Warning: Panel for '{target_str}' does not have 'update_gamma_from_spinbox' method.")
                return True # Assume setting spinbox might have worked via signals

        except (ValueError, TypeError) as e:
            QMessageBox.critical(None, "Parameter Error", f"Invalid gamma value format for '{target_str}': {e}")
            return False
        except Exception as e:
            print(f"Error in handle_set_gamma for '{target_str}': {e}\n{traceback.format_exc()}")
            QMessageBox.critical(None, "Execution Error", f"Failed to set gamma for '{target_str}': {e}")
            return False
