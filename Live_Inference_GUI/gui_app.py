"""
Emotion AI Dashboard Application
--------------------------------
A CustomTkinter-based GUI for:
1. Live Camera Inference with Grad-CAM visualization.
2. Batch Evaluation on test datasets.
3. Model management and performance tracking.
"""

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageOps
import threading
import time
import sys
import os
import numpy as np
import json

# Add parent directory to path to import inference module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import ModelWrapper, FaceDetector
from evaluator import Evaluator

# Set Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Emotion AI - Live Inference & Evaluation")
        self.geometry("1400x900")
        
        # --- Configuration ---
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.config_path = os.path.join(self.project_root, 'configs', 'config.yaml')
        self.checkpoint_path = os.path.join(self.project_root, 'checkpoints', 'last_checkpoint.pth')
        self.holdout_dir = os.path.join(self.project_root, 'archive', 'test_holdout')
        
        # --- State ---
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.frame_count = 0
        self.last_inference_time = 0
        self.current_faces = [] 
        self.current_heatmap_img = None
        self.eval_thread = None
        self.stop_eval = False
        
        # --- GUI Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # Content row expands

        self.create_header()
        self.create_tabs()
        self.create_status_bar()
        
        # Load resources in background
        self.status_label.configure(text="Loading Model & Resources...")
        threading.Thread(target=self.load_resources, daemon=True).start()

    def create_header(self):
        # Header Frame
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        # Title
        title = ctk.CTkLabel(self.header_frame, text="Emotion AI Dashboard", font=("Roboto Medium", 24))
        title.pack(side="left")
        
        # Controls (Right aligned in header)
        self.controls_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.controls_frame.pack(side="right")
        
        # Model Selector
        self.checkpoint_files = self.get_checkpoint_list()
        self.model_var = ctk.StringVar(value=self.checkpoint_files[0] if self.checkpoint_files else "No Checkpoints")
        self.model_combo = ctk.CTkComboBox(self.controls_frame, values=self.checkpoint_files, variable=self.model_var, command=self.change_model, width=200)
        self.model_combo.pack(side="left", padx=(0, 5))
        
        self.btn_refresh = ctk.CTkButton(self.controls_frame, text="↻", width=30, command=self.refresh_checkpoints)
        self.btn_refresh.pack(side="left", padx=(0, 20))
        
        # Live Controls (Only visible/active in Live Tab, but placed here for consistency)
        self.btn_start = ctk.CTkButton(self.controls_frame, text="Start Camera", command=self.start_camera, state="disabled", width=120)
        self.btn_start.pack(side="left", padx=5)
        
        self.btn_pause = ctk.CTkButton(self.controls_frame, text="Pause", command=self.toggle_pause, state="disabled", width=80, fg_color="gray")
        self.btn_pause.pack(side="left", padx=5)
        
        self.btn_stop = ctk.CTkButton(self.controls_frame, text="Stop", command=self.stop_camera, state="disabled", width=80, fg_color="#C62828", hover_color="#B71C1C")
        self.btn_stop.pack(side="left", padx=5)
        
        # Frequency Selector
        self.freq_var = ctk.StringVar(value="Every Frame")
        self.freq_combo = ctk.CTkComboBox(self.controls_frame, values=["Every Frame", "Every 5 Frames", "Every 10 Frames", "Once per Second"], variable=self.freq_var, width=150)
        self.freq_combo.pack(side="left", padx=(20, 0))

    def create_tabs(self):
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        self.tab_view.add("Live Camera")
        self.tab_view.add("Batch Evaluation")
        
        self.create_live_ui(self.tab_view.tab("Live Camera"))
        self.create_eval_ui(self.tab_view.tab("Batch Evaluation"))

    def create_live_ui(self, parent):
        # 2:1 Split
        parent.grid_columnconfigure(0, weight=2, uniform="split")
        parent.grid_columnconfigure(1, weight=1, uniform="split")
        parent.grid_rowconfigure(0, weight=1)
        
        # Left: Camera Feed
        self.video_container = ctk.CTkFrame(parent, corner_radius=15)
        self.video_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        
        self.lbl_video = ctk.CTkLabel(self.video_container, text="", corner_radius=15)
        self.lbl_video.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right: Analysis
        self.analysis_container = ctk.CTkFrame(parent, corner_radius=15)
        self.analysis_container.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        
        # Title for Analysis
        ctk.CTkLabel(self.analysis_container, text="Model Focus (Grad-CAM)", font=("Roboto Medium", 16)).pack(pady=(15, 5))
        
        # Heatmap Image
        self.lbl_heatmap = ctk.CTkLabel(self.analysis_container, text="Waiting for face...", corner_radius=15)
        self.lbl_heatmap.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Info / Stats
        self.info_frame = ctk.CTkFrame(self.analysis_container, fg_color="transparent")
        self.info_frame.pack(fill="x", padx=15, pady=15)
        
        self.lbl_emotion = ctk.CTkLabel(self.info_frame, text="--", font=("Roboto", 28, "bold"), text_color="#4FC3F7")
        self.lbl_emotion.pack()
        
        self.lbl_conf = ctk.CTkLabel(self.info_frame, text="Confidence: --%", font=("Roboto", 14), text_color="gray")
        self.lbl_conf.pack()

    def create_eval_ui(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=3) # Main content
        parent.grid_rowconfigure(1, weight=1) # Results
        
        # --- Top Section: Preview & Progress ---
        
        # Left: Image Preview
        self.eval_preview_frame = ctk.CTkFrame(parent, corner_radius=15)
        self.eval_preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.eval_preview_frame, text="Current Image", font=("Roboto Medium", 14)).pack(pady=5)
        self.lbl_eval_image = ctk.CTkLabel(self.eval_preview_frame, text="No Evaluation Running", corner_radius=10)
        self.lbl_eval_image.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right: Real-time Stats
        self.eval_stats_frame = ctk.CTkFrame(parent, corner_radius=15)
        self.eval_stats_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.eval_stats_frame, text="Evaluation Progress", font=("Roboto Medium", 14)).pack(pady=5)
        
        self.eval_progress = ctk.CTkProgressBar(self.eval_stats_frame)
        self.eval_progress.pack(fill="x", padx=20, pady=20)
        self.eval_progress.set(0)
        
        self.lbl_eval_status = ctk.CTkLabel(self.eval_stats_frame, text="Ready", font=("Roboto", 14))
        self.lbl_eval_status.pack(pady=5)
        
        self.lbl_eval_acc = ctk.CTkLabel(self.eval_stats_frame, text="Current Accuracy: --%", font=("Roboto", 20, "bold"))
        self.lbl_eval_acc.pack(pady=20)
        
        self.btn_run_eval = ctk.CTkButton(self.eval_stats_frame, text="Start Evaluation", command=self.start_evaluation, height=40, font=("Roboto", 14, "bold"))
        self.btn_run_eval.pack(pady=20)

        # --- Bottom Section: Results & Comparison ---
        self.results_frame = ctk.CTkScrollableFrame(parent, label_text="Final Results & Comparison")
        self.results_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        self.lbl_results_text = ctk.CTkLabel(self.results_frame, text="Run evaluation to see results.", justify="left", font=("Consolas", 14))
        self.lbl_results_text.pack(fill="both", expand=True, padx=10, pady=10)

    def create_status_bar(self):
        self.status_label = ctk.CTkLabel(self, text="Initializing...", anchor="w", font=("Roboto", 12), text_color="gray")
        self.status_label.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))

    def get_checkpoint_list(self):
        """Scans checkpoints dir and returns list of .pth files sorted by newest first."""
        checkpoints_dir = os.path.join(self.project_root, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            return ["No Checkpoints Found"]
            
        files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        if not files:
            return ["No Checkpoints Found"]
            
        files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
        return files

    def refresh_checkpoints(self):
        self.checkpoint_files = self.get_checkpoint_list()
        self.model_combo.configure(values=self.checkpoint_files)
        if self.checkpoint_files and self.checkpoint_files[0] != "No Checkpoints Found":
            # Don't auto-switch, just update list, or maybe switch to newest?
            # Let's just update values.
            pass
        self.status_label.configure(text="Refreshed checkpoint list")

    def change_model(self, choice):
        if choice == "No Checkpoints Found":
            return
        self.status_label.configure(text=f"Loading model: {choice}...")
        self.btn_start.configure(state="disabled")
        self.model_combo.configure(state="disabled")
        threading.Thread(target=self.reload_model, args=(choice,), daemon=True).start()

    def reload_model(self, checkpoint_name):
        try:
            checkpoint_path = os.path.join(self.project_root, 'checkpoints', checkpoint_name)
            self.model_wrapper = ModelWrapper(self.config_path, checkpoint_path)
            self.evaluator = Evaluator(self.model_wrapper, self.holdout_dir)
            
            if self.model_wrapper.is_trained:
                arch = self.model_wrapper.config['model'].get('name', 'unknown')
                self.after(0, lambda: self.status_label.configure(text=f"Loaded: {checkpoint_name} [{arch}]"))
            else:
                self.after(0, lambda: self.status_label.configure(text=f"Failed to load weights: {checkpoint_name} (Check Console)"))
            
            self.after(0, lambda: self.btn_start.configure(state="normal" if not self.is_running else "disabled"))
            self.after(0, lambda: self.model_combo.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error loading model: {e}"))
            self.after(0, lambda: self.model_combo.configure(state="normal"))

    def load_resources(self):
        try:
            self.detector = FaceDetector()
            
            current_ckpt = self.model_var.get()
            if current_ckpt and current_ckpt != "No Checkpoints Found":
                ckpt_path = os.path.join(self.project_root, 'checkpoints', current_ckpt)
            else:
                ckpt_path = "dummy"

            self.model_wrapper = ModelWrapper(self.config_path, ckpt_path)
            self.evaluator = Evaluator(self.model_wrapper, self.holdout_dir)
            
            self.after(0, lambda: self.status_label.configure(text="Ready"))
            self.after(0, lambda: self.btn_start.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error: {str(e)}"))
            print(f"Error loading resources: {e}")

    # --- Live Camera Methods ---
    def start_camera(self):
        if self.cap is not None: return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.configure(text="Error: Could not open camera")
            return
        self.is_running = True
        self.is_paused = False
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal")
        self.btn_stop.configure(state="normal")
        self.status_label.configure(text="Running - Detecting Faces...")
        self.update_loop()

    def stop_camera(self):
        self.is_running = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start.configure(state="normal")
        self.btn_pause.configure(state="disabled")
        self.btn_stop.configure(state="disabled")
        self.status_label.configure(text="Stopped")
        self.lbl_video.configure(image="", text="")
        self.lbl_heatmap.configure(image="", text="Waiting for face...")
        self.lbl_emotion.configure(text="--")
        self.lbl_conf.configure(text="Confidence: --%")

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.status_label.configure(text="Paused")
            self.btn_pause.configure(text="Resume")
        else:
            self.status_label.configure(text="Running")
            self.btn_pause.configure(text="Pause")

    def should_process_frame(self):
        mode = self.freq_var.get()
        if mode == "Every Frame": return True
        elif mode == "Every 5 Frames": return self.frame_count % 5 == 0
        elif mode == "Every 10 Frames": return self.frame_count % 10 == 0
        elif mode == "Once per Second":
            now = time.time()
            if now - self.last_inference_time >= 1.0:
                self.last_inference_time = now
                return True
            return False
        return True

    def resize_image_to_fit(self, pil_image, target_width, target_height):
        if target_width <= 1 or target_height <= 1: return pil_image
        w_ratio = target_width / pil_image.width
        h_ratio = target_height / pil_image.height
        scale = min(w_ratio, h_ratio)
        return pil_image.resize((int(pil_image.width * scale), int(pil_image.height * scale)), Image.Resampling.LANCZOS)

    def round_corners(self, im, radius):
        mask = Image.new('L', im.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), im.size], radius, fill=255)
        out = im.copy()
        out.putalpha(mask)
        return out

    def update_loop(self):
        if not self.is_running: return
        if self.is_paused:
            self.after(30, self.update_loop)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        self.frame_count += 1
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        if self.should_process_frame():
            faces = self.detector.detect_faces(frame)
            self.current_faces = [] 
            largest_face = None
            max_area = 0
            for (x, y, w, h) in faces:
                self.current_faces.append((x, y, w, h))
                if w * h > max_area:
                    max_area = w * h
                    largest_face = (x, y, w, h)

            if largest_face:
                x, y, w, h = largest_face
                if self.model_wrapper.is_trained:
                    face_img = frame[y:y+h, x:x+w]
                    try:
                        class_name, conf, heatmap_overlay = self.model_wrapper.predict(face_img)
                        heatmap_rgb = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)
                        heatmap_pil = Image.fromarray(heatmap_rgb)
                        
                        target_w = self.lbl_heatmap.winfo_width()
                        target_h = self.lbl_heatmap.winfo_height()
                        if target_w > 10 and target_h > 10:
                            heatmap_pil = self.resize_image_to_fit(heatmap_pil, target_w, target_h)
                            heatmap_pil = self.round_corners(heatmap_pil, 15)
                        
                        self.current_heatmap_img = ctk.CTkImage(light_image=heatmap_pil, dark_image=heatmap_pil, size=heatmap_pil.size)
                        self.lbl_heatmap.configure(image=self.current_heatmap_img, text="")
                        self.lbl_emotion.configure(text=class_name)
                        self.lbl_conf.configure(text=f"Confidence: {conf:.1%}")
                        self.last_result = (class_name, conf, largest_face)
                    except Exception as e:
                        print(f"Inference error: {e}")
                else:
                    self.lbl_heatmap.configure(image="", text="Model Untrained\nPlease run training first.")
                    self.lbl_emotion.configure(text="Untrained")
                    self.lbl_conf.configure(text="--")
                    self.last_result = None
            else:
                self.last_result = None
                self.lbl_heatmap.configure(image="", text="No Face Detected")
                self.lbl_emotion.configure(text="--")
                self.lbl_conf.configure(text="Confidence: --%")

        for (x, y, w, h) in self.current_faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if hasattr(self, 'last_result') and self.last_result:
            class_name, conf, (lx, ly, lw, lh) = self.last_result
            label = f"{class_name} ({conf:.1%})"
            cv2.putText(display_frame, label, (lx, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        target_w = self.lbl_video.winfo_width()
        target_h = self.lbl_video.winfo_height()
        if target_w > 10 and target_h > 10:
            img_pil = self.resize_image_to_fit(img_pil, target_w, target_h)
            img_pil = self.round_corners(img_pil, 15)
        
        img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=img_pil.size)
        self.lbl_video.configure(image=img_ctk, text="")
        self.after(10, self.update_loop)

    # --- Evaluation Methods ---
    def start_evaluation(self):
        if not self.model_wrapper.is_trained:
            self.lbl_results_text.configure(text="Error: Model is untrained. Cannot evaluate.")
            return
            
        self.btn_run_eval.configure(state="disabled", text="Running...")
        self.lbl_results_text.configure(text="Evaluation started...")
        self.eval_progress.set(0)
        
        threading.Thread(target=self.run_eval_thread, daemon=True).start()

    def run_eval_thread(self):
        def progress_callback(current, total, img_path, pred, true):
            # Update UI from thread
            self.after(0, lambda: self.update_eval_ui(current, total, img_path, pred, true))
            
        results = self.evaluator.evaluate(progress_callback)
        self.after(0, lambda: self.show_eval_results(results))

    def update_eval_ui(self, current, total, img_path, pred, true):
        # Update Progress
        progress = current / total
        self.eval_progress.set(progress)
        self.lbl_eval_status.configure(text=f"Processing: {current}/{total}")
        
        # Update Image
        try:
            img = Image.open(img_path)
            target_w = self.lbl_eval_image.winfo_width()
            target_h = self.lbl_eval_image.winfo_height()
            if target_w > 10 and target_h > 10:
                img = self.resize_image_to_fit(img, target_w, target_h)
                img = self.round_corners(img, 15)
            
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self.lbl_eval_image.configure(image=ctk_img, text="")
        except:
            pass
            
        # Color code status
        color = "#00E676" if pred == true else "#FF1744" # Green or Red
        self.lbl_eval_status.configure(text_color=color, text=f"Processing: {current}/{total} | Pred: {pred} (True: {true})")

    def show_eval_results(self, results):
        self.btn_run_eval.configure(state="normal", text="Start Evaluation")
        self.lbl_eval_acc.configure(text=f"Final Accuracy: {results['accuracy']:.2%}")
        
        # Format Report
        text = f"Total Images: {results['total_images']}\n"
        text += f"Correct: {results['correct_count']}\n"
        text += f"Accuracy: {results['accuracy']:.4f}\n\n"
        
        text += "--- Per Class Metrics ---\n"
        text += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n"
        text += "-"*50 + "\n"
        
        for cls, metrics in results['per_class'].items():
            text += f"{cls:<15} {metrics['precision']:.2f}       {metrics['recall']:.2f}       {metrics['f1-score']:.2f}\n"
            
        # Comparison
        current, prev = self.evaluator.get_last_comparison()
        if prev:
            text += "\n--- Comparison with Previous Run ---\n"
            acc_diff = current['accuracy'] - prev['accuracy']
            sign = "+" if acc_diff >= 0 else ""
            text += f"Accuracy Change: {sign}{acc_diff:.4f}\n"
            
            text += f"Previous Accuracy: {prev['accuracy']:.4f}\n"
            
        self.lbl_results_text.configure(text=text)

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()
