import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import json
import os

class ColorSelectionDialog(tk.Toplevel):
    def __init__(self, parent, detected_colors):
        super().__init__(parent)
        self.title("Color Selection")
        self.detected_colors = detected_colors
        self.selected_color = None
        
        # Center the window
        window_width = 300
        window_height = 400
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        ttk.Label(self, text="Select color for your action:").pack(pady=10)
        
        # Create a canvas for each detected color
        for i, color in enumerate(detected_colors):
            frame = ttk.Frame(self)
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Canvas to display the color
            canvas = tk.Canvas(frame, width=30, height=30, bg=self.rgb_to_hex(color))
            canvas.pack(side=tk.LEFT, padx=5)
            
            # Button to select the color
            ttk.Button(frame, text=f"Select RGB{color}", 
                      command=lambda c=color: self.select_color(c)).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self, text="Note: You can adjust tolerance afterwards").pack(pady=10)
    
    def rgb_to_hex(self, rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb[:3])
    
    def select_color(self, color):
        self.selected_color = color
        self.destroy()

class CropWindow(tk.Toplevel):
    def __init__(self, parent, image_path, callback):
        super().__init__(parent)
        self.title("Crop Image")
        
        self.image_path = image_path
        self.callback = callback
        self.crop_coords = None
        self.control_points = []
        self.selected_point = None
        self.point_radius = 5
        
        self.original_image = Image.open(image_path)
        
        display_size = (800, 800)
        self.display_image = self.original_image.copy()
        self.display_image.thumbnail(display_size)
        
        self.scale_x = self.original_image.width / self.display_image.width
        self.scale_y = self.original_image.height / self.display_image.height
        
        self.canvas = tk.Canvas(self, width=self.display_image.width, 
                              height=self.display_image.height)
        self.canvas.pack(pady=10)
        
        self.photo = ImageTk.PhotoImage(self.display_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        
        ttk.Button(self, text="Validate Selection", 
                  command=self.validate_crop).pack(pady=5)
        ttk.Label(self, text="Click and drag to select the range area.\n"
                 "Use control points to adjust the selection.").pack(pady=5)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)
    
    def create_control_points(self, x1, y1, x2, y2):
        # Clear existing control points
        for point in self.control_points:
            self.canvas.delete(point)
        self.control_points.clear()
        
        # Create new control points at corners and midpoints
        points = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
            ((x1 + x2) / 2, y1),  # Top-middle
            (x2, (y1 + y2) / 2),  # Right-middle
            ((x1 + x2) / 2, y2),  # Bottom-middle
            (x1, (y1 + y2) / 2)   # Left-middle
        ]
        
        for x, y in points:
            point = self.canvas.create_oval(
                x - self.point_radius, y - self.point_radius,
                x + self.point_radius, y + self.point_radius,
                fill='red'
            )
            self.control_points.append(point)
    
    def get_point_type(self, point_index):
        if point_index < 4:  # Corner points
            return "corner", point_index
        else:  # Middle points
            return "middle", point_index - 4
    
    def update_rectangle(self, point_index, new_x, new_y):
        coords = list(self.canvas.coords(self.rect_id))
        point_type, idx = self.get_point_type(point_index)
        
        if point_type == "corner":
            if idx == 0:  # Top-left
                coords[0], coords[1] = new_x, new_y
            elif idx == 1:  # Top-right
                coords[2], coords[1] = new_x, new_y
            elif idx == 2:  # Bottom-right
                coords[2], coords[3] = new_x, new_y
            elif idx == 3:  # Bottom-left
                coords[0], coords[3] = new_x, new_y
        else:  # Middle points
            if idx == 0:  # Top-middle
                coords[1] = new_y
            elif idx == 1:  # Right-middle
                coords[2] = new_x
            elif idx == 2:  # Bottom-middle
                coords[3] = new_y
            elif idx == 3:  # Left-middle
                coords[0] = new_x
        
        self.canvas.coords(self.rect_id, *coords)
        self.create_control_points(*coords)
    
    def find_nearest_point(self, x, y):
        for i, point in enumerate(self.control_points):
            point_coords = self.canvas.coords(point)
            point_x = (point_coords[0] + point_coords[2]) / 2
            point_y = (point_coords[1] + point_coords[3]) / 2
            
            distance = ((point_x - x) ** 2 + (point_y - y) ** 2) ** 0.5
            if distance <= self.point_radius * 2:
                return i
        return None
    
    def on_press(self, event):
        self.selected_point = self.find_nearest_point(event.x, event.y)
        
        if self.selected_point is None:
            self.start_x = event.x
            self.start_y = event.y
            
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                for point in self.control_points:
                    self.canvas.delete(point)
                self.control_points.clear()
            
            self.rect_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, self.start_x, self.start_y,
                outline='red', width=2
            )
    
    def on_drag(self, event):
        if self.selected_point is not None:
            self.update_rectangle(self.selected_point, event.x, event.y)
        elif self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)
    
    def on_release(self, event):
        if self.selected_point is None and self.rect_id:
            coords = self.canvas.coords(self.rect_id)
            self.create_control_points(*coords)
        
        self.selected_point = None
        
        if self.rect_id:
            coords = self.canvas.coords(self.rect_id)
            x1 = min(coords[0], coords[2]) * self.scale_x
            y1 = min(coords[1], coords[3]) * self.scale_y
            x2 = max(coords[0], coords[2]) * self.scale_x
            y2 = max(coords[1], coords[3]) * self.scale_y
            self.crop_coords = (int(x1), int(y1), int(x2), int(y2))
    
    def on_motion(self, event):
        if self.find_nearest_point(event.x, event.y) is not None:
            self.canvas.configure(cursor="hand2")
        else:
            self.canvas.configure(cursor="")
    
    def validate_crop(self):
        if self.crop_coords:
            cropped_image = self.original_image.crop(self.crop_coords)
            self.callback(cropped_image)
            self.destroy()
            
class PreviewWindow(tk.Toplevel):
    def __init__(self, parent, image, color_analyzer):
        super().__init__(parent)
        self.title("Color Preview")
        
        self.image = image
        self.color_analyzer = color_analyzer
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = min(800, screen_width-100)
        window_height = min(800, screen_height-100)
        self.geometry(f"{window_width}x{window_height}")
        
        self.canvas = tk.Canvas(self, width=window_width, height=window_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.update_preview()
        
    def update_preview(self):
        preview = self.image.copy()
        draw = ImageDraw.Draw(preview)
        
        cell_width = preview.width // 13
        cell_height = preview.height // 13
        
        for row in range(13):
            for col in range(13):
                center_x = int((col + 0.5) * cell_width)
                center_y = int((row + 0.5) * cell_height)
                color = self.color_analyzer.get_average_color_around_point(center_x, center_y)
                
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = (col + 1) * cell_width
                y2 = (row + 1) * cell_height
                
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='white')
                
                hand = self.get_hand_notation(row, col)
                text_color = 'white' if sum(color[:3]) < 384 else 'black'
                draw.text((x1 + 5, y1 + 5), hand, fill=text_color)
        
        display_size = (800, 800)
        preview.thumbnail(display_size)
        
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
    
    def get_hand_notation(self, row, col):
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        if row == col:
            return f"{ranks[row]}{ranks[col]}"
        elif row < col:
            return f"{ranks[row]}{ranks[col]}s"
        else:
            return f"{ranks[col]}{ranks[row]}o"

class StatsWindow(tk.Toplevel):
    def __init__(self, parent, stats):
        super().__init__(parent)
        self.title("Range Statistics")
        
        self.geometry("600x800")
        
        self.text_widget = tk.Text(self, wrap=tk.WORD, padx=10, pady=10)
        scrollbar = ttk.Scrollbar(self, command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.display_stats(stats)
        
        ttk.Button(self, text="Export Statistics", 
                  command=self.export_stats).pack(side=tk.BOTTOM, pady=10)
    
    def display_stats(self, stats):
        self.text_widget.delete(1.0, tk.END)
        
        self.text_widget.insert(tk.END, "GENERAL STATISTICS\n", "heading")
        self.text_widget.insert(tk.END, f"Total number of hands: {stats['total_hands']}\n\n")
        
        self.text_widget.insert(tk.END, "DISTRIBUTION BY TYPE\n", "heading")
        for hand_type, count in stats['hand_types'].items():
            percentage = (count / stats['total_hands']) * 100 if stats['total_hands'] > 0 else 0
            self.text_widget.insert(tk.END, 
                f"{hand_type.capitalize()}: {count} ({percentage:.1f}%)\n")
        self.text_widget.insert(tk.END, "\n")
        
        self.text_widget.insert(tk.END, "DISTRIBUTION BY ACTION\n", "heading")
        for action, percentage in stats['percentage_by_action'].items():
            self.text_widget.insert(tk.END, 
                f"{action}: {percentage:.1f}%\n")
        self.text_widget.insert(tk.END, "\n")
        
        self.text_widget.tag_configure("heading", font=("Arial", 12, "bold"))
    
    def export_stats(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="poker_range_stats.txt"
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.text_widget.get(1.0, tk.END))

class PokerRangeAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Poker Range Extraction")
        self.geometry("1400x800")
        
        self.ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        self.cell_positions = self.create_cell_positions()
        
        self.reference_colors = {
            "allin": None,
            "raise_2.5": None,
            "raise_3.5": None,
            "call": None,
            "check": None,
            "fold": None
        }
        
        self.color_tolerance = 30
        self.grid_visible = False
        self.highlight_cells = {}
        
        self.preview_window = None
        self.stats_window = None
        
        self.setup_ui()

    def create_cell_positions(self):
        positions = {}
        for i, high in enumerate(self.ranks):
            for j, low in enumerate(self.ranks):
                if i == j:
                    hand = f"{high}{low}"
                elif i < j:
                    hand = f"{high}{low}s"
                else:
                    hand = f"{low}{high}o"
                positions[hand] = (i, j)
        return positions

    def rgb_to_hsv(self, rgb):
        r, g, b = [x/255.0 for x in rgb[:3]]
        max_rgb = max(r, g, b)
        min_rgb = min(r, g, b)
        diff = max_rgb - min_rgb
        
        if max_rgb == min_rgb:
            h = 0
        elif max_rgb == r:
            h = (60 * ((g-b)/diff) + 360) % 360
        elif max_rgb == g:
            h = (60 * ((b-r)/diff) + 120) % 360
        else:
            h = (60 * ((r-g)/diff) + 240) % 360
        
        s = 0 if max_rgb == 0 else (diff/max_rgb)
        v = max_rgb
        
        return h, s, v

    def merge_similar_colors(self, colors, similarity_threshold=50):
        """
        Merges colors that are too similar to each other.
        """
        merged_colors = []
        skip_indices = set()
        
        # First sort colors by brightness
        sorted_colors = sorted(colors, key=lambda c: sum(c[:3]))
        
        for i, color1 in enumerate(sorted_colors):
            if i in skip_indices:
                continue
                
            similar_colors = [color1]
            
            # Look for similar colors
            for j, color2 in enumerate(sorted_colors[i+1:], start=i+1):
                if j in skip_indices:
                    continue
                    
                r1, g1, b1 = color1[:3]
                r2, g2, b2 = color2[:3]
                
                # Calculate brightness difference
                lum1 = 0.299 * r1 + 0.587 * g1 + 0.114 * b1
                lum2 = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
                lum_diff = abs(lum1 - lum2)
                
                # Calculate color difference
                color_diff = np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(color1[:3], color2[:3])]))
                
                # Combine differences
                weighted_diff = 0.7 * color_diff + 0.3 * lum_diff
                
                if weighted_diff <= similarity_threshold:
                    similar_colors.append(color2)
                    skip_indices.add(j)
            
            # Calculate average color
            if similar_colors:
                avg_color = tuple(np.median(similar_colors, axis=0).astype(int))
                merged_colors.append(avg_color)
        
        return merged_colors

    def detect_range_colors(self, min_cluster_size=5):
        """
        Automatically detects main colors in a poker range.
        """
        image_array = np.array(self.original_image)
        height, width = image_array.shape[:2]
        cell_height, cell_width = height // 13, width // 13
        
        cell_colors = []
        for row in range(13):
            for col in range(13):
                center_y = int((row + 0.5) * cell_height)
                center_x = int((col + 0.5) * cell_width)
                
                radius = 7
                y1, y2 = max(0, center_y - radius), min(height, center_y + radius)
                x1, x2 = max(0, center_x - radius), min(width, center_x + radius)
                color_sample = image_array[y1:y2, x1:x2]
                avg_color = tuple(np.median(color_sample, axis=(0, 1)).astype(int))
                cell_colors.append(avg_color)
        
        colors_array = np.array(cell_colors)
        initial_clusters = 8
        kmeans = KMeans(n_clusters=initial_clusters, random_state=42)
        color_labels = kmeans.fit_predict(colors_array)
        
        cluster_sizes = Counter(color_labels)
        significant_clusters = [idx for idx, count in cluster_sizes.items() 
                               if count >= min_cluster_size]
        
        representative_colors = kmeans.cluster_centers_[significant_clusters].astype(int)
        detected_colors = [tuple(color) for color in representative_colors]
        
        color_variance = np.std([np.mean(color[:3]) for color in detected_colors])
        similarity_threshold = max(40, min(70, color_variance * 0.4))
        
        merged_colors = self.merge_similar_colors(detected_colors, similarity_threshold)
        
        return merged_colors

    def is_color_similar(self, color1, color2):
        """
        Compares two colors using a more sophisticated metric.
        """
        h1, s1, v1 = self.rgb_to_hsv(color1[:3])
        h2, s2, v2 = self.rgb_to_hsv(color2[:3])
        
        h_diff = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180.0
        s_diff = abs(s1 - s2)
        v_diff = abs(v1 - v2)
        
        weighted_diff = (h_diff * 0.5 + s_diff * 0.25 + v_diff * 0.25) * 100
        
        return weighted_diff <= (self.color_tolerance / 100.0 * 50)

    def get_average_color_around_point(self, x, y):
        """
        Calculates the average color around a point in a more robust way.
        """
        image_array = np.array(self.original_image)
        radius = 7
        
        x1 = max(0, x - radius)
        x2 = min(image_array.shape[1], x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(image_array.shape[0], y + radius + 1)
        
        color_sample = image_array[y1:y2, x1:x2]
        
        mean_color = np.mean(color_sample, axis=(0, 1))
        median_color = np.median(color_sample, axis=(0, 1))
        final_color = tuple((median_color * 0.7 + mean_color * 0.3).astype(int))
        
        return final_color

    def find_similar_cells(self, reference_color, action):
        image_array = np.array(self.original_image)
        cell_width = image_array.shape[1] // 13
        cell_height = image_array.shape[0] // 13
        
        similar_cells = []
        
        for hand, (row, col) in self.cell_positions.items():
            center_y = int((row + 0.5) * cell_height)
            center_x = int((col + 0.5) * cell_width)
            
            sample_points = [
                (center_x, center_y),
                (center_x - cell_width//4, center_y),
                (center_x + cell_width//4, center_y),
                (center_x, center_y - cell_height//4),
                (center_x, center_y + cell_height//4)
            ]
            
            matches = 0
            for px, py in sample_points:
                avg_color = self.get_average_color_around_point(px, py)
                if self.is_color_similar(avg_color, reference_color):
                    matches += 1
            
            if matches >= 3:
                similar_cells.append((row, col))
        
        self.highlight_cells[action] = similar_cells

    def setup_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Basic Controls
        controls_frame = ttk.LabelFrame(left_frame, text="Basic Controls")
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="Load Range", 
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Detect Colors", 
                  command=self.auto_detect_colors).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Show/Hide Grid",
                  command=self.toggle_grid).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Preview Colors",
                  command=self.show_color_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="View Statistics",
                  command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        
        # Settings
        settings_frame = ttk.LabelFrame(left_frame, text="Settings")
        settings_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(settings_frame, text="Save Settings", 
                  command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(settings_frame, text="Load Settings",
                  command=self.load_settings).pack(side=tk.LEFT, padx=5)
        
        # Tolerance
        tolerance_frame = ttk.Frame(settings_frame)
        tolerance_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(tolerance_frame, text="Tolerance:").pack(side=tk.LEFT)
        self.tolerance_scale = ttk.Scale(tolerance_frame, from_=30, to=200,
                                      orient='horizontal', value=30,
                                      command=self.update_tolerance)
        self.tolerance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Actions
        actions_frame = ttk.LabelFrame(left_frame, text="Actions")
        actions_frame.pack(fill=tk.X, pady=5)
        
        self.action_var = tk.StringVar(value="allin")
        actions_labels = {
            "allin": "All-in",
            "raise_2.5": "Raise 2.5x",
            "raise_3.5": "Raise 3.5x",
            "call": "Call",
            "check": "Check",
            "fold": "Fold"
        }
        
        for action, label in actions_labels.items():
            rb = ttk.Radiobutton(actions_frame, text=label, value=action, 
                              variable=self.action_var)
            rb.pack(side=tk.LEFT, padx=5)
        
        # Clear
        clear_frame = ttk.LabelFrame(left_frame, text="Clear")
        clear_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(clear_frame, text="Clear Current Selection",
                 command=self.clear_current_selection).pack(side=tk.LEFT, padx=5)
        ttk.Button(clear_frame, text="Clear All Selections",
                 command=self.clear_all_selections).pack(side=tk.LEFT, padx=5)
        
        # Analysis
        analyze_frame = ttk.LabelFrame(left_frame, text="Analysis")
        analyze_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analyze_frame, text="Generate Hand List",
                 command=self.analyze_range).pack(side=tk.LEFT, padx=5)
        
        # Image
        self.image_frame = ttk.Frame(left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()
        
        # Results
        self.results_text = tk.Text(right_frame, height=40, width=60)
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            CropWindow(self, file_path, self.on_crop_complete)
    
    def on_crop_complete(self, cropped_image):
        self.original_image = cropped_image
        self.image_label.bind('<Button-1>', self.on_image_click)
        
        self.highlight_cells.clear()
        self.reference_colors = {key: None for key in self.reference_colors}
        
        self.update_display_with_highlights()
        
        display_size = (600, 600)
        display_image = self.original_image.copy()
        display_image.thumbnail(display_size)
        self.scale_x = self.original_image.width / display_image.width
        self.scale_y = self.original_image.height / display_image.height

    def on_image_click(self, event):
        if not hasattr(self, 'original_image'):
            return
        
        x = int(event.x * self.scale_x)
        y = int(event.y * self.scale_y)
        
        color = self.get_average_color_around_point(x, y)
        selected_action = self.action_var.get()
        self.reference_colors[selected_action] = color
        
        self.find_similar_cells(color, selected_action)
        self.update_display_with_highlights()
        
        self.results_text.insert(tk.END, 
            f"Selected color for {selected_action}: RGB{color}\n")

    def auto_detect_colors(self):
        if not hasattr(self, 'original_image'):
            self.results_text.insert(tk.END, "Please load an image first\n")
            return
        
        try:
            detected_colors = self.detect_range_colors()
            
            self.results_text.insert(tk.END, "\nDetected colors after merging:\n")
            for color in detected_colors:
                self.results_text.insert(tk.END, f"RGB{color}\n")
            
            self.results_text.insert(tk.END, f"\nTolerance automatically adjusted to: {self.color_tolerance}\n")
            
            dialog = ColorSelectionDialog(self, detected_colors)
            self.wait_window(dialog)
            
            if dialog.selected_color is not None:
                current_action = self.action_var.get()
                self.reference_colors[current_action] = dialog.selected_color
                self.find_similar_cells(dialog.selected_color, current_action)
                self.update_display_with_highlights()
                
                self.results_text.insert(tk.END, 
                    f"\nColor RGB{dialog.selected_color} associated with action {current_action}\n")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"\nError during automatic detection: {str(e)}\n")

    def show_color_preview(self):
        if not hasattr(self, 'original_image'):
            self.results_text.insert(tk.END, "Please load an image first\n")
            return
            
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = PreviewWindow(self, self.original_image, self)
        else:
            self.preview_window.lift()
            self.preview_window.update_preview()

    def show_statistics(self):
        if not self.highlight_cells:
            self.results_text.insert(tk.END, "No selection to analyze\n")
            return
            
        stats = self.analyze_range_statistics()
        
        if self.stats_window is None or not self.stats_window.winfo_exists():
            self.stats_window = StatsWindow(self, stats)
        else:
            self.stats_window.display_stats(stats)
            self.stats_window.lift()

    def analyze_range_statistics(self):
        total_hands = sum(len(self.get_hands_for_cells(cells)) 
                         for cells in self.highlight_cells.values())
        
        stats = {
            'total_hands': total_hands,
            'hand_types': {
                'pairs': 0,
                'suited': 0,
                'offsuit': 0
            },
            'percentage_by_action': {}
        }
        
        if total_hands == 0:
            return stats
            
        for action, cells in self.highlight_cells.items():
            hands = self.get_hands_for_cells(cells)
            stats['percentage_by_action'][action] = (len(hands) / total_hands) * 100
            
            for hand in hands:
                if len(hand) == 2:  # Pair
                    stats['hand_types']['pairs'] += 1
                elif hand.endswith('s'):  # Suited
                    stats['hand_types']['suited'] += 1
                else:  # Offsuit
                    stats['hand_types']['offsuit'] += 1
        
        return stats

    def get_hands_for_cells(self, cells):
        hands = []
        for row, col in cells:
            for hand, (h_row, h_col) in self.cell_positions.items():
                if h_row == row and h_col == col:
                    hands.append(hand)
        return hands

    def clear_current_selection(self):
        current_action = self.action_var.get()
        if current_action in self.highlight_cells:
            del self.highlight_cells[current_action]
            self.reference_colors[current_action] = None
            self.update_display_with_highlights()
            self.results_text.insert(tk.END, f"\nSelection cleared for {current_action}\n")

    def clear_all_selections(self):
        self.highlight_cells.clear()
        self.reference_colors = {k: None for k in self.reference_colors}
        self.update_display_with_highlights()
        self.results_text.insert(tk.END, "\nAll selections have been cleared\n")

    def update_tolerance(self, value):
        self.color_tolerance = int(float(value))
        if hasattr(self, 'original_image'):
            self.update_all_highlights()

    def update_all_highlights(self):
        if not hasattr(self, 'original_image'):
            return
            
        for action, color in self.reference_colors.items():
            if color is not None:
                self.find_similar_cells(color, action)
        
        self.update_display_with_highlights()

    def draw_grid(self, image):
        draw = ImageDraw.Draw(image)
        width, height = image.size
        cell_width = width // 13
        cell_height = height // 13
        
        for i in range(14):
            y = i * cell_height
            draw.line([(0, y), (width, y)], fill='red', width=1)
        
        for i in range(14):
            x = i * cell_width
            draw.line([(x, 0), (x, height)], fill='red', width=1)
        
        return image

    def update_display_with_highlights(self):
        if not hasattr(self, 'original_image'):
            return
            
        display_image = self.original_image.copy()
        
        if self.grid_visible:
            display_image = self.draw_grid(display_image)
        
        draw = ImageDraw.Draw(display_image)
        cell_width = display_image.width // 13
        cell_height = display_image.height // 13
        
        highlight_colors = {
            "allin": "red",
            "raise_2.5": "yellow",
            "raise_3.5": "green",
            "call": "blue",
            "check": "purple",
            "fold": "orange"
        }
        
        for action, cells in self.highlight_cells.items():
            color = highlight_colors.get(action, "white")
            for row, col in cells:
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = (col + 1) * cell_width
                y2 = (row + 1) * cell_height
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        display_size = (600, 600)
        display_image.thumbnail(display_size)
        
        self.photo = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=self.photo)

    def toggle_grid(self):
        self.grid_visible = not self.grid_visible
        self.update_display_with_highlights()

    def save_settings(self):
        settings = {
            'color_tolerance': int(self.color_tolerance),
            'reference_colors': {
                k: [int(x) for x in v] if v is not None else None
                for k, v in self.reference_colors.items()
            }
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="poker_range_settings.json"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(settings, f)
            self.results_text.insert(tk.END, f"\nSettings saved to {file_path}\n")

    def load_settings(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                self.color_tolerance = int(settings['color_tolerance'])
                self.tolerance_scale.set(self.color_tolerance)
                
                self.reference_colors = {
                    k: tuple(v) if v is not None else None
                    for k, v in settings['reference_colors'].items()
                }
                
                if hasattr(self, 'original_image'):
                    self.update_all_highlights()
                
                self.results_text.insert(tk.END, f"\nSettings loaded from {file_path}\n")
            except Exception as e:
                self.results_text.insert(tk.END, f"\nError loading settings: {str(e)}\n")

    def analyze_range(self):
        if not hasattr(self, 'original_image') or not self.highlight_cells:
            self.results_text.insert(tk.END, 
                "Please load an image and select colors first\n")
            return
        
        hands_by_action = defaultdict(list)
        
        for action, cells in self.highlight_cells.items():
            hands = self.get_hands_for_cells(cells)
            hands_by_action[action].extend(hands)
        
        self.results_text.delete(1.0, tk.END)
        action_names = {
            "allin": "All-in",
            "raise_2.5": "Raise 2.5x",
            "raise_3.5": "Raise 3.5x",
            "call": "Call",
            "check": "Check",
            "fold": "Fold"
        }
        
        for action in action_names:
            if action in hands_by_action:
                hands = sorted(hands_by_action[action], key=self.hand_value)
                self.results_text.insert(tk.END, f"\n{action_names[action]}\n")
                text_hands = self.format_hands_for_text(hands)
                self.results_text.insert(tk.END, text_hands)
                self.results_text.insert(tk.END, f"\nNumber of hands: {len(hands)}\n")
                self.results_text.insert(tk.END, "\n")

    def hand_value(self, hand):
        if len(hand) == 2:  # Pair
            rank = self.ranks.index(hand[0])
            return (0, rank)
        else:
            high = self.ranks.index(hand[0])
            low = self.ranks.index(hand[1])
            is_suited = hand.endswith('s')
            return (1 if is_suited else 2, high, low)

    def format_hands_for_text(self, hands):
        pairs = []
        suited = []
        offsuit = []
        
        for hand in hands:
            if len(hand) == 2:
                pairs.append(hand)
            elif hand.endswith('s'):
                suited.append(hand)
            else:
                offsuit.append(hand)
        
        text_parts = []
        if pairs:
            text_parts.append(" ".join(pairs))
        if suited:
            text_parts.append(" ".join(suited))
        if offsuit:
            text_parts.append(" ".join(offsuit))
        
        return "\n".join(text_parts)

if __name__ == "__main__":
    app = PokerRangeAnalyzer()
    app.mainloop()
