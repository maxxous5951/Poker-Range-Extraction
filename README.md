# Poker Range Extraction

A Python tool for extracting and analyzing poker ranges from images Automatically by color.
![PKE](https://github.com/user-attachments/assets/94e56ca5-8dfa-46e9-bb79-7819bc10678c)
# Poker Range Extraction - User Guide

## Table of Contents
1. Getting Started
2. Basic Workflow
3. Interface Overview
4. Step-by-Step Instructions
5. Advanced Features
6. Tips and Troubleshooting

## 1. Getting Started

### Installation
1. Install Python 3.7 or higher
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Launch the application:
```bash
python poker_range_extraction.py
```

## 2. Basic Workflow

The basic process of extracting a range follows these steps:
1. Load a range image
2. Select the range area by cropping
3. Choose actions and detect colors
4. Adjust settings if needed
5. Generate range analysis
6. Export results

## 3. Interface Overview

The interface is divided into several sections:

### Basic Controls
- **Load Range**: Opens file dialog to select an image
- **Detect Colors**: Automatically detects colors in the image
- **Show/Hide Grid**: Toggles grid overlay
- **Preview Colors**: Shows color detection preview
- **View Statistics**: Displays range statistics

### Settings
- **Save Settings**: Save current color and tolerance settings
- **Load Settings**: Load previously saved settings
- **Tolerance Slider**: Adjust color detection sensitivity

### Actions
- All-in
- Raise 2.5x
- Raise 3.5x
- Call
- Check
- Fold

## 4. Step-by-Step Instructions

### Loading and Cropping an Image

1. Click "Load Range"
2. Select your range image file
3. In the crop window, click and drag to select the exact range area
4. Click "Validate Selection"

### Detecting Colors

Method 1 - Automatic:
1. Click "Detect Colors"
2. Select the detected color that matches your action
3. Choose which action this color represents using the radio buttons

Method 2 - Manual:
1. Select an action using the radio buttons
2. Click directly on the color in the image
3. The program will highlight all similar colored cells

### Adjusting Settings

1. Use the Tolerance slider to fine-tune color detection:
   - Higher values: More lenient color matching
   - Lower values: Stricter color matching
2. Use Show/Hide Grid to verify cell alignment
3. Use Preview Colors to check color detection accuracy

### Analyzing Results

1. Click "Generate Hand List" to see the extracted hands
2. Use "View Statistics" to see:
   - Total number of hands
   - Distribution by hand type (pairs, suited, offsuit)
   - Percentage by action

### Saving Your Work

1. Save your settings for future use with "Save Settings"
2. Export statistics to a text file from the statistics window
3. Select the created list, and just do a crtl+c for copy

## 5. Advanced Features

### Color Preview
- Shows the detected color for each cell
- Displays hand notation overlay
- Helps verify color detection accuracy

### Statistics Window
- Detailed breakdown of your range
- Export functionality
- Visual representation of hand distributions

### Multiple Action Support
- Work with multiple actions simultaneously
- Different highlight colors for each action
- Clear individual or all selections

## 6. Tips and Troubleshooting

### For Best Results:
- Use clear, high-quality range images
- Ensure good contrast between different actions
- Crop the image precisely to the range area
- Start with default tolerance and adjust as needed

### Common Issues:

1. Poor Color Detection:
   - Try adjusting the tolerance
   - Use manual color selection
   - Ensure image quality is good

2. Misaligned Grid:
   - Ensure precise cropping
   - Use grid overlay to verify alignment

3. Incorrect Hands:
   - Double-check color assignments
   - Verify cell highlighting
   - Adjust tolerance if needed

### Tips for Accuracy:
- Start with automatic color detection
- Fine-tune with manual selection if needed
- Use the preview feature to verify results
- Save settings when you find good configurations

Remember: The quality of the extraction depends largely on the quality of the input image and precise cropping. Take time to get these right for best results.
