# AI Photo Enhancer

## Overview

This is a Flask-based web application that provides AI-powered photo enhancement capabilities. The application allows users to upload images, apply various enhancement filters, and compare original vs enhanced versions. It's designed as a social media photo enhancement tool with a clean, responsive interface.

## System Architecture

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Flask
- **CSS Framework**: Bootstrap 5 (dark theme variant)
- **JavaScript**: Vanilla JavaScript for interactive features
- **Icons**: Font Awesome 6.0
- **Responsive Design**: Mobile-first approach with Bootstrap grid system

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Architecture Pattern**: MVC-like structure with separated routes, utilities, and templates
- **Image Processing**: PIL (Python Imaging Library) for image manipulation
- **File Handling**: Werkzeug utilities for secure file operations
- **Session Management**: Flask sessions with configurable secret key

## Key Components

### Core Files
1. **app.py**: Main Flask application configuration and initialization
2. **main.py**: Entry point for running the application
3. **routes.py**: URL routing and request handling logic
4. **utils.py**: Image processing utilities and helper functions
5. **stable_diffusion_enhancer.py**: ComfyUI + Stable Diffusion + LoRA enhancement engine

### Frontend Components
1. **templates/index.html**: Main upload page with drag-and-drop functionality
2. **templates/gallery.html**: Gallery view for uploaded images
3. **templates/enhance.html**: Enhancement interface with comparison tools
4. **static/css/style.css**: Custom styling for upload areas and gallery cards
5. **static/js/main.js**: Upload functionality and form handling
6. **static/js/comparison.js**: Image comparison slider for before/after views

### Key Features
- **File Upload**: Drag-and-drop and click-to-upload functionality
- **Image Gallery**: Grid-based display of uploaded images with metadata
- **Advanced AI Enhancement Engine**: ComfyUI + Stable Diffusion + LoRA-based processing
- **Comprehensive Feature Set**: 15+ enhancement capabilities including:
  - Eye color and shape adjustment
  - Face shape modification
  - Hair color, style, and length changes
  - Lip shape and color enhancement
  - Height adjustment with pose AI
  - Body shape and proportion adjustment
  - Background changes or replacement
  - Skin tone correction and blemish removal
  - Expression and makeup changes
  - Lighting and clothing modifications
- **Natural Language Processing**: AI prompt parsing for intuitive enhancement requests
- **Fallback Processing**: Traditional PIL-based enhancement when AI is unavailable
- **Comparison Tool**: Before/after slider for enhanced images
- **Responsive Design**: Mobile-friendly interface

## Data Flow

1. **Upload Process**:
   - User uploads image via drag-and-drop or file selection
   - File validation checks for allowed extensions (PNG, JPG, JPEG, GIF)
   - Secure filename generation with UUID
   - File saved to `uploads/` directory

2. **Enhancement Process**:
   - User selects image from gallery
   - Enhancement parameters configured via form
   - PIL applies filters (brightness, contrast, sharpness, etc.)
   - Enhanced image saved to `enhanced/` directory with prefix

3. **Gallery Display**:
   - Scans upload directory for images
   - Extracts metadata (file size, dimensions)
   - Checks for corresponding enhanced versions
   - Displays in responsive grid layout

## External Dependencies

### Python Packages
- **Flask**: Web framework
- **Pillow (PIL)**: Image processing library
- **Werkzeug**: WSGI utilities (included with Flask)

### Frontend Dependencies (CDN)
- **Bootstrap 5**: CSS framework (agent dark theme)
- **Font Awesome 6.0**: Icon library

### File System Dependencies
- **uploads/**: Directory for original uploaded images
- **enhanced/**: Directory for AI-enhanced images

## Deployment Strategy

### Development Configuration
- **Debug Mode**: Enabled in development
- **Host**: 0.0.0.0 (allows external connections)
- **Port**: 5000
- **File Size Limit**: 16MB maximum upload size

### Production Considerations
- Session secret key should be set via environment variable
- Debug mode should be disabled
- Consider using reverse proxy (nginx) for static file serving
- Implement proper error handling and logging
- Add rate limiting for uploads

### Environment Variables
- `SESSION_SECRET`: Flask session secret key (defaults to development key)
- `OPENAI_API_KEY`: OpenAI API key (legacy, kept for backward compatibility)

## Changelog
- July 03, 2025. Initial setup
- July 03, 2025. Integrated real AI enhancement using OpenAI API with Stable Diffusion-style image generation, added text prompt functionality for custom enhancement descriptions
- September 02, 2025. **Major Architecture Overhaul**: Replaced OpenAI-based enhancement with ComfyUI + Stable Diffusion + LoRA system. Implemented 15+ professional photo editing features including eye color/shape adjustment, face modification, hair styling, body proportions, background replacement, skin correction, makeup application, and advanced lighting controls. Added natural language prompt parsing for intuitive enhancement requests.
- September 07, 2025. **Gemini Integration**: Replaced OpenCV/Stable Diffusion system with Google's Gemini AI for image enhancement. Simplified architecture by removing complex facial detection algorithms and using Gemini's natural language processing for photo enhancement. Removed opencv-python and numpy dependencies. The system now uses Gemini's multimodal capabilities for AI-powered photo enhancement with natural language prompts.

## User Preferences

Preferred communication style: Simple, everyday language.