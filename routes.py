import os
import uuid
from flask import render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app import app
from utils import allowed_file, enhance_image, get_all_images

@app.route('/')
def index():
    """Main page with upload functionality"""
    return render_template('index.html')

@app.route('/gallery')
def gallery():
    """Gallery page showing all uploaded images"""
    images = get_all_images()
    return render_template('gallery.html', images=images)

@app.route('/enhance/<filename>')
def enhance_page(filename):
    """Enhancement page for a specific image"""
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(original_path):
        flash('Image not found', 'error')
        return redirect(url_for('gallery'))
    
    # Check if enhanced version already exists
    enhanced_filename = f"enhanced_{filename}"
    enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], enhanced_filename)
    has_enhanced = os.path.exists(enhanced_path)
    
    return render_template('enhance.html', 
                         original_filename=filename, 
                         enhanced_filename=enhanced_filename if has_enhanced else None,
                         has_enhanced=has_enhanced)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and file.filename and allowed_file(file.filename):
        # Generate unique filename
        original_filename = file.filename
        filename = secure_filename(original_filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        flash('Image uploaded successfully!', 'success')
        return redirect(url_for('enhance_page', filename=unique_filename))
    else:
        flash('Invalid file type. Please upload JPG, PNG, or GIF files.', 'error')
        return redirect(url_for('index'))

@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    """API endpoint to enhance an image with AI"""
    data = request.get_json()
    filename = data.get('filename')
    enhancements = data.get('enhancements', {})
    enhancement_prompt = data.get('enhancement_prompt', '')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(original_path):
        return jsonify({'error': 'Original image not found'}), 404
    
    try:
        # Generate enhanced image with AI prompt
        enhanced_filename = f"enhanced_{filename}"
        enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], enhanced_filename)
        
        success = enhance_image(original_path, enhanced_path, enhancements, enhancement_prompt)
        
        if success:
            return jsonify({
                'success': True,
                'enhanced_filename': enhanced_filename,
                'message': 'Image enhanced successfully with AI!'
            })
        else:
            return jsonify({'error': 'Enhancement failed'}), 500
            
    except Exception as e:
        app.logger.error(f"Enhancement error: {str(e)}")
        return jsonify({'error': 'Enhancement failed'}), 500

@app.route('/api/continue-edit', methods=['POST'])
def api_continue_edit():
    """API endpoint for conversational follow-up editing"""
    data = request.get_json()
    filename = data.get('filename')
    enhancement_prompt = data.get('enhancement_prompt', '')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    if not enhancement_prompt.strip():
        return jsonify({'error': 'Please provide editing instructions'}), 400
    
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(original_path):
        return jsonify({'error': 'Original image not found'}), 404
    
    try:
        # Check if enhanced version exists to use as source
        enhanced_filename = f"enhanced_{filename}"
        current_enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], enhanced_filename)
        
        # Use enhanced version as source if it exists, otherwise use original
        source_path = current_enhanced_path if os.path.exists(current_enhanced_path) else original_path
        
        # Create new enhanced version
        success = enhance_image(
            original_path, 
            current_enhanced_path, 
            {}, 
            enhancement_prompt,
            source_path  # This enables conversational editing
        )
        
        if success:
            return jsonify({
                'success': True,
                'enhanced_filename': enhanced_filename,
                'message': 'Image updated successfully!'
            })
        else:
            return jsonify({'error': 'Enhancement failed'}), 500
            
    except Exception as e:
        app.logger.error(f"Conversational edit error: {str(e)}")
        return jsonify({'error': 'Enhancement failed'}), 500

@app.route('/delete/<filename>', methods=['POST'])
def delete_image(filename):
    """Delete an uploaded image and its enhanced version"""
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], f"enhanced_{filename}")
    
    deleted = False
    if os.path.exists(original_path):
        os.remove(original_path)
        deleted = True
    
    if os.path.exists(enhanced_path):
        os.remove(enhanced_path)
        deleted = True
    
    if deleted:
        flash('Image deleted successfully!', 'success')
    else:
        flash('Image not found', 'error')
    
    return redirect(url_for('gallery'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/enhanced/<filename>')
def enhanced_file(filename):
    """Serve enhanced files"""
    return send_from_directory(app.config['ENHANCED_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))
