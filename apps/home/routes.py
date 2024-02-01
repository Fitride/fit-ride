# -*- encoding: utf-8 -*-

from apps.home import blueprint
from flask import render_template, request, Response, send_file, url_for, jsonify
from flask_login import login_required
from jinja2 import TemplateNotFound
from apps.modules.drawing import Drawing
import tempfile
import os
import uuid
from werkzeug.utils import secure_filename
import mimetypes

camera = None  # Initialize camera as None
is_camera_active = False  # Initially, the camera is not active
temp_files = {}


@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index')


@blueprint.route('/video')
@login_required
def video():
    global is_camera_active, camera
    draw_utils = Drawing()  # Initialize Drawing class
    is_camera_active, camera = draw_utils.initialize_camera(
        camera)  # Ensure the camera is initialized
    return Response(draw_utils.process_livestream(is_camera_active, camera), mimetype='multipart/x-mixed-replace; boundary=frame')


@blueprint.route('/shutdown_video', methods=['GET'])
@login_required
def shutdown_video():
    global is_camera_active, camera
    # Signal to stop/run the video capture loop
    if (is_camera_active == False):
        is_camera_active = True
        return {'status': 'Camera on'}
    else:
        is_camera_active = False
        camera.release()
        return {'status': 'Camera shutdown'}


@blueprint.route('/analyze_photo', methods=['POST'])
@login_required
def analyze_photo():
    if 'photo' not in request.files:
        return 'No file part', 400
    file = request.files['photo']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            file.save(temp_file.name)
            # Your analysis code here
            draw_image = Drawing()

            # Generate a unique identifier for the temporary file
            temp_file_id = str(uuid.uuid4())
            temp_files[temp_file_id] = temp_file.name

            drawed_image, angles_data = draw_image.process_frame_file(temp_file.name)
            drawed_image.save(temp_file.name)

        return jsonify({
            "src_url": url_for('home_blueprint.temp_file', file_id=temp_file_id),
            "angles_data": angles_data
        })


@blueprint.route('/analyze_video', methods=['POST'])
@login_required
def analyze_video():
    if 'video' not in request.files:
        return 'No file part', 400
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            # Save the uploaded video to a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_video_path = os.path.join(
                temp_dir, secure_filename(file.filename))
            file.save(temp_file.name)

            # Generate a unique identifier for the temporary file
            temp_file_id = str(uuid.uuid4())
            temp_files[temp_file_id] = temp_file.name

            draw_analysis = Drawing()
            drawed_image_temp_name, angles_data, fps_input, temp_data_name = draw_analysis.process_video_file(
                temp_file.name)

            temp_file_id_drawed = str(uuid.uuid4())
            temp_file_id_data = str(uuid.uuid4())
            temp_files[temp_file_id_drawed] = drawed_image_temp_name
            temp_files[temp_file_id_data] = temp_data_name

            return jsonify({
                "processed_video_url": url_for('home_blueprint.temp_file', file_id=temp_file_id_drawed),
                "angles_data": angles_data,
                "fps": fps_input,
                "temp_id": temp_file_id_data
            })


@blueprint.route('/temp_file/<file_id>')
@login_required
def temp_file(file_id):
    # Retrieve the file path from the global dictionary using the file_id
    file_path = temp_files.get(file_id)
    if file_path and os.path.exists(file_path):
        # Guess the MIME type based on the file extension
        mimetype, _ = mimetypes.guess_type(file_path)
        if mimetype is None:
            # Default to 'application/octet-stream' if MIME type cannot be guessed
            mimetype = 'video/mp4'
        return send_file(file_path, mimetype=mimetype)
    else:
        return 'File not found', 404


@blueprint.route('/download/<path:path>')
@login_required
def download_file(path):
    return send_file(path, as_attachment=True)


@blueprint.route('/<template>')
@login_required
def route_template(template):
    # try:
    if not template.endswith('.html'):
        template += '.html'

    # Detect the current page
    segment = get_segment(request)

    # Serve the file (if exists) from app/templates/home/FILE.html
    return render_template("home/" + template, segment=segment)


"""
    except TemplateNotFound:
        return render_template('home/page-404.html'), 404
    except:
        return render_template('home/page-500.html'), 500
"""


def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None
