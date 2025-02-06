from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import shutil
from flask_cors import CORS
import argparse
import numpy as np

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('log.txt')
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

UPLOAD_FOLDER = '/root/servers/IMV_SEMANTIC_SEARCH_YOLO_SERVER/INDEX/index_hetz4/output/INDEX/hetz1_index'
DEST_FOLDER = '/root/crop-images-coordinates/static/images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
image_data = []

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            logger.info(f"New image detected: {event.src_path}")
            image_data.append({'file': os.path.basename(event.src_path), 'crop_coords': [0, 0, 0, 0]})

def start_watcher():
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path=UPLOAD_FOLDER, recursive=False)
    observer.start()
    logger.info("Started monitoring image directory.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info("Stopped monitoring image directory.")
    observer.join()

def load_images():
    logger.info("Loading specified images.")
    os.makedirs(DEST_FOLDER, exist_ok=True)

    image_path = os.getenv('IMAGE_PATH', '/root/servers/IMV_SEMANTIC_SEARCH_YOLO_SERVER/INDEX/index_hetz4/output/INDEX/hetz1_index/video/POE-Selina-Cetate-2-20240926171425-20240926171642/image_181-20250120_123954.jpg')

    for image_path in [image_path] if image_path is not list else image_path:
        if os.path.exists(image_path) and image_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            filename = os.path.basename(image_path)
            shutil.copy2(image_path, os.path.join(DEST_FOLDER, filename))
            logger.info(f"Loaded image: {filename}")
            image_data.append({'file': filename, 'crop_coords': [0, 0, 0, 0]})

@app.route('/')
def index():
    logger.info(f"Rendering index with {len(image_data)} images.")
    return render_template('index.html', images=image_data)

def crop_image(img_path, crop_coords):

    try:
        with Image.open(img_path) as img_obj:
            img_width, img_height = img_obj.size
            x1, y1 = crop_coords[0]['x'] * img_width, crop_coords[0]['y'] * img_height
            x2, y2 = crop_coords[1]['x'] * img_width, crop_coords[1]['y'] * img_height
            x3, y3 = crop_coords[2]['x'] * img_width, crop_coords[2]['y'] * img_height
            x4, y4 = crop_coords[3]['x'] * img_width, crop_coords[3]['y'] * img_height
            left, top, right, bottom = min(x1, x3), min(y1, y2), max(x2, x4), max(y3, y4)

            cropped_image = img_obj.crop((left, top, right, bottom))
            cropped_image.save(f"cropped_{os.path.basename(img_path)}")
            logger.info(f"Image cropped and saved as cropped_{os.path.basename(img_path)}")

            return [left,top,right,bottom]
    except Exception as e:
        logger.error(f"Error cropping image {img_path}: {str(e)}")
        raise


@app.route('/crop/<image>', methods=['POST'])
def set_crop(image):
    data = request.json
    crop_coords_ = data.get('crop_coords')
    if len(crop_coords_) != 4:
        return jsonify({'error': 'Invalid crop coordinates format'}), 400

    for img in image_data:
        if img['file'] == image:
            try:
                img_path = os.path.join(DEST_FOLDER, image)
                crop_area = crop_image(img_path, crop_coords_)
                img['crop_coords_'] = crop_coords_
                img['cropped_area'] = crop_area

                return jsonify({'success': True, 'crop_coords_': img['crop_coords_'], 'cropped_area': img['cropped_area']}), 200
            except Exception as e:
                return jsonify({'error': f"Error processing image: {str(e)}"}), 500
    
    return jsonify({'error': 'Image not found'}), 404

if __name__ == "__main__":
    load_images()
    threading.Thread(target=start_watcher, daemon=True).start()
    logger.info("Starting Flask application.")
    app.run(host='0.0.0.0', debug=True, port=5050)
