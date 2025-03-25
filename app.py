# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from traffic_detection import process_video
import threading
import time
import tempfile
from moviepy.editor import VideoFileClip

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Định nghĩa thư mục lưu video đầu ra
OUTPUT_FOLDER = 'static/videos'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Tạo thư mục nếu chưa tồn tại
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected. Please choose a video file.')
        return redirect(request.url)
    
    if file:
        # Tạo file tạm thời để lưu video đầu vào
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
            file.save(temp_input)
            temp_input_path = temp_input.name

        # Tạo tên file đầu ra duy nhất
        output_filename = f"processed_{int(time.time())}_{file.filename}"
        temp_output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"temp_{output_filename}")
        final_output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Hàm xử lý video, chuyển đổi, và lưu vào static/videos
        def process_and_cleanup(temp_input_path, temp_output_path, final_output_path):
            try:
                # Xử lý video bằng traffic_detection.py (với codec mp4v)
                process_video(temp_input_path, temp_output_path)
                
                # Chuyển đổi video sang codec H.264 bằng moviepy
                clip = VideoFileClip(temp_output_path)
                clip.write_videofile(final_output_path, codec='libx264', audio=False)
                clip.close()
                
                # Xóa file tạm (video với codec mp4v)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                    print(f"Deleted temp file: {temp_output_path}")
            except Exception as e:
                print(f"Error during processing or conversion: {str(e)}")
            finally:
                # Xóa file đầu vào tạm
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    print(f"Deleted temp file: {temp_input_path}")
        
        # Chạy xử lý video trong luồng riêng
        try:
            threading.Thread(target=process_and_cleanup, args=(temp_input_path, temp_output_path, final_output_path)).start()
        except Exception as e:
            flash(f'Error starting video processing: {str(e)}')
            return redirect(request.url)
        
        # Trả về thông báo đang xử lý với tên file đầu ra
        return render_template('index.html', processing=True, video_url=output_filename)

    flash('An error occurred while uploading the file.')
    return redirect(request.url)

@app.route('/check_status/<filename>')
def check_status(filename):
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(output_path):
        return jsonify({'status': 'completed'})
    return jsonify({'status': 'processing'})

if __name__ == '__main__':
    app.run(debug=True)