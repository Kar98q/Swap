import os
import sys
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import requests
from tqdm import tqdm
import subprocess

# Konfigurasi awal
st.set_page_config(page_title="Aplikasi Swap Face", layout="wide")

# Fungsi untuk memastikan dependensi terinstall
def install_dependencies():
    dependencies = {
        'insightface': 'insightface==0.7.3',
        'onnxruntime': 'onnxruntime==1.15.1',
        'opencv': 'opencv-python-headless==4.8.1.78'
    }
    
    for package, version in dependencies.items():
        try:
            __import__(package)
        except ImportError:
            st.warning(f"Memasang {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", version], check=True)

install_dependencies()

# Fungsi untuk download model
def download_model(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

# Load model dengan caching
@st.cache_resource
def load_models():
    try:
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        
        # Download model jika belum ada
        model_path = "inswapper_128.onnx"
        if not os.path.exists(model_path):
            st.warning("Mengunduh model swap wajah...")
            download_model(
                "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip", 
                "buffalo_l.zip"
            )
            # Ekstrak model (dalam implementasi nyata perlu menambahkan ekstraksi zip)
        
        model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
        
        swapper = get_model(model_path, download=False)
        return model, swapper
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None

model, swapper = load_models()

# Fungsi untuk swap wajah
def swap_face(img, source_face, target_face):
    if swapper is None:
        raise ValueError("Model swapper tidak tersedia")
    return swapper.get(img, target_face, source_face, paste_back=True)

# Fungsi deteksi wajah
def detect_faces(img):
    if model is None:
        raise ValueError("Model deteksi wajah tidak tersedia")
    return model.get(img)

# UI Utama
st.title("🔄 Aplikasi Swap Wajah Foto/Video")
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #008CBA;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Pilihan mode
mode = st.radio("Pilih Mode:", ["Foto", "Video"], horizontal=True, label_visibility="collapsed")

if mode == "Foto":
    st.subheader("🔁 Swap Wajah pada Foto")
    
    col1, col2 = st.columns(2)
    with col1:
        source_img = st.file_uploader("Upload Gambar Sumber Wajah", type=["jpg", "jpeg", "png"], key="src_img")
    with col2:
        target_img = st.file_uploader("Upload Gambar Target", type=["jpg", "jpeg", "png"], key="tgt_img")
    
    if source_img and target_img:
        try:
            with st.spinner("Memproses gambar..."):
                # Baca gambar
                source_img = Image.open(source_img).convert("RGB")
                target_img = Image.open(target_img).convert("RGB")
                
                source_np = np.array(source_img)
                target_np = np.array(target_img)
                
                # Deteksi wajah
                source_faces = detect_faces(source_np)
                target_faces = detect_faces(target_np)
                
                if not source_faces:
                    st.error("🚨 Tidak ada wajah terdeteksi pada gambar sumber!")
                elif not target_faces:
                    st.error("🚨 Tidak ada wajah terdeteksi pada gambar target!")
                else:
                    st.success(f"✔️ Terdeteksi {len(source_faces)} wajah sumber dan {len(target_faces)} wajah target")
                    
                    # Pilih wajah
                    source_face = source_faces[0]
                    target_face = target_faces[0]  # Ambil wajah pertama
                    
                    # Tombol swap
                    if st.button("🔀 Swap Wajah", type="primary"):
                        with st.spinner("Melakukan swap wajah..."):
                            result_img = swap_face(target_np, source_face, target_face)
                            
                            # Tampilkan hasil
                            st.subheader("🎭 Hasil Swap Wajah")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(target_np, caption="Sebelum", use_column_width=True)
                            with col2:
                                st.image(result_img, caption="Sesudah", use_column_width=True)
                            
                            # Download hasil
                            result_bytes = cv2.imencode('.jpg', result_img)[1].tobytes()
                            st.download_button(
                                label="⬇️ Download Hasil",
                                data=result_bytes,
                                file_name="hasil_swap.jpg",
                                mime="image/jpeg"
                            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

else:  # Mode Video
    st.subheader("🎥 Swap Wajah pada Video")
    
    col1, col2 = st.columns(2)
    with col1:
        source_img = st.file_uploader("Upload Gambar Sumber Wajah", type=["jpg", "jpeg", "png"], key="vid_src")
    with col2:
        target_video = st.file_uploader("Upload Video Target", type=["mp4", "mov"], key="vid_tgt")
    
    if source_img and target_video:
        try:
            with st.spinner("Mempersiapkan proses..."):
                # Baca gambar sumber
                source_img = Image.open(source_img).convert("RGB")
                source_np = np.array(source_img)
                source_faces = detect_faces(source_np)
                
                if not source_faces:
                    st.error("🚨 Tidak ada wajah terdeteksi pada gambar sumber!")
                else:
                    source_face = source_faces[0]
                    
                    # Simpan video sementara
                    temp_dir = tempfile.mkdtemp()
                    video_path = os.path.join(temp_dir, "input_video.mp4")
                    with open(video_path, "wb") as f:
                        f.write(target_video.read())
                    
                    # Persiapan video output
                    output_path = os.path.join(temp_dir, "output_video.mp4")
                    
                    # Proses video
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    # Progress bar
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    processing_container = st.empty()
                    
                    frame_count = 0
                    success = True
                    
                    while success and frame_count < total_frames:
                        success, frame = cap.read()
                        if not success:
                            break
                        
                        # Proses frame
                        frame_rgb = cv2.cvtColor(frame, cv2.CAP_PROP_CONVERT_RGB)
                        target_faces = detect_faces(frame_rgb)
                        
                        if targe
