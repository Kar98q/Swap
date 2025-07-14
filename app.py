import streamlit as st
import cv2
import numpy as np
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
import tempfile
import os

# Judul aplikasi
st.title("Aplikasi Swap Face Foto/Video")

# Inisialisasi model face detection dan swapping
@st.cache_resource
def load_model():
    model = FaceAnalysis(name='buffalo_l')
    model.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
    return model, swapper

model, swapper = load_model()

# Fungsi untuk swap wajah pada gambar
def swap_face_image(img, source_face, target_face):
    return swapper.get(img, target_face, source_face, paste_back=True)

# Fungsi untuk deteksi wajah
def detect_faces(img):
    faces = model.get(img)
    return faces

# Fungsi untuk menggambar bounding box
def draw_bboxes(img, faces):
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    return img

# Pilihan mode: Foto atau Video
mode = st.radio("Pilih Mode:", ("Foto", "Video"))

if mode == "Foto":
    st.subheader("Swap Wajah pada Foto")
    
    # Upload gambar sumber dan target
    col1, col2 = st.columns(2)
    with col1:
        source_img = st.file_uploader("Upload Gambar Sumber Wajah", type=["jpg", "jpeg", "png"])
    with col2:
        target_img = st.file_uploader("Upload Gambar Target", type=["jpg", "jpeg", "png"])
    
    if source_img and target_img:
        # Baca gambar
        source_img = Image.open(source_img).convert("RGB")
        target_img = Image.open(target_img).convert("RGB")
        
        # Konversi ke numpy array
        source_np = np.array(source_img)
        target_np = np.array(target_img)
        
        # Deteksi wajah
        source_faces = detect_faces(source_np)
        target_faces = detect_faces(target_np)
        
        if len(source_faces) == 0:
            st.error("Tidak ada wajah terdeteksi pada gambar sumber!")
        elif len(target_faces) == 0:
            st.error("Tidak ada wajah terdeteksi pada gambar target!")
        else:
            # Pilih wajah pertama dari gambar sumber
            source_face = source_faces[0]
            
            # Tampilkan preview
            st.subheader("Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.image(draw_bboxes(source_np.copy(), source_faces), caption="Gambar Sumber", use_column_width=True)
            with col2:
                st.image(draw_bboxes(target_np.copy(), target_faces), caption="Gambar Target", use_column_width=True)
            
            # Pilih wajah target
            if len(target_faces) > 1:
                face_idx = st.selectbox("Pilih wajah target untuk ditukar:", range(len(target_faces)), format_func=lambda x: f"Wajah {x+1}")
                target_face = target_faces[face_idx]
            else:
                target_face = target_faces[0]
            
            # Lakukan swap wajah
            if st.button("Swap Wajah"):
                result_img = swap_face_image(target_np, source_face, target_face)
                st.subheader("Hasil")
                st.image(result_img, caption="Hasil Swap Wajah", use_column_width=True)
                
                # Download hasil
                result_pil = Image.fromarray(result_img)
                st.download_button(
                    label="Download Hasil",
                    data=cv2.imencode('.jpg', result_img)[1].tobytes(),
                    file_name="hasil_swap.jpg",
                    mime="image/jpeg"
                )

else:  # Mode Video
    st.subheader("Swap Wajah pada Video")
    
    # Upload gambar sumber dan video target
    col1, col2 = st.columns(2)
    with col1:
        source_img = st.file_uploader("Upload Gambar Sumber Wajah", type=["jpg", "jpeg", "png"])
    with col2:
        target_video = st.file_uploader("Upload Video Target", type=["mp4", "mov", "avi"])
    
    if source_img and target_video:
        # Baca gambar sumber
        source_img = Image.open(source_img).convert("RGB")
        source_np = np.array(source_img)
        source_faces = detect_faces(source_np)
        
        if len(source_faces) == 0:
            st.error("Tidak ada wajah terdeteksi pada gambar sumber!")
        else:
            source_face = source_faces[0]
            
            # Simpan video sementara
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(target_video.read())
            
            # Baca video
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Tampilkan preview
            st.subheader("Preview")
            st.image(draw_bboxes(source_np.copy(), source_faces), caption="Gambar Sumber", use_column_width=True)
            
            # Buat video output
            output_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Konversi warna BGR ke RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Deteksi wajah
                target_faces = detect_faces(frame_rgb)
                
                if len(target_faces) > 0:
                    # Swap wajah pertama yang terdeteksi
                    target_face = target_faces[0]
                    result_frame = swap_face_image(frame_rgb, source_face, target_face)
                else:
                    result_frame = frame_rgb
                
                # Konversi kembali ke BGR untuk video output
                result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                out.write(result_frame_bgr)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Memproses frame {frame_count}/{total_frames}")
            
            cap.release()
            out.release()
            os.unlink(tfile.name)
            
            st.success("Video selesai diproses!")
            
            # Tampilkan video hasil
            st.subheader("Hasil Video")
            st.video(output_path)
            
            # Download hasil
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Video Hasil",
                    data=f,
                    file_name="hasil_swap.mp4",
                    mime="video/mp4"
                )
            
            # Hapus file sementara
            os.remove(output_path)
