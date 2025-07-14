import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Judul aplikasi
st.title("Aplikasi Swap Face Foto/Video")

# Fungsi untuk memuat model dengan caching
@st.cache_resource
def load_models():
    try:
        from insightface.app import FaceAnalysis
        model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
        
        from insightface.model_zoo import get_model
        swapper = get_model('inswapper_128.onnx', download=True, download_zip=True)
        return model, swapper
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

model, swapper = load_models()

# Fungsi untuk swap wajah
def swap_face(img, source_face, target_face):
    if swapper is None:
        st.error("Model swapper tidak tersedia")
        return img
    return swapper.get(img, target_face, source_face, paste_back=True)

# Fungsi untuk deteksi wajah
def detect_faces(img):
    if model is None:
        st.error("Model deteksi wajah tidak tersedia")
        return []
    return model.get(img)

# UI untuk pilihan mode
mode = st.radio("Pilih Mode:", ("Foto", "Video"), horizontal=True)

if mode == "Foto":
    st.subheader("Swap Wajah pada Foto")
    
    col1, col2 = st.columns(2)
    with col1:
        source_img = st.file_uploader("Upload Gambar Sumber Wajah", type=["jpg", "jpeg", "png"])
    with col2:
        target_img = st.file_uploader("Upload Gambar Target", type=["jpg", "jpeg", "png"])
    
    if source_img and target_img and model and swapper:
        try:
            source_img = Image.open(source_img).convert("RGB")
            target_img = Image.open(target_img).convert("RGB")
            
            source_np = np.array(source_img)
            target_np = np.array(target_img)
            
            source_faces = detect_faces(source_np)
            target_faces = detect_faces(target_np)
            
            if not source_faces:
                st.error("Tidak ada wajah terdeteksi pada gambar sumber!")
            elif not target_faces:
                st.error("Tidak ada wajah terdeteksi pada gambar target!")
            else:
                source_face = source_faces[0]
                target_face = target_faces[0]
                
                if st.button("Swap Wajah"):
                    result_img = swap_face(target_np, source_face, target_face)
                    st.image(result_img, caption="Hasil Swap Wajah", use_column_width=True)
                    
                    # Konversi ke format yang bisa di-download
                    result_pil = Image.fromarray(result_img)
                    st.download_button(
                        label="Download Hasil",
                        data=cv2.imencode('.jpg', result_img)[1].tobytes(),
                        file_name="hasil_swap.jpg",
                        mime="image/jpeg"
                    )
        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")

else:  # Mode Video
    st.subheader("Swap Wajah pada Video")
    
    col1, col2 = st.columns(2)
    with col1:
        source_img = st.file_uploader("Upload Gambar Sumber Wajah", type=["jpg", "jpeg", "png"])
    with col2:
        target_video = st.file_uploader("Upload Video Target", type=["mp4", "mov"])
    
    if source_img and target_video and model and swapper:
        try:
            source_img = Image.open(source_img).convert("RGB")
            source_np = np.array(source_img)
            source_faces = detect_faces(source_np)
            
            if not source_faces:
                st.error("Tidak ada wajah terdeteksi pada gambar sumber!")
            else:
                source_face = source_faces[0]
                
                # Simpan video sementara
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(target_video.read())
                tfile.close()
                
                # Proses video
                cap = cv2.VideoCapture(tfile.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # File output
                output_path = "output.mp4"
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
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    target_faces = detect_faces(frame_rgb)
                    
                    if target_faces:
                        result_frame = swap_face(frame_rgb, source_face, target_faces[0])
                        result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    else:
                        result_frame_bgr = frame
                    
                    out.write(result_frame_bgr)
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Memproses: {frame_count}/{total_frames} frame")
                
                cap.release()
                out.release()
                os.unlink(tfile.name)
                
                st.success("Video selesai diproses!")
                st.video(output_path)
                
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="Download Video Hasil",
                        data=f,
                        file_name="hasil_swap.mp4",
                        mime="video/mp4"
                    )
                
                os.remove(output_path)
        except Exception as e:
            st.error(f"Terjadi error saat memproses video: {str(e)}")
            finally:
                if 'tfile' in locals() and os.path.exists(tfile.name):
                    os.unlink(tfile.name)
                if 'output_path' in locals() and os.path.exists(output_path):
                    os.remove(output_path)
