import streamlit as st
from PIL import Image
import io
import os
import insightface
import numpy as np
import time
import sys
sys.path.append("code\SimSwap") #todo : 경로 수정

from myFunc import get_embedding, load_models

def image_to_bytes(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format = "JPEG")
    return img_byte_array.getvalue()

@st.cache_resource
def load_insight_model():
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0)
    return model

@st.cache_data
def load_embedding_vectors(path):
    dir = os.path.join(path, "embedding_vectors.npz")
    ref_vectors = np.load(dir)['data']
    return ref_vectors

def get_reference_images(domain, gender, src):
    
    # # dataset path
    # path = "serving/data" # todo : domain + gender에 따라 path 설정
    # files = os.listdir(path)
    # files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # # ref 이미지 embedding vector 읽어오기
    # ref_vectors = load_embedding_vectors(path)

    # # src embedding vector 구하기
    # model = load_insight_model()
    # src = np.array(Image.open(src))
    # src_vector = model.get(src)[0]['embedding']

    # # cosine similarity 
    # similarities = np.dot(ref_vectors, src_vector) / (np.linalg.norm(ref_vectors, axis=1) * np.linalg.norm(src_vector))

    # # 가장 유사한 이미지 3개 
    # top3_img_idx = np.argsort(similarities)[::-1][:3]

    # results = []
    # for idx in top3_img_idx:
    #     file = files[idx]
    #     img = Image.open(os.path.join(path, file))
    #     results.append(img)

    results = []
    results.append(Image.open('serving/data/id_f_1.jpg'))
    results.append(Image.open('serving/data/id_f_2.jpg'))
    results.append(Image.open('serving/data/id_f_3.jpg'))
    return results

def get_result_images(src, refs):
    
    # todo : simswap inference

    results = []

    return refs

def reupload_callback():
    del st.session_state["src"]
    st.session_state.page = "upload_page"

def home_callback():
    del st.session_state["src"]
    st.session_state.page = "home_page"

def show_resultPage():

    # side bar
    with st.sidebar:
        st.markdown("""
            <center>
            👔스타일 선택

            ↓

            📂 사진 업로드

            ↓

            <span style="font-family: 'Arial'; font-size: 18px; font-weight: bold; color: #ff0000;">🎉 결과 확인</span>
            </center>
        """, unsafe_allow_html=True)

    # 설명
    description = st.container()
    description.write("결과를 확인하고 원하는 이미지를 다운할 수 있습니다. ")
    
    st.divider()

    # test
    app, model = load_models()
    embedding_vec = get_embedding(app, model, st.session_state.src)[0]
    st.write(embedding_vec)

    # with st.spinner('wait...'):
    #     time.sleep(1)

    # # reference 이미지
    # refs = get_reference_images(st.session_state.domain, st.session_state.gender, st.session_state.src)

    # # result 이미지
    # results = get_result_images(st.session_state.src, refs)

    # cols = st.columns(3)
    # for i in range(3):
    #     col = cols[i]
        
    #     # image
    #     col.image(results[i])

    #     # download button
    #     img_byte = image_to_bytes(results[i])
    #     col.download_button("다운로드", data=img_byte, file_name=f"image{i}.jpg", use_container_width=True)

    # button
    buttons = st.container()
    reupload_button = buttons.button("이미지 다시 업로드하기", use_container_width=True, on_click=reupload_callback)
    home_button = buttons.button("처음으로 돌아가기", use_container_width=True, on_click=home_callback)




