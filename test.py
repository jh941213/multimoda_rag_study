import streamlit as st

# 동영상 파일 경로
video_path = "/Users/kdb/Desktop/rag_study/video/내가 엘든링 7번 깬 비법.f614.mp4"

# 동영상 파일을 바이너리로 읽기
with open(video_path, 'rb') as video_file:
    video_bytes = video_file.read()

# 스트림릿에서 동영상 출력
st.title("동영상 출력 예시")
st.video(video_bytes)
