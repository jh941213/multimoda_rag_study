import streamlit as st
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
import cv2
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.callbacks import StreamlitCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import json
from crewai import Agent, Task, Crew
import base64

# 환경 변수 로드
load_dotenv()

# API 키 설정
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# 비디오 처리 함수들
def upload_and_process_file(file_path):
    print(f"Uploading file: {file_path}...")
    video_file = genai.upload_file(path=file_path)
    
    # 파일 처리 상태 확인
    while video_file.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(10)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError(f"File processing failed: {video_file.state.name}")
    
    print(f"\nCompleted upload: {video_file.uri}")
    return video_file

# LLM 요청 함수
def generate_content_from_video(video_file, prompt, model_name="gemini-1.5-flash-001", timeout=600):
    print("Making LLM inference request...")
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content([video_file, prompt], request_options={"timeout": timeout})
    return response

def extract_video_segment(input_video, start_time, end_time, output_folder, food_name):
    cap = cv2.VideoCapture(input_video)
    
    # 비디오 속성 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 시작 및 종료 프레임 계산
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # 또는 'avc1', 'H264' 등을 시도해볼 수 있습니다.
    output_filename = f"{food_name}_{start_time}_{end_time}.mp4"
    output_path = os.path.join(output_folder, output_filename)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Could not open output video file: {output_path}")
        return None
    
    # 시작 프레임으로 이동
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 프레임 추출 및 저장
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_num}")
            break
        out.write(frame)
    
    # 리소스 해제
    cap.release()
    out.release()
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Successfully extracted video segment: {output_path}")
        return output_path
    else:
        print(f"Failed to extract video segment or output file is empty: {output_path}")
        return None

def create_metadata_table(extracted_videos, food_name):
    metadata = []
    for video in extracted_videos:
        metadata.append({
            'video_path': video,
            'food_name': food_name
        })
    
    df = pd.DataFrame(metadata)
    df.to_csv('metadata.csv', index=False)
    print("메타데이터가 저장되었습니다.")
    return df

# Pydantic 모델
class Seconds(BaseModel):
    start: int = Field(description="음식 시작 시간(초)")
    end: int = Field(description="음식 종료 시간(초)")

class VideoParser(BaseModel):
    time: list[Seconds] = Field(description="음식 시간(초)")
    food_name: list[str] = Field(description="음식 이름")

model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0)

@st.cache_resource
def process_video_and_create_qa(file_path):
    video_file = upload_and_process_file(file_path)
    
    prompt = '''
    해당 영상에서 당신은 먹는 시간의 시작과 끝을 출력하고, 먹는 메뉴도 출력해야 합니다. 영어로 출력하세요.
    [출력예시]
    [1:01,1:31], bulgogi
    [2:01,2:31], bibimbab
    [3:01,3:31], kimchijjigae
    '''
    
    response = generate_content_from_video(video_file, prompt)
    
    parser = JsonOutputParser(pydantic_object=VideoParser)
    prompt_template = PromptTemplate(
        template="사용자 쿼리에 답하세요.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt_template | model | parser
    parsed_response = chain.invoke({"query": response.text})
    
    output_folder = "extract_video"
    os.makedirs(output_folder, exist_ok=True)
    extracted_videos = []
    
    # 각 세그먼트에 대해 비디오를 추출하고 각 세그먼트에 맞는 음식 이름을 적용
    for idx, segment in enumerate(parsed_response['time']):
        # 각 세그먼트의 음식 이름을 인덱스로 가져옵니다.
        food_name = parsed_response['food_name'][idx]  # 음식 이름 리스트에서 인덱스를 사용
        extracted_path = extract_video_segment(file_path, segment['start'], segment['end'], output_folder, food_name)
        extracted_videos.append(extracted_path)
    
    # 메타데이터 생성 (음식 이름이 리스트로 제대로 넘어가도록 수정)
    create_metadata_table(extracted_videos, parsed_response['food_name'])

    
    loader = CSVLoader(file_path='metadata.csv', encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings)

    food_names = parsed_response['food_name']
    print(food_names)

    system_prompt = f"""당신은 음식 영상 분석 전문가입니다. 주어진 정보를 바탕으로 사용자의 질문에 정확하고 상세하게 답변해주세요. 
    context에는 유저가 질문한 음식에 관한 정보가 있습니다. 
    만약 질문에 대한 답변을 할 수 없는 경우, 정직하게 모른다고 말하고 가능한 경우 관련된 정보를 제공해주세요.
    출력은 최대한 간결하게 음식에 맞는 맛있는 표현을 생각해서 출력해주세요.
    
    영상에서 확인된 음식 목록: {', '.join(food_names)}"""
    
    qa_prompt = PromptTemplate(
        template=system_prompt + "\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=model, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt}
    )
    
    return qa


def find_relevant_video(query, vectorstore):
    # 쿼리와 가장 관련성 높은 문서 검색
    docs = vectorstore.similarity_search(query, k=1)
    print("관련있는 문서: ", docs)  # 문서가 검색되었는지 확인
    
    if docs:
        # 첫 번째 문서에서 page_content 추출
        content = docs[0].page_content
        print("문서의 내용: ", content)  # page_content 내용 확인
        
        # video_path와 food_name을 추출하기 위한 간단한 파싱
        lines = content.split('\n')
        video_path = None
        food_name = None
        
        for line in lines:
            if line.startswith('video_path:'):
                video_path = line.split('video_path: ')[1].strip()
            elif line.startswith('food_name:'):
                food_name = line.split('food_name: ')[1].strip()
        
        print("추출된 video_path: ", video_path)
        print("추출된 food_name: ", food_name)
        
        return video_path, food_name
    
    return None, None

# Streamlit 앱
def main():
    st.title("음식 영상 분석 및 채팅 앱")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "동영상을 업로드하고 분석 결과를 확인한 후 질문해주세요."}]
    
    if "qa" not in st.session_state:
        st.session_state["qa"] = None
    
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    
    if "extracted_videos" not in st.session_state:
        st.session_state["extracted_videos"] = []

    # 사이드바에 파일 업로더 추가
    uploaded_file = st.sidebar.file_uploader("동영상 파일을 업로드하세요", type=["mp4"])

    if uploaded_file and st.session_state["qa"] is None:
        with st.spinner("동영상을 처리하고 있습니다..."):
            # 임시 파일로 저장
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state["qa"] = process_video_and_create_qa("temp_video.mp4")
            
            # 벡터 스토어 초기화
            loader = CSVLoader(file_path='metadata.csv', encoding='utf-8')
            documents = loader.load()
            embeddings = OpenAIEmbeddings()
            st.session_state["vectorstore"] = Chroma.from_documents(documents, embeddings)
            
            st.session_state.messages.append({"role": "assistant", "content": "영상 분석이 완료되었습니다. 질문해주세요."})

    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖").write(msg["content"])

    # 채팅 인터페이스
    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)

        if st.session_state["qa"] and st.session_state["vectorstore"]:
            # Crew 생성 및 실행
            research_agent = Agent(
                role='Video Recommender',
                goal='Determine if the user wants to watch a video based on the query',
                backstory="""You are an AI agent responsible for analyzing the user's query
                and deciding whether or not they want to watch a video.""",
                verbose=True
            )
            task = Task(
                description=f'{prompt}에 대하여 동영상을 틀지 말지 결정하는 작업',
                expected_output='0 if the user does not want to watch a video, 1 if the user wants to watch a video',
                agent=research_agent,
            )
            crew = Crew(
                agents=[research_agent],
                tasks=[task],
                verbose=True
            )
            result = crew.kickoff(inputs=dict(query=prompt))
            
            show_video = int(result.raw) == 1

            with st.chat_message("assistant", avatar="🤖"):
                st_callback = StreamlitCallbackHandler(st.container())
                response_placeholder = st.empty()
                full_response = ""

                for chunk in st.session_state["qa"].stream({"query": prompt}, callbacks=[st_callback]):
                    if isinstance(chunk, dict) and 'result' in chunk:
                        full_response += chunk['result']
                        response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            if show_video:
                # 쿼리에 기반하여 관련 비디오 찾기
                video_path, food_name = find_relevant_video(prompt, st.session_state["vectorstore"])
                
                if video_path:
                    # 절대 경로로 변환
                    absolute_path = os.path.abspath(video_path)
                    print("비디오 파일의 절대 경로:", absolute_path)
                    
                    # 비디오 출력 (크기 조절 없이)
                    st.video(absolute_path)
                    st.write(f"재생 중인 음식: {food_name}")
                    
                    # 파일이 제대로 로드되었는지 확인
                    if os.path.exists(absolute_path):
                        print("비디오 파일이 성공적으로 로드되었습니다.")
                    else:
                        print("비디오 파일을 찾을 수 없습니다.")
                else:
                    st.warning("쿼리와 관련된 비디오를 찾을 수 없습니다.")
        else:
            st.session_state.messages.append({"role": "assistant", "content": "먼저 동영상을 업로드해주세요."})
            st.chat_message("assistant", avatar="🤖").write("먼저 동영상을 업로드해주세요.")

if __name__ == "__main__":
    main()
