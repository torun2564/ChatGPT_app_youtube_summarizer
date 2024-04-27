import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader

def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="☺"
    )
    st.header("Youtube Summarizer ☺")
    st.sidebar.title("Options")
    st.session_state.costs = []
    
        
def select_model():
    model = st.sidebar.radio("Choose a model: ", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"
        
    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0, 刻み幅は0.01
    temperature = st.sidebar.slider("Temperature: ", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    
    return ChatOpenAI(temperature=temperature, model_name=model_name)

def select_n_chars():
    n_chars = st.sidebar.slider("Number of summary characters: ", min_value=20, max_value=10000, value=500, step=10)
    return n_chars

def get_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url

def get_document(url):
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True, # タイトルや再生数も取得できる
            language=["en", "ja"], # 英語->日本語の優先順位で字幕を取得
        )
        return loader.load()


def summarize(llm, docs, n_chars=300):
    partial_variables = {"n_chars": n_chars}
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.

============
    
{text}

============

ここから日本語で書いてね
必ず{n_chars}文字以内で簡潔にまとめること:
""" 
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"], partial_variables=partial_variables)
    
    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
        
    return response["output_text"], cb.total_cost
    

def main():
    init_page()    
    llm = select_model()
    
    container = st.container()
    response_container = st.container()
    n_chars = select_n_chars()
    
    with container:
        url = get_url_input()
        if url:
            document = get_document(url)
            with st.spinner("ChatGPT is typing ..."):
                output_text, cost = summarize(llm, document, n_chars)
            st.session_state.costs.append(cost)
        else:
            output_text = None
                
    if output_text:
        with response_container:
            st.markdown("## Summary")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)
            
    # コストの表示
    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")
        
if __name__ == "__main__":
    main()
