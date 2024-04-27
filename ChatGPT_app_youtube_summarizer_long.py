import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="☺"
    )
    st.header("Youtube Summarizer ☺")
    st.sidebar.title("Options")
    st.session_state.costs = []
    
        
def select_model():
    model = st.sidebar.radio("Choose a model: ", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k-0613"
    else:
        st.session_state.model_name = "gpt-4"
    
    # 300: 要約指示のtoken数(要約指示を書くためのtoken数の分を除外しておかないとエラーが出るため）
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300   
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url

def select_n_chars():
    n_chars = st.sidebar.slider("Number of summary characters: ", min_value=20, max_value=10000, value=1000, step=10)
    return n_chars


def get_document(url):
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True, # タイトルや再生数も取得できる
            language=["en", "ja"], # 英語->日本語の優先順位で字幕を取得
        )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.model_name,
            chunk_size=st.session_state.max_token,
            chunk_overlap=0,
        )
        return loader.load_and_split(text_splitter=text_splitter)


def summarize(llm, docs, n_chars):
    partial_variables = {"n_chars": n_chars}
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.
    
{text}

ここから日本語で書いてね
必ず{n_chars}文字以内でまとめること:
""" 
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"], partial_variables=partial_variables)

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            verbose=True,
            map_prompt=PROMPT,
            combine_prompt=PROMPT
        )
        response = chain(
            {
                "input_documents": docs,
                # token_max を指示しないと、GPT3.5など通常の
                # モデルサイズに合わせた内部処理になってしまうので注意
                "token_max": st.session_state.max_token
            },
            return_only_outputs=True
        )
        
    return response['output_text'], cb.total_cost

    

def main():
    init_page()
    n_chars = select_n_chars()
    llm = select_model()
     
    container = st.container()
    response_container = st.container()
    
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
