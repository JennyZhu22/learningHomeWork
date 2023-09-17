
import gradio as gr
import random
import time
import os


from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

##格式转换
def format_array_to_tuples(input_array):
    formatted_tuples = []
    for entry in input_array:
        if len(entry) == 2:
            formatted_tuples.append((entry[0], entry[1]))
    return formatted_tuples


#headphone

def initialize_sales_bot(vector_store_dir: str="headphone"):
    vect_store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), vector_store_dir)
    db = FAISS.load_local(vect_store_path, OpenAIEmbeddings())
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Build prompt for rephrase questions
    template =  """Please extract key product information from chat history, like product model no, brand and so on, and then use it in the following rephrase 
                   Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
                   Chat History:
                   {chat_history}
                   Follow Up Input: {question}
                   Standalone question:
                """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    global SALES_BOT    
    SALES_BOT = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(search_type="similarity_score_threshold",
                                                                 search_kwargs={"score_threshold": 0.8}),
                                           verbose=True,
                                           return_generated_question=True,
                                           condense_question_prompt=QA_CHAIN_PROMPT
                                          # combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
                                          )
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True
 
    
    pastHistory = []

    if len(history) > 0 :
       pastHistory = format_array_to_tuples(history)

    print("history information1:")
    print(pastHistory)

    print("history information2:")
    print(history)
   

    ans = SALES_BOT({"question": message, "chat_history": pastHistory})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['answer']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["answer"]
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="耳机销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化耳机销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
