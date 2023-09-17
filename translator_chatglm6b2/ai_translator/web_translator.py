import gradio as gr
import random
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import ArgumentParser, ConfigLoader, LOG
from model import GLMModel, OpenAIModel
from translator import PDFTranslator

LANGS = ["英文", "中文"]

def translate(input_pdf,source_lang,target_lang):
      # 获取文件属性信息
    file_path = input_pdf.name  # 文件路径
    file_name = os.path.basename(file_path)  # 文件名
    file_size = os.path.getsize(file_path)  # 文件大小（字节）
    

    print("上传的文件路径:", file_path)
    print("上传的文件名:", file_name)
    print("上传的文件大小:", file_size, "字节")

    model_url = "http://127.0.0.1:8000"
    model = GLMModel(model_url,36000)
     # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    translator = PDFTranslator(model)
    output_filepath = translator.translate_pdf(file_path, "PDF")
    
    print("file path in last step ")

    print(output_filepath)
  
    print(source_lang)
    print(target_lang)
    return output_filepath


def launch_gradio():
    demo = gr.Interface(
      fn=translate,
      inputs=[
        gr.File(label='Input'),
        gr.components.Dropdown(label="Source Language", choices=LANGS),
        gr.components.Dropdown(label="Target Language", choices=LANGS)
        ],
      outputs=[gr.outputs.File(label='Output PDF')],       
      allow_flagging="auto",
      title="Translator Demo"
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化耳机销售机器人
    # initialize_sales_bot()

    # 启动 Gradio 服务
    launch_gradio()
