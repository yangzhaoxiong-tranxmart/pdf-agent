import streamlit as st
import PyPDF2
from collections import Counter
import matplotlib.pyplot as plt
import base64
from PyPDF2 import PdfReader, PdfWriter
import pymupdf
import tempfile
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from transformers import BitsAndBytesConfig
import threading
import queue
import torch

# 检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_TEXT_INPUT_LENGTH = 100000
MAX_OUTPUT_TOKEN = 16384

# 自定义Streamer类用于Streamlit流式输出
class StreamlitStreamer(TextStreamer):
    def __init__(self, tokenizer, text_container, skip_special_tokens=True, **decode_kwargs):
        super().__init__(tokenizer, skip_special_tokens=skip_special_tokens, **decode_kwargs)
        self.text_container = text_container
        self.current_text = ""
        self.skip_special_tokens = skip_special_tokens
        self.input_length = 0  # 记录输入prompt的长度

    def put(self, value):
        """重写put方法以实时更新Streamlit界面，只显示新生成的token"""
        if value is not None:
            # 将token ID解码为文本
            if hasattr(value, 'item') and value.numel() == 1:  # 如果是单个token的tensor
                token_id = value.item()
            elif hasattr(value, 'tolist'):  # 如果是包含多个token的tensor
                token_ids = value.tolist()
                if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
                    # 如果token_ids是嵌套列表，取第一个元素
                    token_ids = token_ids[0]
                # 解码所有token为文本
                decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=self.skip_special_tokens)
                # 只保留新生成的部分（去掉输入prompt部分）
                if len(decoded_text) > self.input_length:
                    new_text = decoded_text[self.input_length:]
                    self.current_text += new_text
                    self.current_text = self.current_text.replace('\\[', '$$')
                    self.current_text = self.current_text.replace('\\]', '$$')
                    self.current_text = self.current_text.replace('\\(', '$')
                    self.current_text = self.current_text.replace('\\)', '$')
                    # 实时更新显示
                    self.text_container.markdown(self.current_text)
                return
            else:
                token_id = value

            # 解码token为文本
            decoded_text = self.tokenizer.decode([token_id], skip_special_tokens=self.skip_special_tokens)
            self.current_text += decoded_text
            self.current_text = self.current_text.replace('\\[', '$$')
            self.current_text = self.current_text.replace('\\]', '$$')
            self.current_text = self.current_text.replace('\\(', '$')
            self.current_text = self.current_text.replace('\\)', '$')

            # 实时更新显示
            self.text_container.markdown(self.current_text)

    def set_input_length(self, length):
        """设置输入prompt的长度，用于过滤掉prompt部分"""
        self.input_length = length

    def end(self):
        """结束流式输出"""
        pass

# 设置页面标题和布局
st.set_page_config(
    page_title="论文阅读助手",
    page_icon="📚",
    layout="wide"
)

# 提取PDF文本
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_pdf_2(uploaded_file):
    doc = pymupdf.open(stream=uploaded_file.getvalue(), filetype='pdf')
    text = ''
    for page in doc:
        text = text + page.get_text()
    doc.close()
    return text

def extract_text_from_pdf_nougat(uploaded_file):
    """
    使用 Nougat API 提取 PDF 文本内容
    """
    try:
        import subprocess
        import tempfile
        import os

        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_input:
            tmp_input.write(uploaded_file.getvalue())
            input_path = tmp_input.name

        # 创建临时输出目录
        output_dir = tempfile.mkdtemp()

        # 调用 Nougat 命令行工具
        # result = subprocess.run([
        #     'nougat',
        #     input_path,
        #     '--out', output_dir,
        #     '--markdown'  # 输出为 Markdown 格式
        # ], capture_output=True, text=True)
        process = subprocess.Popen([
            'nougat',
            input_path,
            '--out', output_dir,
            '--model', '0.1.0-base',
            '--no-skipping',
            '--markdown',  # 输出为 Markdown 格式
            # '2>&1',
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 读取标准输出
        # for line in iter(process.stdout.readline, ''):
        #     print(line.rstrip())
        for line in iter(process.stderr.readline, ''):
            print(line.rstrip())

        # 等待进程完成并获取返回码
        returncode = process.wait()

        if returncode == 0:
            # 读取生成的 Markdown 文件
            output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '.mmd')
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = "Nougat 处理完成但未找到输出文件"
        else:
            # TODO(zhaoxiong): add error message
            text = f"Nougat 处理失败: "

        # 清理临时文件
        os.unlink(input_path)
        import shutil
        shutil.rmtree(output_dir)

        return text

    except Exception as e:
        st.warning(f"Nougat 提取失败，使用备用方法: {str(e)}")
        return extract_text_from_pdf_2(uploaded_file)

# 生成PDF摘要（流式输出版本）
def generate_summary_streaming(tokenizer, text, max_length=1000, streamer=None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        # offload_folder="./offload",  # 设置卸载文件夹
        max_memory={0: "14GB", "cpu": "24GB"}  # 限制GPU和CPU内存
    )
    print('Model specification:')
    print(model)
    # 如果使用CPU，手动移动到CPU
    if device == "cpu":
        return 'GPU unavailable'

    messages = [{"role": "user",
                 "content": f"请为以下文本生成一个简洁的摘要(in Chinese)，不超过{max_length}字：\n\n{text[:MAX_TEXT_INPUT_LENGTH]}"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 使用流式输出
    if streamer:
        # 设置输入长度，用于过滤掉prompt部分
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        streamer.set_input_length(len(input_text))
        outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKEN, temperature=0.7, do_sample=True, streamer=streamer)
    else:
        outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKEN, temperature=0.7, do_sample=True)

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

def generate_technical_derivation_streaming(tokenizer, text, max_length=1000, streamer=None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        # offload_folder="./offload",  # 设置卸载文件夹
        max_memory={0: "14GB", "cpu": "24GB"}  # 限制GPU和CPU内存
    )
    print('Model specification:')
    print(model)
    # 如果使用CPU，手动移动到CPU
    if device == "cpu":
        return 'GPU unavailable'

    # messages = [{"role": "user",
    #              "content": f"请介绍这个论文里的主要理论技术内容，给出关键公式，并进行推导和分析（in Chinese）：\n\n{text[:MAX_TEXT_INPUT_LENGTH]}"}]
    messages = [{"role": "user",
                 "content": f"请用数学的语言介绍这个论文里的主要理论技术内容，给出进行推导和分析：\n\n{text[:MAX_TEXT_INPUT_LENGTH]}"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 使用流式输出
    if streamer:
        # 设置输入长度，用于过滤掉prompt部分
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        streamer.set_input_length(len(input_text))
        outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKEN, temperature=0.7, do_sample=True, streamer=streamer)
    else:
        outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKEN, temperature=0.7, do_sample=True)

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

def generate_reproduction_code_streaming(tokenizer, text, max_length=1000, streamer=None):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        # offload_folder="./offload",  # 设置卸载文件夹
        max_memory={0: "14GB", "cpu": "24GB"}  # 限制GPU和CPU内存
    )
    print('Model specification:')
    print(model)
    # 如果使用CPU，手动移动到CPU
    if device == "cpu":
        return 'GPU unavailable'

    messages = [{"role": "user",
                 "content": f"请根据这篇论文里的理论技术内容，生成一段复现的代码：\n\n{text[:MAX_TEXT_INPUT_LENGTH]}"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # 使用流式输出
    if streamer:
        # 设置输入长度，用于过滤掉prompt部分
        input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        streamer.set_input_length(len(input_text))
        outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKEN, temperature=0.7, do_sample=True, streamer=streamer)
    else:
        outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKEN, temperature=0.7, do_sample=True)

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

# 生成PDF摘要（非流式版本，保持向后兼容）
def generate_summary(tokenizer, text, max_length=500):
    return generate_summary_streaming(tokenizer, text, max_length, streamer=None)


# 创建下载链接
def create_download_link(pdf_data, filename):
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">下载修改后的PDF</a>'
    return href

def main():
    st.info(f"使用设备: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

    st.title("📚 智能PDF阅读助手")
    st.markdown("上传PDF文件，使用AI辅助阅读、理解和分析文档内容，并直接在PDF上高亮和注释")

    # 文件上传
    uploaded_file = st.file_uploader("选择PDF文件", type="pdf")
    print('Uploaded file: ', uploaded_file)
    # print(type(uploaded_file))

    text = ''

    # 在文件上传后检查是否已经提取过文本
    if uploaded_file is not None:
        # 使用文件名作为缓存键
        cache_key = f"text_{uploaded_file.name}"
        extraction_key = f"extraction_method_{uploaded_file.name}"
        
        # 总是显示提取方法选择框，但记住用户的选择
        if extraction_key not in st.session_state:
            st.session_state[extraction_key] = "PyMuPDF (快速)"
        
        extraction_method = st.selectbox(
            "选择文本提取方法",
            ["PyMuPDF (快速)", "Nougat (精确，适合数学公式)"],
            index=0 if st.session_state[extraction_key] == "PyMuPDF (快速)" else 1,
            help="Nougat 更适合包含数学公式的学术文档"
        )
        
        # 更新选择的方法
        st.session_state[extraction_key] = extraction_method
        
        # 检查是否需要重新提取文本
        need_extraction = (
            cache_key not in st.session_state or 
            st.session_state.get(f"last_method_{uploaded_file.name}") != extraction_method
        )
        
        if need_extraction:
            with st.spinner("正在解析PDF..."):
                if extraction_method == "Nougat (精确，适合数学公式)":
                    extracted_text = extract_text_from_pdf_nougat(uploaded_file)
                else:
                    extracted_text = extract_text_from_pdf_2(uploaded_file)
                
                # 存储提取的文本和使用的提取方法
                st.session_state[cache_key] = extracted_text
                st.session_state[f"last_method_{uploaded_file.name}"] = extraction_method
        
        text = st.session_state[cache_key]

        if not text.strip():
            st.error("无法从PDF中提取文本，可能是扫描件或图像PDF")
            return

        # 创建选项卡
        tab1, tab2, tab3 = st.tabs(["文档概览", "关键推导", "Talk is cheap"])

        with tab1:
            # 显示基本信息
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 文档信息")
                st.write(f"**文件名:** {uploaded_file.name}")
                # 避免重复创建 PdfReader
                if 'pdf_pages' not in st.session_state:
                    st.session_state['pdf_pages'] = len(PyPDF2.PdfReader(uploaded_file).pages)
                st.write(f"**页数:** {st.session_state['pdf_pages']}")
                st.write(f"**文本长度:** {len(text)} 字符")

            # 生成摘要
            with col2:
                st.subheader("📝 文档摘要")
                if st.button("生成摘要", key="summary_btn"):
                    # 创建用于显示流式输出的容器
                    summary_container = st.empty()
                    # summary_container.markdown("正在生成摘要...")

                    # 创建流式输出器
                    streamer = StreamlitStreamer(tokenizer, summary_container)

                    # 使用流式输出生成摘要
                    with st.spinner("正在生成摘要..."):
                        summary = generate_summary_streaming(tokenizer, text, streamer=streamer)

                    # 最终显示完整摘要
                    summary_container.success("摘要生成完成！")
                    st.markdown("**完整摘要：**")
                    st.markdown(summary)

        with tab2:
            # 显示基本信息
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 文档信息")
                st.write(f"**文件名:** {uploaded_file.name}")
                # 避免重复创建 PdfReader
                if 'pdf_pages' not in st.session_state:
                    st.session_state['pdf_pages'] = len(PyPDF2.PdfReader(uploaded_file).pages)
                st.write(f"**页数:** {st.session_state['pdf_pages']}")
                st.write(f"**文本长度:** {len(text)} 字符")

            with col2:
                st.subheader("🎓 关键推导")
                if st.button("生成关键推导", key="derivation_btn"):
                    # 创建用于显示流式输出的容器
                    summary_container = st.empty()

                    # 创建流式输出器
                    streamer = StreamlitStreamer(tokenizer, summary_container)

                    # 使用流式输出生成摘要
                    with st.spinner("正在生成推导..."):
                        summary = generate_technical_derivation_streaming(tokenizer, text, streamer=streamer)

                    # 最终显示完整摘要
                    summary_container.success("推导生成完成！")
                    st.markdown("**关键推导：**")
                    summary = summary.replace('\\[', '$$')
                    summary = summary.replace('\\]', '$$')
                    summary = summary.replace('\\(', '$')
                    summary = summary.replace('\\)', '$')
                    st.markdown(summary)
                    # st.text(summary)

        with tab3:
            # 显示基本信息
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 文档信息")
                st.write(f"**文件名:** {uploaded_file.name}")
                # 避免重复创建 PdfReader
                if 'pdf_pages' not in st.session_state:
                    st.session_state['pdf_pages'] = len(PyPDF2.PdfReader(uploaded_file).pages)
                st.write(f"**页数:** {st.session_state['pdf_pages']}")
                st.write(f"**文本长度:** {len(text)} 字符")

            with col2:
                st.subheader("💻 代码复现")
                summary_container_1 = st.empty()
                summary_container_1.markdown("看不懂，直接帮我把代码写了吧")
                if st.button("生成代码", key="reproduction_btn"):
                    summary_container = st.empty()

                    # 创建流式输出器
                    streamer = StreamlitStreamer(tokenizer, summary_container)

                    # 使用流式输出生成摘要
                    with st.spinner("正在生成代码..."):
                        summary = generate_reproduction_code_streaming(tokenizer, text, streamer=streamer)

                    # 最终显示完整摘要
                    summary_container.success("代码生成完成！")
                    st.markdown("**复现代码：**")
                    st.markdown(summary)

        # 显示原始文本（可选）
        with st.expander("查看提取的文本"):
            st.text_area("文本内容", text, height=300)

if __name__ == "__main__":
    main()
