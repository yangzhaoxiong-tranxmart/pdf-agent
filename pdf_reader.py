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
import threading
import queue
import torch

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
                    # 实时更新显示
                    self.text_container.markdown(self.current_text)
                return
            else:
                token_id = value

            # 解码token为文本
            decoded_text = self.tokenizer.decode([token_id], skip_special_tokens=self.skip_special_tokens)
            self.current_text += decoded_text

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
    page_title="智能PDF阅读助手",
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
            '--markdown',  # 输出为 Markdown 格式
            # '2>&1',
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 读取标准输出
        for line in iter(process.stdout.readline, ''):
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
            text = f"Nougat 处理失败: {stderr}"

        # 清理临时文件
        os.unlink(input_path)
        import shutil
        shutil.rmtree(output_dir)

        return text

    except Exception as e:
        st.warning(f"Nougat 提取失败，使用备用方法: {str(e)}")
        return extract_text_from_pdf_2(uploaded_file)

# 生成PDF摘要（流式输出版本）
def generate_summary_streaming(tokenizer, model, text, max_length=500, streamer=None):
    messages = [{"role": "user",
                 "content": f"请为以下文本生成一个简洁的摘要，不超过{max_length}字：\n\n{text[:10000]}"}]
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
        outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7, do_sample=True, streamer=streamer)
    else:
        outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7, do_sample=True)
    
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

# 生成PDF摘要（非流式版本，保持向后兼容）
def generate_summary(tokenizer, model, text, max_length=500):
    return generate_summary_streaming(tokenizer, model, text, max_length, streamer=None)

# 使用AI提取更相关的关键词
# def ai_extract_keywords(text, num_keywords=15):
#     prompt = f"""请从以下文本中提取{num_keywords}个最重要的关键词。
#     要求：关键词应该是最能代表文档主题和内容的术语、概念或实体。
#     文本内容：
#     {text[:3000]}

#     请以逗号分隔的形式返回关键词："""

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "你是一个专业的关键词提取助手。"},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=100
#         )
#         keywords = response.choices[0].message.content.strip().split(',')
#         keywords = [keyword.strip() for keyword in keywords if keyword.strip()]
#         return keywords[:num_keywords]
#     except Exception as e:
#         return f"提取关键词时出错: {str(e)}"

# 划重点功能
# def highlight_important_points(text, num_points=5):
#     prompt = f"""请从以下文本中提取{num_points}个最重要的观点或信息点。
#     文本内容：
#     {text[:3000]}

#     请以清晰的列表形式返回这些重点："""

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "你是一个专业的文档分析助手，擅长提取关键信息。"},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=300
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"划重点时出错: {str(e)}"

# 生成词云
# def generate_wordcloud(keywords):
#     # 准备词云数据
#     word_freq = {word: count for word, count in keywords}

#     # 生成词云
#     wordcloud = WordCloud(
#         width=800,
#         height=400,
#         background_color='white',
#         colormap='viridis'
#     ).generate_from_frequencies(word_freq)

#     # 显示词云
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     plt.tight_layout()

#     return plt

# 在PDF中添加高亮
# def add_highlight_to_pdf(input_pdf, output_pdf, highlights):
#     """
#     使用PyMuPDF在PDF中添加高亮标记
#     """
#     # 保存上传的文件到临时位置
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#         tmp_file.write(input_pdf.getvalue())
#         tmp_path = tmp_file.name

#     # 打开PDF文档
#     doc = fitz.open(tmp_path)

#     # 对每个高亮项进行处理
#     for highlight in highlights:
#         page_num = highlight.get("page", 0)
#         text = highlight.get("text", "")
#         color = highlight.get("color", yellow)

#         if page_num < len(doc):
#             page = doc[page_num]

#             # 查找文本位置
#             text_instances = page.search_for(text)

#             # 对每个找到的文本实例添加高亮
#             for inst in text_instances:
#                 highlight_annot = page.add_highlight_annot(inst)
#                 highlight_annot.set_colors(stroke=color)
#                 highlight_annot.update()

#     # 保存修改后的PDF
#     doc.save(output_pdf)
#     doc.close()

#     # 清理临时文件
#     os.unlink(tmp_path)

#     return output_pdf

# 在PDF中添加注释
# def add_comment_to_pdf(input_pdf, output_pdf, comments):
#     """
#     使用PyMuPDF在PDF中添加注释
#     """
#     # 保存上传的文件到临时位置
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#         tmp_file.write(input_pdf.getvalue())
#         tmp_path = tmp_file.name

#     # 打开PDF文档
#     doc = fitz.open(tmp_path)

#     # 对每个注释项进行处理
#     for comment in comments:
#         page_num = comment.get("page", 0)
#         text = comment.get("text", "")
#         position = comment.get("position", (50, 50))

#         if page_num < len(doc):
#             page = doc[page_num]

#             # 添加文本注释
#             annot = page.add_text_annot(position, text)
#             annot.set_info(title="AI注释", content=text)
#             annot.update()

#     # 保存修改后的PDF
#     doc.save(output_pdf)
#     doc.close()

#     # 清理临时文件
#     os.unlink(tmp_path)

#     return output_pdf

# 创建下载链接
def create_download_link(pdf_data, filename):
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">下载修改后的PDF</a>'
    return href

# 主应用
def main():
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"使用设备: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # GPU使用半精度，CPU使用全精度
        device_map="auto" if device == "cuda" else None  # GPU自动分配，CPU不使用
    )
    # 如果使用CPU，手动移动到CPU
    if device == "cpu":
        model = model.to(device)

    st.title("📚 智能PDF阅读助手")
    st.markdown("上传PDF文件，使用AI辅助阅读、理解和分析文档内容，并直接在PDF上高亮和注释")

    # 文件上传
    uploaded_file = st.file_uploader("选择PDF文件", type="pdf")
    print(uploaded_file)
    print(type(uploaded_file))

    if uploaded_file is not None:
        # 选择提取方法
        extraction_method = st.selectbox(
            "选择文本提取方法",
            ["PyMuPDF (快速)", "Nougat (精确，适合数学公式)"],
            help="Nougat 更适合包含数学公式的学术文档"
        )
        # 提取文本
        with st.spinner("正在解析PDF..."):
            if extraction_method == "Nougat (精确，适合数学公式)":
                text = extract_text_from_pdf_nougat(uploaded_file)
            else:
                text = extract_text_from_pdf_2(uploaded_file)

        if not text.strip():
            st.error("无法从PDF中提取文本，可能是扫描件或图像PDF")
            return

        # 创建选项卡
        # tab1, tab2, tab3, tab4, tab5 = st.tabs(["文档概览", "关键词分析", "重点提取", "智能问答", "PDF高亮与注释"])
        tab1, = st.tabs(["文档概览"])

        with tab1:
            # 显示基本信息
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📄 文档信息")
                st.write(f"**文件名:** {uploaded_file.name}")
                st.write(f"**页数:** {len(PyPDF2.PdfReader(uploaded_file).pages)}")
                st.write(f"**文本长度:** {len(text)} 字符")

            # 生成摘要
            with col2:
                st.subheader("📝 文档摘要")
                if st.button("生成摘要", key="summary_btn"):
                    # 创建用于显示流式输出的容器
                    summary_container = st.empty()
                    summary_container.markdown("正在生成摘要...")
                    
                    # 创建流式输出器
                    streamer = StreamlitStreamer(tokenizer, summary_container)
                    
                    # 使用流式输出生成摘要
                    with st.spinner("正在生成摘要..."):
                        summary = generate_summary_streaming(tokenizer, model, text, streamer=streamer)
                    
                    # 最终显示完整摘要
                    summary_container.success("摘要生成完成！")
                    st.markdown("**完整摘要：**")
                    st.markdown(summary)

        # with tab2:
        #     st.subheader("🔑 关键词分析")

        #     col1, col2 = st.columns(2)

        #     with col1:
        #         if st.button("提取高频关键词", key="keywords_btn"):
        #             with st.spinner("正在分析关键词..."):
        #                 keywords = extract_keywords(text)
        #                 st.write("**高频关键词:**")
        #                 for i, (word, count) in enumerate(keywords, 1):
        #                     st.write(f"{i}. {word} (出现{count}次)")

        #     with col2:
        #         if st.button("AI提取重要关键词", key="ai_keywords_btn"):
        #             with st.spinner("AI正在分析..."):
        #                 keywords = ai_extract_keywords(text)
        #                 st.write("**AI提取的关键词:**")
        #                 for i, keyword in enumerate(keywords, 1):
        #                     st.write(f"{i}. {keyword}")

        #     # 显示词云
        #     if st.button("生成关键词词云", key="wordcloud_btn"):
        #         with st.spinner("正在生成词云..."):
        #             keywords = extract_keywords(text, 30)
        #             fig = generate_wordcloud(keywords)
        #             st.pyplot(fig)

        # with tab3:
        #     st.subheader("🌟 重点内容提取")

        #     if st.button("划重点", key="highlight_btn"):
        #         with st.spinner("AI正在分析重点内容..."):
        #             highlights = highlight_important_points(text)
        #             st.success("**文档重点内容:**")
        #             st.markdown(highlights)

        #     # 允许用户自定义搜索重点
        #     st.subheader("🔍 自定义搜索")
        #     search_term = st.text_input("输入您关注的关键词或概念:")
        #     if search_term:
        #         # 查找包含关键词的句子
        #         sentences = re.split(r'[.!?]+', text)
        #         relevant_sentences = [s.strip() for s in sentences if search_term.lower() in s.lower()]

        #         if relevant_sentences:
        #             st.write(f"找到 {len(relevant_sentences)} 条相关内容:")
        #             for i, sentence in enumerate(relevant_sentences[:10], 1):
        #                 # 高亮显示搜索词
        #                 highlighted = sentence.replace(
        #                     search_term,
        #                     f"<mark>{search_term}</mark>"
        #                 )
        #                 st.markdown(f"{i}. {highlighted}", unsafe_allow_html=True)

        #                 # 添加到高亮列表
        #                 if 'highlights' not in st.session_state:
        #                     st.session_state.highlights = []

        #                 # 尝试找到句子所在的页面
        #                 page_num = 0  # 默认第一页
        #                 st.session_state.highlights.append({
        #                     "page": page_num,
        #                     "text": search_term,
        #                     "color": yellow
        #                 })
        #         else:
        #             st.warning("未找到相关内容。")

        # with tab4:
        #     st.subheader("❓ 智能问答")

        #     # 分割文本并创建向量存储
        #     with st.spinner("正在处理文档内容..."):
        #         texts = split_text(text)
        #         vectorstore = create_vector_store(texts)

        #     question = st.text_input("输入关于文档的问题:")

        #     if question:
        #         with st.spinner("思考中..."):
        #             qa_chain = RetrievalQA.from_chain_type(
        #                 llm=OpenAI(),
        #                 chain_type="stuff",
        #                 retriever=vectorstore.as_retriever(),
        #                 return_source_documents=True
        #             )
        #             result = qa_chain({"query": question})

        #             st.success("**答案:** " + result["result"])

        #             with st.expander("查看相关段落"):
        #                 for doc in result["source_documents"]:
        #                     st.write(doc.page_content)
        #                     st.markdown("---")

        # with tab5:
        #     st.subheader("🖍️ PDF高亮与注释")
        #     st.info("在此选项卡中，您可以将AI识别的重要内容和注释直接添加到PDF文件中")

        #     # 高亮选项
        #     st.subheader("高亮选项")
        #     highlight_color = st.selectbox(
        #         "选择高亮颜色",
        #         ["黄色", "红色", "蓝色", "绿色"],
        #         key="highlight_color"
        #     )

        #     color_map = {
        #         "黄色": yellow,
        #         "红色": (1, 0, 0),
        #         "蓝色": (0, 0, 1),
        #         "绿色": (0, 1, 0)
        #     }

        #     # 注释选项
        #     st.subheader("添加注释")
        #     comment_text = st.text_area("输入注释内容:", height=100)
        #     comment_page = st.number_input("注释所在页面:", min_value=1, value=1) - 1  # 转换为0-based索引

        #     if st.button("添加注释到PDF", key="add_comment_btn"):
        #         if comment_text:
        #             if 'comments' not in st.session_state:
        #                 st.session_state.comments = []

        #             st.session_state.comments.append({
        #                 "page": comment_page,
        #                 "text": comment_text,
        #                 "position": (50, 50)  # 默认位置，可以扩展为让用户选择位置
        #             })

        #             st.success("注释已添加到队列中！")
        #         else:
        #             st.warning("请输入注释内容")

        #     # 应用高亮和注释到PDF
        #     if st.button("生成带高亮和注释的PDF", key="apply_highlights_btn"):
        #         with st.spinner("正在处理PDF..."):
        #             # 创建临时输出文件
        #             with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_output:
        #                 output_path = tmp_output.name

        #             # 应用高亮
        #             if 'highlights' in st.session_state and st.session_state.highlights:
        #                 # 更新高亮颜色
        #                 for highlight in st.session_state.highlights:
        #                     highlight["color"] = color_map[highlight_color]

        #                 add_highlight_to_pdf(uploaded_file, output_path, st.session_state.highlights)

        #             # 应用注释
        #             if 'comments' in st.session_state and st.session_state.comments:
        #                 add_comment_to_pdf(uploaded_file, output_path, st.session_state.comments)

        #             # 读取处理后的PDF
        #             with open(output_path, "rb") as f:
        #                 pdf_data = f.read()

        #             # 创建下载链接
        #             href = create_download_link(pdf_data, f"annotated_{uploaded_file.name}")
        #             st.markdown(href, unsafe_allow_html=True)

        #             # 清理临时文件
        #             os.unlink(output_path)

        # 显示原始文本（可选）
        with st.expander("查看提取的文本"):
            st.text_area("文本内容", text, height=300)

if __name__ == "__main__":
    main()
