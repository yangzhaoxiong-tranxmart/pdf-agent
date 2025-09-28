## 运行步骤
1. `source ./setup.sh` 
（后续请在虚拟环境中运行，要退出虚拟环境请执行`deactivate`）
2. `streamlit run pdf_reader.py`

## 操作示意
1. 初始界面
![](./doc/Screenshot%20from%202025-09-28%2021-19-34.png)
2. 点击Browse File，上传PDF文件
3. 选择文本提取方式Nougat（初次运行需要下载模型，共1.4GB），在已下好模型的情况下大约需要几十秒钟，取决于PDF长度
![](./doc/Screenshot%20from%202025-09-28%2021-20-02.png)
4. 点击生成摘要，开始“思考”然后输出文本总结
![](./doc/Screenshot%20from%202025-09-28%2021-21-39.png)
（初次运行需要下载模型，总共30GB，如不想在pdf_reader中下载可先手动预下载，执行以下python脚本）
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
```
或者选用其他开源大语言模型，14B经过量化的模型应该都能跑。
4. 高级功能，懒人必备
- 生成推导
![](./doc/Screenshot%20from%202025-09-28%2021-23-10.png)
![](./doc/Screenshot%20from%202025-09-28%2021-24-28.png)
- Talk is cheap, show me the code
![](./doc/Screenshot%20from%202025-09-28%2021-27-32.png)
![](./doc/Screenshot%20from%202025-09-28%2021-32-08.png)
- 自行开发，只要修改prompt就能得到想要的功能
