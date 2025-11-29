from pprint import pprint

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

class ResearchPaperExtraction(BaseModel):
    title: str = Field(description="The title of the research paper")
    authors: list[str] = Field(description="The authors of the research paper")
    abstract: str = Field(description="The abstract of the research paper")
    keywords: list[str] = Field(description="The keywords of the research paper")

model = init_chat_model("gpt-4.1-nano", model_provider="openai", temperature=0)
# model = init_chat_model("gpt-4.1-mini", model_provider="openai", temperature=0)
# model = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai", temperature=0)
# model = init_chat_model("smollm2:1.7b", model_provider="ollama", temperature=0)

system_prompt = "You are an expert at structured data extraction. You will be given unstructured text from a research paper and should convert it into the given structure."
user_prompt = ("Title: A Survey on LLM-as-a-Judge. Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of 'LLM-as-a-Judge' where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field. Authors: Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, Jian Guo")


include_raw = True # set true if you want to see the raw response, ie, input/output tokens, usage metadata, etc

model_struct = model.with_structured_output(ResearchPaperExtraction, include_raw=include_raw)
response = model_struct.invoke([system_prompt, user_prompt])
print(response)

authors = {"Jiawei Gu", "Xuhui Jiang", "Zhichao Shi", "Hexiang Tan", "Xuehao Zhai", "Chengjin Xu", "Wei Li", "Yinghan Shen", "Shengjie Ma", "Honghao Liu", "Saizhuo Wang", "Kun Zhang", "Yuanzhuo Wang", "Wen Gao", "Lionel Ni", "Jian Guo"}

if include_raw:
    structured_data = response["parsed"]
    usage = response["raw"].usage_metadata
    response_meta = response["raw"].response_metadata
    refusal = response["raw"].additional_kwargs.get("refusal")
    parsing_error = response_meta.get("parsing_error")

    print("structured_data", structured_data)
    print("usage:", usage)
    print("response_meta:", response_meta)
    print("refusal:", refusal)
    print("parsing_error: ", parsing_error)

    print("Title: ", structured_data.title)
    print("Abstract: ")
    pprint(structured_data.abstract)
    print("Authors", structured_data.authors)
    print("Keywords", structured_data.keywords)

    print(authors == set(structured_data.authors))
    print(authors.symmetric_difference(set(structured_data.authors)))

else:
    extracted_authors = set(response.authors)
    print(extracted_authors == authors)
    print(extracted_authors.symmetric_difference(authors))