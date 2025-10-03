import pandas as pd
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# 1. 데이터 로드
def load_data():
    df = pd.read_csv("sample_data.csv", encoding="utf-8-sig")
    df.columns = ["name", "attention", "emotion", "attitude"]
    return df

# 2. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 프롬프트 템플릿
template = """
너는 교육 상담 전문가야.
다음 학생의 검사 결과를 해석하고 요약 보고서를 작성해줘.

학생명: {name}
주의집중: {attention}
정서안정: {emotion}
학습태도: {attitude}

요약 보고서:
"""
prompt = PromptTemplate(
    input_variables=["name", "attention", "emotion", "attitude"],
    template=template,
)

# 4. LangGraph 상태 정의 (TypedDict로 명시)
class ExamState(TypedDict):
    name: str
    attention: str
    emotion: str
    attitude: str
    report: str

def analyze_student(state: ExamState):
    formatted_prompt = prompt.format(
        name=state["name"],
        attention=state["attention"],
        emotion=state["emotion"],
        attitude=state["attitude"]
    )
    result = llm.invoke(formatted_prompt)
    return {"report": result.content}

# 5. LangGraph 그래프 구성
graph = StateGraph(ExamState)
graph.add_node("analyze", analyze_student)
graph.set_entry_point("analyze")
graph.add_edge("analyze", END)
app = graph.compile()

# 6. 실행부
if __name__ == "__main__":
    df = load_data()
    print("CSV 컬럼명 확인:", df.columns.tolist())
    for _, row in df.iterrows():
        student_dict = row.to_dict()
        result = app.invoke({
            "name": student_dict["name"],
            "attention": student_dict["attention"],
            "emotion": student_dict["emotion"],
            "attitude": student_dict["attitude"],
            "report": ""
        })
        print(f"\n=== {student_dict['name']} 보고서 ===")
        print(result["report"])


