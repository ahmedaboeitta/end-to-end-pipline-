from pydantic import BaseModel


class QAPair(BaseModel):
    question: str
    answer: str


class QAOutput(BaseModel):
    qa_pairs: list[QAPair]