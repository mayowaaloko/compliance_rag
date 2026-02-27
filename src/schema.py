from pydantic import BaseModel, Field
from typing import Optional


class CitedSource(BaseModel):
    document_name: str = Field(description="Name of the source document")
    page_number: Optional[int] = Field(description="Page number in the document")
    section: Optional[str] = Field(
        description="Section or article reference if mentioned e.g. Section 4.2"
    )
    law_category: Optional[str] = Field(
        description="Category of the law e.g. Taxation, Data Protection, Corporate/CAC, Employment, General Compliance"
    )
    contains_table: Optional[bool] = Field(
        description="True if the source chunk contained a markdown table"
    )


class ComplianceAnswer(BaseModel):
    answer: str = Field(
        description="The compliance answer based strictly on the provided context"
    )
    found_in_docs: bool = Field(
        description="True if the answer was found in the documents, False if not"
    )
    sources: list[CitedSource] = Field(
        description="List of sources that support the answer"
    )
    confidence: str = Field(
        description="How confident the answer is: high, medium, or low"
    )
