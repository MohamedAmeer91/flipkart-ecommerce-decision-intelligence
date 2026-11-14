"""Pydantic models for request validation."""
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    tenure_months: float = Field(ge=1, le=120, description="X1 customer tenure in months")
    complaints_90d: int = Field(ge=0, le=30, description="X2 complaints in last 90 days")
    discount_share: float = Field(ge=0, le=1, description="X3 share of orders on discount")
    avg_order_value: float = Field(ge=0.5, le=200, description="X4 average past order, hundreds of INR")
    loyalty_score: float = Field(ge=0, le=100, description="X5 loyalty score")
    cod_flag: int = Field(ge=0, le=1, description="X6 cash on delivery user")
    plus_member_flag: int = Field(ge=0, le=1, description="X7 plus member")
