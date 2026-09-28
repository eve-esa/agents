from pydantic import BaseModel, Field


class ContextSettings(BaseModel):
    input_tokens: int = Field(default=32000, ge=1024)
    window_tokens: int = Field(default=128000, ge=4096)
    output_reserve: int = Field(default=8192, ge=0)
    preview_tokens: int = Field(default=1800, ge=512, le=4096)
    memory_tokens: int = Field(default=4000, ge=128) # for active memory
