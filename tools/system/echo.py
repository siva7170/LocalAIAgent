from pydantic import BaseModel, Field


class EchoArguments(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=1000
    )


def echo(message: str):

    return {
        "message": message,
        "status": "success"
    }