from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

model = init_chat_model("smollm2:1.7b", model_provider="ollama", temperature=0)
model_struct = model.with_structured_output(ContactInfo)


response = model_struct.invoke("Extract contact info from: John Doe, john@demo.com, (555) 123-4567")

print(response)
print(type(response))
print(response.name)
# ContactInfo(name='John Doe', email='john@demo.com', phone='(555) 123-4567')
