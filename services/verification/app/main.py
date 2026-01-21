from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from email_validator import validate_email, EmailNotValidError

app = FastAPI(title="Verification Service")

class EmailVerification(BaseModel):
    email: EmailStr

@app.post("/verify-email")
async def verify_email(data: EmailVerification):
    try:
        valid = validate_email(data.email)
        return {"message": f"Email {valid.email} is valid"}
    except EmailNotValidError as e:
        raise HTTPException(status_code=400, detail=str(e))