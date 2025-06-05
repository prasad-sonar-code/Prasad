from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import openai
import os
from typing import Optional

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class GenerationRequest(BaseModel):
    api_key: str
    ideal_customer: str
    tone: str = "professional"
    length: str = "medium"

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_lead_strategy(request: Request):
    form_data = await request.form()
    req = GenerationRequest(
        api_key=form_data.get("api_key"),
        ideal_customer=form_data.get("ideal_customer"),
        tone=form_data.get("tone"),
        length=form_data.get("length")
    )
    
    if not req.ideal_customer or len(req.ideal_customer) < 10:
        raise HTTPException(status_code=400, detail="Please provide a detailed ideal customer description")
    
    try:
        openai.api_key = req.api_key
        
        # Generate LinkedIn search query
        search_query = generate_search_query(req.ideal_customer)
        
        # Generate connection request message
        connection_message = generate_connection_message(
            req.ideal_customer, 
            req.tone, 
            req.length
        )
        
        # Generate follow-up message
        follow_up_message = generate_follow_up_message(
            req.ideal_customer,
            req.tone,
            req.length
        )
        
        return HTMLResponse(f"""
            <div class="space-y-6">
                <div class="bg-blue-50 p-4 rounded-lg border border-blue-200">
                    <h3 class="font-semibold text-blue-800">LinkedIn Search Query</h3>
                    <p class="mt-2 p-3 bg-white rounded-md">{search_query}</p>
                    <p class="mt-1 text-sm text-gray-500">Copy this into LinkedIn's search bar to find your ideal customers</p>
                </div>
                
                <div class="bg-green-50 p-4 rounded-lg border border-green-200">
                    <h3 class="font-semibold text-green-800">Personalized Connection Request</h3>
                    <p class="mt-2 p-3 bg-white rounded-md">{connection_message}</p>
                    <p class="mt-1 text-sm text-gray-500">Use this when sending connection requests</p>
                </div>
                
                <div class="bg-purple-50 p-4 rounded-lg border border-purple-200">
                    <h3 class="font-semibold text-purple-800">Follow-Up Message (Send 3-5 days later)</h3>
                    <p class="mt-2 p-3 bg-white rounded-md">{follow_up_message}</p>
                    <p class="mt-1 text-sm text-gray-500">Send this after they accept your connection</p>
                </div>
            </div>
        """)
        
    except Exception as e:
        return HTMLResponse(f"""
            <div class="bg-red-50 p-4 rounded-lg border border-red-200">
                <h3 class="font-semibold text-red-800">Error</h3>
                <p class="mt-2 text-red-600">Failed to generate content: {str(e)}</p>
                <p class="mt-1 text-sm text-gray-500">Please check your API key and try again</p>
            </div>
        """)

def generate_search_query(ideal_customer: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a LinkedIn search query expert. Create precise LinkedIn search queries based on the user's ideal customer profile."},
            {"role": "user", "content": f"Create a LinkedIn search query to find: {ideal_customer}. Include relevant filters for industry, job title, and location if specified."}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def generate_connection_message(ideal_customer: str, tone: str, length: str) -> str:
    tone_instructions = {
        "professional": "Use formal business language",
        "friendly": "Use warm, approachable language",
        "direct": "Be concise and to the point",
        "casual": "Use conversational language"
    }
    
    length_instructions = {
        "short": "1-2 sentences max",
        "medium": "3-4 sentences",
        "long": "5+ sentences with more detail"
    }
    
    prompt = f"""
    Create a personalized LinkedIn connection request message for {ideal_customer}.
    {tone_instructions[tone]}. {length_instructions[length]}.
    Make it personalized by referencing something from their profile that would resonate.
    Include a clear reason for connecting that provides value to them.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at writing high-conversion LinkedIn connection requests."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def generate_follow_up_message(ideal_customer: str, tone: str, length: str) -> str:
    prompt = f"""
    Create a follow-up message to send after a LinkedIn connection is accepted.
    The message should provide value to {ideal_customer} and naturally lead to a conversation.
    Tone: {tone}. Length: {length}.
    Offer something useful like an insight, resource, or question.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at writing LinkedIn follow-up messages that start valuable conversations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)