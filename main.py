import uvicorn
import os
from api.energy_api import app

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

