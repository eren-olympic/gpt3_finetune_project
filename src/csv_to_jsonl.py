import os
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set the training_file variable to your uploaded file ID
training_file = "file-XGinujblHPwGLSztz8cPS8XY"

# Create a fine-tuning job
response = openai.FineTune.create(
    model="text-davinci-003",
    training_file=training_file,
    n_epochs=4,
    learning_rate_multiplier=0.1,
    batch_size=4,
    prompt_loss_weight=0.1,
)

# Print the response
print(response)
