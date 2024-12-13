# Usage

This is just a quick hack.  Honestly you're better off just manually uploading your images through the web (or phone) UI.  I created this mess so I could play with short sequences of images.

1. Create a python virtual environment and activate it.
2. Install required packages: pip install -r requirements.txt
3. Create image capture directory: mkdir capture
4. Create directory and put a video in it (I used sample_inputs).  High resolution images are expensive to use with OpenAI/Anthropic.  I downsampled my video to 480p to save some green.
5. Get an API key from OpenAI and/or Anthropic and store it in your shell environment.  Add some credits ($$) to your account to pay for your requests.  During debugging I spent about $1.00 on OpenAI.
6. Edit the code to point to your sample video.  Modify other configuration variables as needed.
7. Run it: python describe_images.py
