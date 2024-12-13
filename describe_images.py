"""
Not production code.  Just a quick hack to see what an LLM has to say
about a sequence of images.
"""

import cv2
import concurrent
import base64
import copy
from openai import OpenAI
from anthropic import Anthropic
import os

#
# Config
#

# source = 0  # primary camera capture
source = "sample_inputs/combined_480p.mov"


# Path to store frame captures for upload to LLM
capture_directory = "capture"

# Prompt stuff
animal = "cat"
animal_plural = "cats"
animal_indef_art = "a"

# number of frames to send to LLM
# processing images is expensive!!!
number_of_frames = 3

# collect every frame_spacing-th frame
#
# Frames rates vary (live cature vs video) and there is also keystroke detection
# delay (see code below for details) so the actual time between each
# frame is difficult to predict.
frame_spacing = 5

# pause the video playback when labeling image?
pause_while_labeling = True

# automatically send when frame buffer is full?  If False, only sent on space keystroke
autosend_frames = False

# show tranmitted frames in preview window
preview_frames_on_send = False

# must be openai or anthropic
llm_api_provider="anthropic"

# mini seems to work fine
openai_model = "gpt-4o-mini"
anthropic_model = "claude-3-5-sonnet-20241022"

system_prompt = f"You are an online {animal}-sitting service.  You are supplied with a sequence of images that should include one or more {animal_plural}.  The {animal}'s owner will ask one or more questions.  Your answer to each question should be a single word.  If you are asked multiple questions, separate your answers by a comma (,).  Do not include any text in your response except for the one-word answers to the questions each separated by a comma."

prompt = f"This is a sequence of images in time order showing {animal_indef_art} {animal} engaged in some activity.  Two questions: 1) In a single word, what is this {animal} doing?  Keep your answer general and if you don't know, just guess.  Example responses include 'sitting', 'playing', 'sleeping'.  If there is no {animal} in this image, just respond with 'none' to this first question; 2) How does the {animal} in this image feel?  Please answer with a single word.  Examples of responses are 'happy', 'hungry', 'playful'.  If you're not sure, try to give your best guess.  If there is no {animal} in the image just respond with 'none' to this second question."

#
# End Config
#

if preview_frames_on_send and autosend_frames:
  raise Exception("Don't set both preview_frames_on_send and autosend_frames")

openai_client = OpenAI()
anthropic_client = Anthropic()

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Add support for
#    video-llama: https://huggingface.co/papers/2306.02858
def send_openai_request(image_paths):
    # Getting the base64 string
    base64_images = []
    for path in image_paths:
        base64_images.append(encode_image(path))

    content = [{
                        "type": "text",
                        "text": prompt,
                    }]
    for image in base64_images:
        content.append({
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/png;base64,{image}"
                        }})

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text",
                     "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    return response.choices[0].message.content

def send_anthropic_request(image_paths):
    # Getting the base64 string
    base64_images = []
    for path in image_paths:
        base64_images.append(encode_image(path))

    # anthropic recommends image-then-text structure
    content = []
    for image in base64_images:
        content.append({
                        "type": "image",
                        "source": {
                          "type": "base64",
                          "media_type": "image/png",
                          "data":  image
                        }})
    content.append({
                        "type": "text",
                        "text": prompt,
                    })

    response = anthropic_client.messages.create(
      model=anthropic_model,
      max_tokens=1000,
      system=system_prompt,
      messages=[
        {
          "role": "user",
          "content": content,
        }
      ],
    )
    return response.content[0].text

def write_frames(frames):
    print("Writing")
    names = []
    for i, frame in enumerate(frames):
        names.append(f"{capture_directory}/frame{i:03d}.png")
        print(names[-1])
        cv2.imwrite(names[-1], frame)
    print("Wrote " + str(names))
    if preview_frames_on_send:
      for name in names:
        os.system("open "+name)
    return names

def fetch_label(frames):
    print("Annotating")
    capture_files = write_frames(frames)
    if llm_api_provider == "anthropic":
      response = send_anthropic_request(capture_files)
    else:
      response = send_openai_request(capture_files)
    print(response)
    return response

def main():

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(source)
    current_annotation = ""
    current_future = None
    frame_history = []

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    frame_count = 1
    with concurrent.futures.ProcessPoolExecutor() as executor:
        while rval:
            if frame_count % frame_spacing == 0:
                frame_history.append(copy.copy(frame))
                if len(frame_history) > number_of_frames:
                    frame_history.pop(0)
            cv2.putText(frame, current_annotation, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            frame_count += 1
            key = cv2.waitKey(20)
            if current_future and (current_future.done() or pause_while_labeling):
                current_annotation = current_future.result()
                current_future = None
                frame_history = []
                print("Updated annotation: " + current_annotation)
            if ((autosend_frames and len(frame_history) == number_of_frames) or (key == 32)) and (not current_future or current_future.done()):
              current_future = executor.submit(fetch_label, copy.copy(frame_history))
              # fetch_label(frame_history)
            if key == 27: # exit on ESC
                break

    vc.release()
    cv2.destroyWindow("preview")

if __name__ == "__main__":
    main()
