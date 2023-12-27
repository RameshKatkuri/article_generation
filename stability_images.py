import io
import os
import random
import string

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image
from stability_sdk import client


def create_image(image_prompt: str):
    res = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    filename = "article-generation-" + res + ".png"
    # Set up our connection to the API.
    stability_api = client.StabilityInference(key=os.environ['STABILITY_KEY'], verbose=True,
                                              engine="stable-diffusion-xl-1024-v1-0", )

    # Set up our seed initial generation parameters.
    answers = stability_api.generate(prompt=image_prompt, seed=4253978046, steps=50, cfg_scale=8.0, width=1024,
                                     height=1024, samples=1, sampler=generation.SAMPLER_K_DPMPP_2M)
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                img.save(filename)
                return filename
