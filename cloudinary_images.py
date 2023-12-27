from dotenv import load_dotenv

load_dotenv()
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Set configuration parameter: return "https" URLs by setting secure=True
config = cloudinary.config(secure=True)


# Upload the image
def upload_image(image_path: str):
    cloudinary.uploader.upload(image_path, public_id=image_path.split(".")[0], unique_filename=False, overwrite=True)
    srcURL = cloudinary.CloudinaryImage(image_path.split(".")[0]).build_url()
    return srcURL
