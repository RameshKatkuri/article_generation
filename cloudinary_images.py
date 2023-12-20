from dotenv import load_dotenv

load_dotenv()
import cloudinary
import cloudinary.uploader
import cloudinary.api


# Set configuration parameter: return "https" URLs by setting secure=True
config = cloudinary.config(secure=True)


# Upload the image
def upload_image(image_path: str):
    cloudinary.uploader.upload(image_path, public_id="quickstart_butterfly-rol", unique_filename=False, overwrite=True)
    srcURL = cloudinary.CloudinaryImage("quickstart_butterfly-rol").build_url()
    return srcURL
