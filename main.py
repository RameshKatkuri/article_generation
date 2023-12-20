
import json
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from cloudinary_images import upload_image
from stability_images import create_image

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")


def stateful_button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]

    return st.session_state[key]


def llm_model(output_template, input_variables, web_search_results):
    prompt_template = PromptTemplate(input_variables=input_variables, template=output_template)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    with get_openai_callback() as cb:
        response = chain.run(web_search_results)
    return response, cb


def scrape_article(url):
    scrape_article = []
    # Fetch the article content from the website using BeautifulSoup
    response = requests.get(url)

    # Extract text content from HTML, modify as needed based on the website structure
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all the meta tags in the HTML
        meta_tags = soup.find_all("meta")
        # Get the name or property attribute of the meta tag
        for meta_tag in meta_tags:
            name = meta_tag.get("name") or meta_tag.get("property")
            # Get the content attribute of the meta tag
            content = meta_tag.get("content")
            # Print the name and content of the meta tag
            if name:
                scrape_article.append({name: content})  # Convert the content into a JSON string
        web_search_results = json.dumps(scrape_article)
        return web_search_results
    else:
        print(f"Error: Unable to scrape the URL. Status code: {response.status_code}")
    return


def first_button(url: str):
    if url is not None:
        web_search_results = scrape_article(url)
        output_template = "summarize this content {web_search_results}. and  " \
                          "generated summary output required properties are: " \
                          "1.Suggested 3 Title: " \
                          "2.Tags: " \
                          "3.slugs " \
                          "4.Description(Generate minium 4 lines): " \
                          "5.Synopsis(Generate minium 15 lines): " \
                          "6.Image Title: (give the Image Title based on {web_search_results})" \
                          "7.Image Prompt(Image prompt generated based on {web_search_results} and it is used" \
                          " to generate image by using stability.io):"
        return llm_model(output_template, ["web_search_results"], web_search_results)


def second_button(summary: str):
    output_template = "Summarized existing summary text {summary} as input and " \
                      "regenerate summary in same context with different phrase and sentence" \
                      "but Generated output properties keys same as input"
    return llm_model(output_template, ["summary"], summary)


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    col1, col2 = st.columns(2)
    input = st.empty()
    url = input.text_input(label="Regenerate Article", placeholder="Enter Url", label_visibility="hidden")
    button1 = st.button("Generate summary")
    button2 = st.button("Regenerate summary")
    if url:
        summary, cb1 = first_button(url)
        if button1:
            st.write(summary)
        if button2:
            regenerated_summary, cb2 = second_button(summary)
            image_prompt = regenerated_summary.split("Image Prompt:")[1]
            with col1:
                st.header("Old Article")
                st.write(summary)
            with col2:
                st.header("New Article")
                st.write(regenerated_summary)
                image_path = create_image(image_prompt)
                image_URL = upload_image(image_path)
                st.image(image_URL, width=400, )
                st.write(cb2)

