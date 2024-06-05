import os
import time
import base64
import requests
from selenium.webdriver.common.by import By
from multiprocessing import Pool, cpu_count
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from utils import load_strings, generate_queries, init_driver


def fetch_images_by_search(search_query):
    driver = init_driver()
    try:
        # Extract the class from the query for subfolder creation
        query_parts = search_query.split()
        cls = query_parts[1] if len(query_parts) > 1 else "unknown_class"

        # Replace spaces with plus signs
        search_query = search_query.replace(" ", "+")

        # Navigate to DuckDuckGo Images
        driver.get(
            f"https://duckduckgo.com/?q={search_query}&t=h_&iar=images&iax=images&ia=images"
        )

        # Find all the images on the page
        images = driver.find_elements(By.CSS_SELECTOR, "img.tile--img__img")

        # Create a directory for the class to store the images
        class_dir = os.path.join("images", cls)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Download and store the images
        for idx, image in enumerate(images[:2]):
            try:
                img_url = image.get_attribute("src")

                # Dealing with base64 encoded images
                if img_url.startswith("data:image"):
                    img_data = img_url.split(",")[1]
                    img_data = base64.b64decode(img_data)
                else:
                    img_data = requests.get(img_url).content
                # Open image and resize
                with BytesIO(img_data) as img_buffer:
                    with Image.open(img_buffer) as img:
                        resized_img = img.resize((64, 64))
                        resized_img.save(
                            f"{class_dir}/{search_query}_image_{idx+1}.jpg"
                        )
            except Exception as e:
                print(
                    f'Error downloading image {idx+1} for the query "{search_query}": {e}'
                )
    except Exception as e:
        print(f'Error searching image for query "{search_query}": {e}')

if __name__ == "__main__":
    # Load queries from the JSON file
    json_file = r"collector\string_data.json"
    adjectives, classes, nouns, contexts = load_strings(json_file)

    # Generate queries
    queries = generate_queries(adjectives, classes, nouns, contexts)

    # Define the number of processes to be used
    num_processes = cpu_count()

    # Create a pool of processes and execute the queries in parallel
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap(fetch_images_by_search, queries), total=len(queries), desc="Processing Queries"):
            pass
