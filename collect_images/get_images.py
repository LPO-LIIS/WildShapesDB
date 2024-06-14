import os
import requests
from selenium.webdriver.common.by import By
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

from utils import load_strings, generate_queries, init_driver, set_random_seed


def fetch_images_by_search(search_query, max_retries=2, delay=2):
    for attempt in range(max_retries):
        driver = init_driver()
        try:
            # Extract the class from the query for subfolder creation
            query_parts = search_query.split('+')
            cls = query_parts[1]

            # Navigate to DuckDuckGo Images
            driver.get(
                f"https://duckduckgo.com/?q={search_query}&t=h_&iar=images&iax=images&ia=images"
            )

            # Wait for the page to load
            driver.implicitly_wait(2)

            # Find all the images on the page
            images = driver.find_elements(By.CSS_SELECTOR, "img.tile--img__img")

            # Create a directory for the class to store the images
            class_dir = os.path.join("images", cls)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Download and store the images
            for i, image in enumerate(images[:24]):
                img_url = image.get_attribute("src")
                try:
                  if img_url.startswith("http"):
                      img_data = requests.get(img_url).content
                      with open(os.path.join(class_dir, f"{search_query}_{i}.jpg"), 'wb') as handler:
                          handler.write(img_data)
                except Exception as e:
                    print(f"Error downloading image {search_query}_{i}: {e}")
            break  # Se o código chegou até aqui, não houve erro, então sair do loop
        except Exception as e:
            print(f'Error searching image for query "{search_query}" on attempt {attempt + 1}/{max_retries}: {e}')
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # Espera exponencial
        finally:
            driver.quit()

if __name__ == "__main__":
    set_random_seed(42)
    # Load queries from the JSON file
    json_file = r"collector\string_data.json"
    adjectives, classes, nouns = load_strings(json_file)

    # Generate queries
    num_queries = 10000
    queries = generate_queries(adjectives, classes, nouns, num_queries)

    # Define the number of processes to be used
    num_processes = cpu_count() // 2

    # Create a pool of processes and execute the queries in parallel
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap(fetch_images_by_search, queries), total=len(queries), desc="Processing Queries"):
            pass