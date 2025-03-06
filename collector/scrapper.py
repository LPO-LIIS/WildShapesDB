import os
import requests
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

from collector.utils import init_driver


# Simular rolagem da página para carregar mais imagens
def scroll_down(driver, times=3, delay=2):
    for _ in range(times):
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        time.sleep(delay)


def store_images(images, class_dir: str, search_query: str):
    # Download and store the images
    for i, image in enumerate(images[:50]):
        img_url = image.get_attribute("src")
        try:
            if img_url.startswith("http"):
                img_data = requests.get(img_url).content
                with open(
                    os.path.join(class_dir, f"{search_query}_{i}.jpg"), "wb"
                ) as handler:
                    handler.write(img_data)
        except Exception as e:
            print(f"Error downloading image {search_query}_{i}: {e}")


def fetch_images_by_search(search_query: str, max_retries=2, delay=2):
    driver = init_driver()
    # Extract the class from the query for subfolder creation
    query_parts = search_query.split("+")
    cls = query_parts[1]

    # Create a directory for the class to store the images
    class_dir = os.path.join("WildShapesDataset/images", cls)
    os.makedirs(class_dir, exist_ok=True)

    for attempt in range(max_retries):
        try:
            # Navigate to DuckDuckGo Images
            driver.get(
                f"https://duckduckgo.com/?q={search_query}&t=h_&iar=images&iax=images&ia=images"
            )

            # Wait for the page to load
            driver.implicitly_wait(1)

            # Rola a página para carregar mais imagens
            scroll_down(driver, times=5, delay=2)

            # Find all the images on the page
            images = driver.find_elements(By.CSS_SELECTOR, "img.tile--img__img")

            store_images(images, class_dir, search_query)

            break  # Se o código chegou até aqui, não houve erro, então sair do loop
        except Exception as e:
            print(
                f'Error searching image for query "{search_query}" on attempt {attempt + 1}/{max_retries}: {e}'
            )
            time.sleep(delay * (2**attempt))  # Espera exponencial

    # Fecha o navegador
    driver.quit()
