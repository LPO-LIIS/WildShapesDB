import json
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium_stealth import stealth
from PIL import Image
from io import BytesIO
import os

def init_driver():
    # Initialize Chrome's WebDriver
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--log-level=3")  # Suppress logs from ChromeDriver
    options.add_experimental_option('excludeSwitches', ['enable-logging'])  # Suppress DevTools logs

    # Suppress DevTools listening message by redirecting stdout and stderr temporarily
    with open(os.devnull, 'w') as devnull:
        original_stdout = os.dup(1)
        original_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Restore stdout and stderr
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(original_stdout)
        os.close(original_stderr)
        
        stealth(driver,
                languages=["en-US", "en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
                )
    return driver

def load_strings(json_file):
    """
    Loads adjectives, classes, nouns, and contexts from a JSON file.

    Parameters:
    json_file (str): The path to the JSON file containing the strings.

    Returns:
    tuple: A tuple containing four lists:
        - adjectives (list): A list of adjectives.
        - classes (list): A list of classes.
        - nouns (list): A list of nouns.
        - contexts (list): A list of contexts.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
        adjectives = data.get('adjectives', [])
        classes = data.get('classes', [])
        nouns = data.get('nouns', [])
        contexts = data.get('contexts', [])
    return adjectives, classes, nouns, contexts

def generate_queries(adjectives, classes, nouns, contexts):
    """
    Generates queries by combining adjectives, classes, nouns, and contexts.

    Parameters:
    adjectives (list): A list of adjectives.
    classes (list): A list of classes.
    nouns (list): A list of nouns.
    contexts (list): A list of contexts.

    Returns:
    list: A list of generated queries, each being a combination of an adjective, 
          a class, a noun, and a context.
    """
    queries = []
    for adjective in adjectives:
        for cls in classes:
            for noun in nouns:
                for context in contexts:
                    queries.append(f"{adjective} {cls} {noun} {context}")
    return queries

def preprocess_image(image_data):
    # Open the image with Pillow
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Resize the image to 128x128
    image = image.resize((128, 128))

    # Return the preprocessed image as bytes
    buffered = BytesIO()
    image.save(buffered, format="JPEG")