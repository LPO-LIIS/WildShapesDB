import json
import os
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from PIL import Image
from io import BytesIO
import random
import numpy as np


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def init_driver():
    user_agents = [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1 Safari/605.1.15',
      'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.120 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Safari/604.1.38',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134',
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
      'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.167 Safari/537.36',
      'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:59.0) Gecko/20100101 Firefox/59.0',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:62.0) Gecko/20100101 Firefox/62.0',
      'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:62.0) Gecko/20100101 Firefox/62.0',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36',
      'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
      'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36',
      'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:41.0) Gecko/20100101 Firefox/41.0',
      'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:45.0) Gecko/20100101 Firefox/45.0',
      'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64; Trident/7.0; AS; rv:11.0) like Gecko',
      'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:44.0) Gecko/20100101 Firefox/44.0',
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) Chrome/12.0.742.112 Safari/534.30',
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:63.0) Gecko/20100101 Firefox/63.0',
      'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:64.0) Gecko/20100101 Firefox/64.0'
    ]

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--disable-features=InterestCohort")
    #options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--log-level=3')
    options.add_argument("--silent")

    # Escolha um User-Agent aleatório
    user_agent = random.choice(user_agents)
    options.add_argument(f'user-agent={user_agent}')
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Redirecionar saída do ChromeDriver para um arquivo nulo
    service = Service()
    service.creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # No Windows, oculta a janela de terminal
    
    # Redireciona saída de logs para os.devnull
    service.log_output = open(os.devnull, 'w')

    # Inicializar o WebDriver
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def load_strings(json_file):
    """
    Loads adjectives, classes, and modifiers from a JSON file.

    Parameters:
    json_file (str): The path to the JSON file containing the strings.

    Returns:
    tuple: A tuple containing three lists:
        - adjectives (list): A list of adjectives.
        - classes (dict): A dictionary where keys are class names and values are lists of related nouns.
        - modifiers (list): A list of modifiers for additional context.
    """
    with open(json_file, "r") as file:
        data = json.load(file)
        adjectives = data.get("adjectives", [])
        classes = data.get("classes", {})  # Now a dictionary {class_name: [nouns]}
        modifiers = data.get("modifiers", [])
    
    return adjectives, classes, modifiers


def generate_queries(adjectives, classes, modifiers, num_queries):
    """
    Generates queries by combining adjectives, classes, nouns, and optional modifiers.

    Parameters:
    adjectives (list): A list of adjectives.
    classes (dict): A dictionary where keys are class names and values are lists of nouns.
    modifiers (list): A list of contextual modifiers.
    num_queries (int): The number of queries to generate.

    Returns:
    list: A list of generated queries, each being a combination of an adjective,
          a class, and a noun, with an optional modifier.
    """
    queries = set()  # Using a set to ensure unique queries
    class_names = list(classes.keys())  # Extract class names from the dictionary

    while len(queries) < num_queries:
        cls = random.choice(class_names)  # Select a geometric class
        nouns = classes[cls]  # Get nouns associated with this class
        if not nouns:
            continue  # Skip if no nouns are available

        noun = random.choice(nouns)  # Pick a noun from the class
        adjective = random.choice(adjectives)
        modifier = random.choice(modifiers) if random.random() < 0.5 else ""  # 50% chance of adding a modifier

        query = f"{adjective}+{cls}+{noun}" + (f"+{modifier}" if modifier else "")
        queries.add(query)

    return list(queries)


def preprocess_image(image_data):
    # Open the image with Pillow
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Resize the image to 128x128
    image = image.resize((128, 128))

    # Return the preprocessed image as bytes
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
