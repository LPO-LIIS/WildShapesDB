import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from multiprocessing import Pool

# Configurações do Selenium
def init_driver():
    options = Options()
    options.headless = True  # Executa o navegador em modo headless (sem interface gráfica)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

# Função para buscar e baixar imagens
def fetch_images(query):
    driver = init_driver()
    try:
        # Realiza a busca no Google Imagens
        driver.get('https://images.google.com/')
        search_box = driver.find_element(By.NAME, 'q')
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        time.sleep(2)  # Aguarda os resultados carregarem

        # Filtra por licença livre para uso
        tools_button = driver.find_element(By.XPATH, "//div[@aria-label='Ferramentas']")
        tools_button.click()
        time.sleep(1)
        usage_rights_button = driver.find_element(By.XPATH, "//span[text()='Direitos de uso']")
        usage_rights_button.click()
        time.sleep(1)
        creative_commons_button = driver.find_element(By.XPATH, "//span[text()='Creative Commons licenses']")
        creative_commons_button.click()
        time.sleep(2)  # Aguarda os resultados filtrados

        # Captura a primeira linha de imagens
        images = driver.find_elements(By.XPATH, "//div[@id='islrg']//div[@class='islrc']/div[@class='isv-r']")
        first_row_images = images[:5]  # Supondo que a primeira linha tenha 5 imagens

        # Cria um diretório para armazenar as imagens
        if not os.path.exists('images'):
            os.makedirs('images')

        # Baixa e armazena as imagens
        for idx, img in enumerate(first_row_images):
            try:
                img_url = img.find_element(By.TAG_NAME, 'img').get_attribute('src')
                img_data = requests.get(img_url).content
                with open(f'images/{query}_image_{idx+1}.jpg', 'wb') as handler:
                    handler.write(img_data)
                print(f'Imagem {idx+1} baixada com sucesso para a query "{query}".')
            except Exception as e:
                print(f'Erro ao baixar a imagem {idx+1} para a query "{query}": {e}')
    finally:
        driver.quit()

if __name__ == "__main__":
    # Lista de consultas (queries)
    queries = [f'query_{i}' for i in range(10000)]

    # Define o número de processos a serem utilizados
    num_processes = 8  # Ajuste este valor conforme necessário

    # Cria um pool de processos e executa as consultas em paralelo
    with Pool(num_processes) as pool:
        pool.map(fetch_images, queries)
