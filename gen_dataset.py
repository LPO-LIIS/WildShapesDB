from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collector.scrapper import fetch_images_by_search
from collector.utils import load_strings, generate_queries, set_random_seed

if __name__ == "__main__":
    set_random_seed(42)
    # Load queries from the search_query_data JSON file
    json_file = "search_query_data.json"
    adjectives, classes, nouns = load_strings(json_file)

    # Generate queries
    num_queries = 3000
    queries = generate_queries(adjectives, classes, nouns, num_queries)

    # Define the number of processes to be used
    num_processes = min(cpu_count() // 2, 8)

    # Create a pool of processes and execute the queries in parallel
    with Pool(num_processes) as pool:
        for _ in tqdm(pool.imap(fetch_images_by_search, queries), total=len(queries), desc="Processing Queries"):
            pass