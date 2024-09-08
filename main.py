import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from dotenv import dotenv_values
from huggingface_hub import login

config = dotenv_values('.env')

login(token=config['HG_TOKEN'])

# Load the LLaMA model (LLaMA 3) from Hugging Face
# You can replace `model_name` with the actual LLaMA 3 model when available.
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with LLaMA 3 model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate text using LLaMA (inference with Hugging Face Transformers)
def generate_search_query(item):
    # Define the prompt for the model
    prompt = f"Generate a search query for finding the best {item} online. Provide a clear and useful query."

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response using the model
    outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)

    # Decode the generated tokens to a string
    generated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_query

# Function to search shopping items via API or web scraping
def search_shopping_items(item):
    """
    Search shopping items from a mock API or any real e-commerce API.
    In this case, let's assume an API request.
    """
    api_url = f"https://api.mockshopping.com/search?query={item}"

    try:
        response = requests.get(api_url)
        results = response.json()  # Parse the results
        return results  # Return results as JSON
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

# Define a filtering function based on user input
def filter_results(results, filter_criteria):
    """
    Filter the shopping results based on user input.
    Example filter criteria: price, rating, or category.
    """
    filtered_results = []

    for result in results:
        if (
            filter_criteria.get('min_price', 0) <= result['price'] <= filter_criteria.get('max_price', float('inf')) and
            filter_criteria.get('min_rating', 0) <= result['rating']
        ):
            filtered_results.append(result)

    return filtered_results

def main():
    # Step 1: User input
    user_item = input("Enter the item you want to search for: ")
    user_min_price = float(input("Enter the minimum price: "))
    user_max_price = float(input("Enter the maximum price: "))
    user_min_rating = float(input("Enter the minimum rating: "))

    # Step 2: Use LLaMA (via Hugging Face) to generate search query
    search_query = generate_search_query(user_item)
    print(f"Generated search query: {search_query}")
    
    # Step 3: Search for the item using API or scraping
    search_results = search_shopping_items(user_item)
    
    if not search_results:
        print("No results found.")
        return
    
    # Step 4: Filter results based on user input
    filter_criteria = {
        "min_price": user_min_price,
        "max_price": user_max_price,
        "min_rating": user_min_rating,
    }

    filtered_results = filter_results(search_results, filter_criteria)

    # Step 5: Display the filtered results
    if filtered_results:
        print(f"Filtered results for {user_item}:")
        for result in filtered_results:
            print(f"Item: {result['name']}, Price: {result['price']}, Rating: {result['rating']}, URL: {result['url']}")
    else:
        print("No items match your criteria.")

if __name__ == "__main__":
    main()