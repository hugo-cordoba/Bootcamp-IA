from apify_client import ApifyClient

# Initialize the ApifyClient with your Apify API token
client = ApifyClient("apify_api_tIlQhfH6kfksfM03YFxnGpZRr2PGpQ2vEvqi")


def load_comments(instagram_url):
    comments = []
    # Prepare the Actor input
    run_input = {
        "directUrls": [instagram_url],
        "resultsType": "posts",
        "resultsLimit": 200,
        "searchType": "hashtag",
        "searchLimit": 1,
    }

    # Run the Actor and wait for it to finish
    run = client.actor("apify/instagram-scraper").call(run_input=run_input)

    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        for dict_comments in item['latestComments']:            
            comments.append(dict_comments.get('text'))
    return comments
        
