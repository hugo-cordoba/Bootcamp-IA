from apify_client import ApifyClient

# Initialize the ApifyClient with your Apify API token
client = ApifyClient("apify_api_tIlQhfH6kfksfM03YFxnGpZRr2PGpQ2vEvqi")

# Prepare the Actor input
run_input = {
    "directUrls": ["https://www.instagram.com/p/C4S4s-lIOWw/"],
    "resultsType": "posts",
    "resultsLimit": 200,
    "searchType": "hashtag",
    "searchLimit": 1,
}

# Run the Actor and wait for it to finish
run = client.actor("apify/instagram-scraper").call(run_input=run_input)

# Fetch and print Actor results from the run's dataset (if there are any)
print("💾 Check your data here: https://console.apify.com/storage/datasets/" + run["defaultDatasetId"])
for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    for dict_comments in item['latestComments']:
        print(dict_comments.get('text'))
    
