import os
import csv
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from tqdm import tqdm


from loguru import logger


USERS_URL = "https://huggingface.co/users"
ORG_URL = "https://huggingface.co/organizations"

GH_TOKEN = os.getenv('GH_TOKEN')

def get_users_accounts(url=USERS_URL):
    '''
    Returns a list of users github links from the huggingface PRO accounts
    '''
    users_gh_accounts = []

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    spans = soup.find_all('span', class_='rounded bg-gray-200 px-0.5 font-mono text-xs text-gray-700 dark:bg-gray-800')

    logger.info(f'Extracting emails from {len(spans)} users')
    users_hf_accounts = [username.text for username in spans]

    hf_profile_link_list = [f"https://huggingface.co/{username.text}" for username in spans]

    with open(f"users_hf_accounts.txt", "w") as file:
        for account in users_hf_accounts:
            file.write(f"{account}\n")
    
    return hf_profile_link_list


def get_gh_link(url):
    response = requests.get(url)
    content = response.content

    soup = BeautifulSoup(content, 'html.parser')

    social_media_links = soup.find_all('a', class_='truncate text-gray-600 hover:underline')

    gh_links = []
    for link in social_media_links:
        if link["href"].startswith("https://github.com/"):
            gh_links.append(link["href"])

    if len(gh_links) == 0:
        return None
    elif len(gh_links) > 1:
        return gh_links[0]

    return gh_links


def get_gh_links_from_hf_profiles(hf_profile_link_list):
    '''
    Returns a list of github links from the huggingface PRO accounts
    '''
    gh_link_list = []
    logger.info(f"Extracting github links from {len(hf_profile_link_list)} profiles")
    for hf_profile_link in tqdm(hf_profile_link_list):
        # logger.debug(f"Extracting github link from {hf_profile_link}")
        gh_link = get_gh_link(hf_profile_link)
        gh_link_list.append(gh_link)
    return gh_link_list



def get_email(username, GH_TOKEN):
    '''
    Get the email from a user's latest GitHub commits
    '''
    
    headers = {
        'Authorization': f'token {GH_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }

    # Get user repositories, excluding forks
    repo_url = f'https://api.github.com/users/{username}/repos?type=owner'
    response = requests.get(repo_url, headers=headers)
    repos = response.json()

    if not repos or isinstance(repos, dict) and 'message' in repos:
        return None

    latest_email = None
    latest_date = None  # Will store the datetime of the latest commit

    # Iterate through repositories
    for repo in repos:
        if repo['fork']:  # Skip forked repositories
            continue

        # Construct the commits API endpoint
        url = f"https://api.github.com/repos/{username}/{repo['name']}/commits"
        response = requests.get(url, headers=headers)
        commits = response.json()

        # Extract and add emails from commits
        for commit in commits:
            if 'commit' not in commit:
                continue  # Skip if there is no commit information

            commit_date_str = commit['commit']['author']['date']
            commit_date = datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%SZ')

            # Check if this commit is the latest we have found so far
            if latest_date is None or commit_date > latest_date:
                latest_date = commit_date  # Update the latest commit date
                author_email = commit['commit']['author'].get('email')
                if author_email and 'noreply' not in author_email:
                    latest_email = author_email  # Update the latest email

    return latest_email


def get_PRO_emails():
    '''
    Returns a list of emails from the huggingface PRO accounts
    '''
    email_list = []

    users_profile_link_list = get_users_accounts()
    gh_link_list = get_gh_links_from_hf_profiles(users_profile_link_list)
    for github_url in tqdm(gh_link_list):
        if github_url is not None:
            username = github_url[0].split('/')[-1]
            user_email = get_email(username, GH_TOKEN)
            if user_email:  # Check if new_emails is not empty
                email_list.append(user_email)

    logger.success(f"Get {len(email_list)} emails from PRO accounts!")

    with open(f"pro_emails.txt", "w") as file:
        for email in email_list:
            file.write(f"{email}\n")
    return email_list



def get_ORG_list():
    org_list = []

    page = 1  # Start from the first page
    while True:  # Keep looping until there are no more pages
        logger.info(f"Extracting ORG from page {page}")
        response = requests.get(f"{ORG_URL}?p={page}")  # Ensure the query parameter is correct for pagination
        soup = BeautifulSoup(response.text, 'html.parser')

        org_containers = soup.find_all('article', class_='overview-card-wrapper')
        
        if len(org_containers) == 0:
            logger.success(f"Reached the last ORG page: {page}")
            break

        for container in org_containers:
            # Extract the organization's URL
            org_url = container.find('a').get('href')
            
            # Extract the organization's name, assuming it's contained within an <h4> tag or similar
            org_name = container.find('h4').text.strip() if container.find('h4') else "No Name Found"
            
            # Extract the organization's type, assuming it's denoted by a <span> with text like "Community" or "Enterprise"
            org_type = container.find('span', class_='capitalize').text.strip() if container.find('span', class_='capitalize') else "No Type Found"
            
            org_list.append((org_name, org_url, org_type))
        
        page += 1  # Move to the next page

    logger.info(f'Extracting orgs from {len(org_list)} organizations')

    # Save org details to a CSV file
    with open("org_list.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Organization Name", "URL", "Type"])  # Header row
        for org in org_list:
            writer.writerow(org)
            
    return org_list


def get_ORG_members(org_url):
    # Get member profiles from each organization
    response = requests.get(org_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    nav_element = soup.find('nav', class_='mb-6 mt-3 flex flex-wrap xl:pr-6 2xl:pr-12')  # Adjust class name as needed
    a_tags = nav_element.find_all('a') if nav_element else []
    org_members = [a_tag.get('href')[1:] for a_tag in a_tags]
    return org_members


def get_ORG_github_links():
    # Filter out the organizations that are not of interest
    ORG_TYPE_LIST = ["community", "company", "non-profit"]

    org_urls = []
    org_member_urls = []

    with open("org_list.csv", "r", newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # Skip the header row
        for org_name, org_href, org_type in reader:
            if org_type.strip() in ORG_TYPE_LIST:  # Use strip() to remove any leading/trailing whitespace
                org_urls.append(f"https://huggingface.co{org_href.strip()}")  # Ensure org_href is stripped of whitespace

    with open("org_urls.txt", "w") as file:
        for url in org_urls:
            file.write(f"{url}\n")

    
    for org_url in tqdm(org_urls, desc="Processing organizations"):
        org_members = get_ORG_members(org_url)
        logger.info(f"Extracting {len(org_members)} members' emails from {org_url}")
        
        # Process each member found for the current organization
        for org_member in tqdm(org_members, desc="Processing members"):
            org_member_url = f"https://huggingface.co/{org_member}"
            org_member_urls.append(org_member_url)

            # Append each member URL to the file as it's processed
            with open("org_hf_accounts.txt", "a") as file:  # Open in append mode
                file.write(f"{org_member_url.split('/')[-1]}\n")
    
    return org_member_urls


def get_ORG_emails():
    '''
    Returns a list of emails from the Hugging Face ORG accounts.
    '''

    
    email_list = []

    if not os.path.exists("org_list.csv"):
        org_list = get_ORG_list()

    if not os.path.exists("org_hf_accounts.txt"):
        org_member_urls = get_ORG_github_links()
    else:
        with open("org_hf_accounts.txt", "r") as file:
            org_member_urls = file.readlines()
            org_hf_accounts = [url.strip() for url in org_member_urls]
            org_member_urls = [f"https://huggingface.co/{org_hf_account}" for org_hf_account in org_hf_accounts]

    logger.debug(f"Extracting github links from {len(org_member_urls)} profiles")

    gh_link_list = get_gh_links_from_hf_profiles(org_member_urls)

    for github_url in gh_link_list:
        # logger.debug(f"Extracting email from {github_url}")
        if type(github_url) == list:
            github_url = github_url[0]
        # logger.debug(f"Extracting email from {github_url}")
        if github_url:
            username = github_url.split('/')[-1]
            user_email = get_email(username, GH_TOKEN)
            if user_email:
                email_list.append(user_email)


    logger.success(f"Get {len(email_list)} emails from ORG accounts!")

    with open("org_emails.txt", "w") as file:
        for email in email_list:
            file.write(f"{email}\n")

    return email_list




if __name__ == "__main__":
    # get_PRO_emails()
    get_ORG_emails()
       