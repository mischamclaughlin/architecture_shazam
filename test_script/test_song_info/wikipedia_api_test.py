import requests
import mwparserfromhell


HEADERS = {"User-Agent": "ArchitectureShazamScript/0.1 (you@example.com)"}
API_URL = "https://en.wikipedia.org/w/api.php"


def find_page_title(name):
    params = {
        "action": "opensearch",
        "search": name,
        "limit": 1,
        "namespace": 0,
        "format": "json",
    }
    r = requests.get(API_URL, params=params, headers=HEADERS)
    if r.status_code != 200:
        return None
    titles = r.json()[1]
    return titles[0] if titles else None


def fetch_wikitext(title):
    params = {
        "action": "parse",
        "page": title,
        "prop": "wikitext",
        "format": "json",
    }
    r = requests.get(API_URL, params=params, headers=HEADERS)
    if r.status_code != 200:
        return None
    return r.json()["parse"]["wikitext"]["*"]


def get_origin_wikipedia(artist_name):
    title = find_page_title(artist_name)
    if not title:
        return None

    wikitext = fetch_wikitext(title)
    if not wikitext:
        return None

    wd = mwparserfromhell.parse(wikitext)

    infoboxes = [
        tpl
        for tpl in wd.filter_templates()
        if tpl.name.strip().lower().startswith("infobox")
    ]
    if not infoboxes:
        return None

    for tpl in infoboxes:
        for field in ("origin", "birth_place", "birth_place_location"):
            if tpl.has(field):
                raw = tpl.get(field).value
                clean = raw.strip_code().strip()
                return clean

    return None
