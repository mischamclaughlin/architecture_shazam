# ./flask_server/modules/services/wiki_service.py
import requests
from flask import current_app
import mwparserfromhell


class WikiService:
    def __init__(self, artist_name: str) -> None:
        self.api_url = current_app.config["APP_SETTINGS"].WIKI_API_URL
        self.headers = {
            "User-Agent": current_app.config["APP_SETTINGS"].WIKI_USER_AGENT
        }
        self.artist_name = artist_name

    def find_page_title(self):
        params = {
            "action": "opensearch",
            "search": self.artist_name,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }
        r = requests.get(self.api_url, params=params, headers=self.headers)
        if r.status_code != 200:
            return None

        titles = r.json()[1]
        return titles[0] if titles else None

    def fetch_wikitext(self, title):
        params = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "format": "json",
        }
        r = requests.get(self.api_url, params=params, headers=self.headers)
        if r.status_code != 200:
            return None

        return r.json()["parse"]["wikitext"]["*"]

    def get_origin_wikipedia(self):
        title = self.find_page_title()
        if not title:
            return None

        wikitext = self.fetch_wikitext(title)
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
