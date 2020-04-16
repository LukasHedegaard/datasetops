import json
from pathlib import Path
from typing import Callable, IO
import os


class Cache:
    DEFAULT_PATH: Path = Path(".datasetops_cache")

    @staticmethod
    def clear(path: Path = None):

        if path is None:
            path = Cache.DEFAULT_PATH

        if not path.is_dir():
            print(f"No cache at: {path}")
            return

        files = os.listdir(path)
        files = list(
            filter(lambda name: ("database.json" in name or "cache_" in name), files)
        )

        for file in files:
            os.remove(path / file)

        if len(files) > 0:
            print("Cleared " + str(len(files) - 1) + " cache entries")
        else:
            print(f"No cache at: {path}")

    def __init__(self, path: Path) -> None:

        if path is None:
            path = Cache.DEFAULT_PATH

        self.path = Path(path)

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)
            with open(Path(self.path) / ".gitignore", "w") as file:
                file.write("# Created by datasetops automatically. \n*\n")

        self.database_path = self.path / "database.json"
        self.database = self.__load_database()

    def __load_database(self):
        try:
            with open(self.database_path, "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"entries": [], "last_id": 0}

    def __save_database(self):
        with open(self.database_path, "w") as file:
            return json.dump(self.database, file)

    def __cache_file_path(self, cache_id):
        return Path(self.path) / ("cache_" + str(cache_id) + ".pkl")

    def is_cached(self, identifier):
        for entry in self.database["entries"]:
            if entry["identifier"] == identifier:
                return True

        return False

    def save(self, identifier, saver: Callable[[IO], bool]):
        if self.is_cached(identifier):
            raise Exception("Already cached for identifier=" + identifier)

        cache_id = self.database["last_id"] + 1

        entry = {
            "cache_id": cache_id,
            "identifier": identifier,
        }

        self.database["last_id"] = cache_id
        self.database["entries"].append(entry)

        file_path = self.__cache_file_path(cache_id)

        with open(file_path, "wb") as file:
            while saver(file):
                pass

        self.__save_database()

    def create_stream(self, identifier):
        if not self.is_cached(identifier):
            raise Exception("No cache for identifier=" + identifier)

        entry = next(
            x for x in self.database["entries"] if x["identifier"] == identifier
        )

        cache_id = entry["cache_id"]
        file_path = self.__cache_file_path(cache_id)

        file = open(file_path, "rb")

        return file
