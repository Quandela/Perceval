"""build the index.html for multiversion docs"""
import os
from perceval import PMetadata

DIR_SCRIPT = os.path.dirname(__file__)
INDEX_OF_HTML = os.path.join(DIR_SCRIPT, "../../build/html/index.html")

LATEST_TAG = f"v{PMetadata.version()}"

TEMPLATE = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>Redirecting to latest branch</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="0; url=./{LATEST_TAG}">
        <link rel="canonical" href="{LATEST_TAG}">
    </head>
    <body>
        <p>Redirecting you to <a href="./{LATEST_TAG}">{LATEST_TAG}</a></p>
    </body>
</html>
"""

with open(INDEX_OF_HTML, "w", encoding="utf-8") as file:
    file.write(TEMPLATE)
