import os
from perceval import PMetadata

DIRECTORY_OF_SRIPT = os.path.dirname(__file__)
INDEX_OF_HTML = os.path.join(DIRECTORY_OF_SRIPT, "../../build/html/index.html")

latest_tag = f"v{PMetadata.version()}"

TEMPLATE = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>Redirecting to latest branch</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="0; url=./{latest_tag}">
        <link rel="canonical" href="{latest_tag}">
    </head>
    <body>
        <p>Redirecting you to <a href="./{latest_tag}">{latest_tag}</a></p>
    </body>
</html>
"""

with open(INDEX_OF_HTML, "w", encoding="utf-8") as file:
    file.write(TEMPLATE)
