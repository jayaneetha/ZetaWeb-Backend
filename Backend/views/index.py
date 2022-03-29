from django.http import JsonResponse, HttpRequest
from django.urls import get_resolver


def index(request: HttpRequest):
    urls = []
    for up in get_resolver().url_patterns[1].url_patterns:
        urls.append(str(up.pattern))

    response = {
        "urls:": urls
    }
    return JsonResponse(response, safe=False)
