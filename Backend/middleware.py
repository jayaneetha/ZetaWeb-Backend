import logging

from django.http import HttpRequest, HttpResponse, HttpResponseNotAllowed
from django.urls import resolve


class MethodCheckMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest):

        logging.log(logging.INFO, [request.method, request.path])

        resolver = resolve(request.path)
        resolver_view_name = resolver.view_name.split(".")[-1]

        is_rest: bool = False
        for method in ['get', 'post', 'put', 'delete']:
            if str(resolver_view_name).lower().startswith(method):
                is_rest = True

        if is_rest and not str(resolver_view_name).lower().startswith(str(request.method).lower()):
            response = HttpResponseNotAllowed(permitted_methods=[str(resolver_view_name).upper()])
        else:
            response: HttpResponse = self.get_response(request)

        logging.log(logging.INFO, [request.method, request.path, response.status_code])

        return response
