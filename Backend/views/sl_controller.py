from django.http import HttpRequest

from Backend.service.sl_service import train_sl


def post_train_sl(request: HttpRequest):
    train_sl()
